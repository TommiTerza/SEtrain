"""
Independent block-wise distillation for GTCRN Conv/Deconv blocks.

Goal:
- Train each selected convolutional block as its own standalone network.
- Each block sees exactly the tensor that the corresponding teacher block sees.
- ConvBlock units learn teacher block output.
- GTConv units learn teacher `point_bn2` output (conv branch only), excluding TRA/shuffle.
- Optional progressive mode feeds each block with inputs from a progressively
  updated student context model instead of teacher-only inputs.
- After all blocks are trained independently, re-assemble a single student model
  by inserting trained blocks and then copy GRU weights from teacher (optional).

This script is intentionally different from end-to-end KD in `train.py`.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.ao.quantization as quant
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import DNS3Dataset as Dataset
from models.gtcrn_end2end import GTCRN as Model, ConvBlock, GTConvBlock
from scheduler import LinearWarmupCosineAnnealingLR as WarmupLR

seed = 43
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class SingleBlockIOCapture:
    """
    Capture teacher input/output for one target block.

    - Input is taken from a forward pre-hook.
    - Output is taken from a forward hook.
    - Both are detached to keep teacher fully frozen.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_module_name: str,
        target_module_name: str,
    ):
        self.input_module_name = input_module_name
        self.target_module_name = target_module_name
        self.block_input: torch.Tensor | None = None
        self.block_output: torch.Tensor | None = None

        named_modules = dict(model.named_modules())
        if input_module_name not in named_modules:
            raise RuntimeError(f"Input module '{input_module_name}' not found in teacher model.")
        if target_module_name not in named_modules:
            raise RuntimeError(f"Target module '{target_module_name}' not found in teacher model.")

        input_module = named_modules[input_module_name]
        target_module = named_modules[target_module_name]

        self.pre_handle = input_module.register_forward_pre_hook(self._pre_hook)
        self.post_handle = target_module.register_forward_hook(self._post_hook)

    def _pre_hook(self, _module, args):
        if not args:
            raise RuntimeError(f"Teacher input module '{self.input_module_name}' got no positional input.")
        x = args[0]
        if isinstance(x, (tuple, list)):
            if not x:
                raise RuntimeError(f"Teacher input module '{self.input_module_name}' got empty tuple/list input.")
            x = x[0]
        if not torch.is_tensor(x):
            raise RuntimeError(f"Teacher input module '{self.input_module_name}' input is not a tensor.")
        self.block_input = x.detach()

    def _post_hook(self, _module, _args, output):
        if isinstance(output, (tuple, list)):
            if not output:
                raise RuntimeError(f"Teacher target module '{self.target_module_name}' returned empty tuple/list output.")
            output = output[0]
        if not torch.is_tensor(output):
            raise RuntimeError(f"Teacher target module '{self.target_module_name}' output is not a tensor.")
        self.block_output = output.detach()

    def clear(self) -> None:
        self.block_input = None
        self.block_output = None

    def remove(self) -> None:
        self.pre_handle.remove()
        self.post_handle.remove()
        self.clear()


class GTConvConvPath(torch.nn.Module):
    """
    Standalone GTConv convolutional branch only.

    This module reproduces the teacher GTConv path up to `point_bn2`:
      input -> x1 split -> SFE -> point/depth/point conv stack -> point_bn2
    It explicitly excludes:
      - TRA (GRU attention)
      - channel shuffle with x2
    """

    def __init__(self, source_block: GTConvBlock):
        super().__init__()
        self.pad_size = source_block.pad_size
        self.sfe = deepcopy(source_block.sfe)

        self.point_quant1 = deepcopy(source_block.point_quant1)
        self.point_conv1 = deepcopy(source_block.point_conv1)
        self.point_dequant1 = deepcopy(source_block.point_dequant1)
        self.point_bn1 = deepcopy(source_block.point_bn1)
        self.point_act = deepcopy(source_block.point_act)

        self.point_quant2 = deepcopy(source_block.point_quant2)
        self.depth_conv = deepcopy(source_block.depth_conv)
        self.point_dequant2 = deepcopy(source_block.point_dequant2)
        self.depth_bn = deepcopy(source_block.depth_bn)
        self.depth_act = deepcopy(source_block.depth_act)

        self.point_quant3 = deepcopy(source_block.point_quant3)
        self.point_conv2 = deepcopy(source_block.point_conv2)
        self.point_dequant3 = deepcopy(source_block.point_dequant3)
        self.point_bn2 = deepcopy(source_block.point_bn2)

    def forward(self, x):
        x1, _x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_quant1(x1)
        h1 = self.point_conv1(h1)
        h1 = self.point_dequant1(h1)
        h1 = self.point_bn1(h1)
        h1 = self.point_act(h1)

        h1 = F.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.point_quant2(h1)
        h1 = self.depth_conv(h1)
        h1 = self.point_dequant2(h1)
        h1 = self.depth_bn(h1)
        h1 = self.depth_act(h1)

        h1 = self.point_quant3(h1)
        h1 = self.point_conv2(h1)
        h1 = self.point_dequant3(h1)
        h1 = self.point_bn2(h1)
        return h1


def _extract_model_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format. Expected dict-like checkpoint.")

    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError("Could not find model weights in checkpoint (expected 'model' or 'state_dict').")


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = _extract_model_state_dict(checkpoint)

    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value

    missing_keys, unexpected_keys = model.load_state_dict(cleaned, strict=False)
    if missing_keys:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is incompatible. Missing keys: {missing_keys[:10]}"
        )

    ignored_prefixes = ("activation_post_process", "weight_fake_quant")
    bad_unexpected = [k for k in unexpected_keys if not any(pfx in k for pfx in ignored_prefixes)]
    if bad_unexpected:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' has unexpected keys: {bad_unexpected[:10]}"
        )


def _copy_gru_from_teacher(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    freeze: bool = False,
    skip_prefixes: tuple[str, ...] = (),
) -> tuple[int, int]:
    def _is_skipped(module_name: str) -> bool:
        return any(module_name.startswith(prefix) for prefix in skip_prefixes)

    student_grus = {
        name: module
        for name, module in student_model.named_modules()
        if isinstance(module, torch.nn.GRU) and not _is_skipped(name)
    }
    if not student_grus:
        return 0, 0

    teacher_grus = {
        name: module
        for name, module in teacher_model.named_modules()
        if isinstance(module, torch.nn.GRU) and not _is_skipped(name)
    }
    missing = [name for name in student_grus if name not in teacher_grus]
    if missing:
        raise RuntimeError(f"Teacher missing GRU layers required by student: {missing[:10]}")

    total_params = 0
    for name, student_gru in student_grus.items():
        student_gru.load_state_dict(teacher_grus[name].state_dict(), strict=True)
        if freeze:
            for parameter in student_gru.parameters():
                parameter.requires_grad_(False)
                total_params += parameter.numel()
        else:
            total_params += sum(p.numel() for p in student_gru.parameters())

    return len(student_grus), total_params


def _sanitize_name(name: str) -> str:
    return name.replace(".", "_")


def _resolve_selected_blocks(
    model: torch.nn.Module,
    train_encoder: bool,
    train_decoder: bool,
    requested_blocks,
) -> list[str]:
    candidate = []
    for name, module in model.named_modules():
        if not isinstance(module, (ConvBlock, GTConvBlock)):
            continue
        is_encoder = name.startswith("encoder.")
        is_decoder = name.startswith("decoder.")
        if is_encoder and not train_encoder:
            continue
        if is_decoder and not train_decoder:
            continue
        candidate.append(name)

    if not candidate:
        raise RuntimeError("No Conv/Deconv blocks available after current encoder/decoder filters.")

    if isinstance(requested_blocks, (list, tuple)):
        names = [str(x) for x in requested_blocks]
        if len(names) == 1 and names[0].lower() == "all":
            return candidate
        missing = [name for name in names if name not in candidate]
        if missing:
            raise RuntimeError(
                f"Requested blocks are missing or filtered out: {missing[:10]}"
            )
        return names

    if str(requested_blocks).lower() == "all":
        return candidate

    raise ValueError("blockwise_distillation.blocks must be 'all' or a list of block names.")


def _match_spatial_dims(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    t_diff = ref.shape[-2] - x.shape[-2]
    f_diff = ref.shape[-1] - x.shape[-1]

    if t_diff < 0:
        x = x[..., : ref.shape[-2], :]
    elif t_diff > 0:
        x = F.pad(x, (0, 0, 0, t_diff))

    if f_diff < 0:
        x = x[..., :, : ref.shape[-1]]
    elif f_diff > 0:
        x = F.pad(x, (0, f_diff, 0, 0))

    return x


def _forward_block_standalone(
    block: torch.nn.Module,
    block_input: torch.Tensor,
    target_output: torch.Tensor,
) -> torch.Tensor:
    if isinstance(block, ConvBlock):
        if block.use_deconv:
            output_size = (
                block_input.shape[0],
                block.conv.out_channels,
                target_output.shape[-2],
                target_output.shape[-1],
            )
            pred = block(block_input, output_size=output_size)
        else:
            pred = block(block_input)
    else:
        pred = block(block_input)
    return _match_spatial_dims(pred, target_output)


class IndependentBlockwiseDistiller:
    def __init__(
        self,
        config,
        device: torch.device,
        max_steps_override: int = -1,
        epochs_override: int = -1,
    ):
        self.config = config
        self.device = device

        self.trainer_cfg = config["trainer"] if "trainer" in config else {}
        self.qat_cfg = config["qat"] if "qat" in config else {}
        self.distill_cfg = config["distillation"] if "distillation" in config else {}
        self.block_cfg = config["blockwise_distillation"] if "blockwise_distillation" in config else {}
        self.progressive_student_input = bool(self.block_cfg.get("progressive_student_input", False))
        self.progressive_context_model: torch.nn.Module | None = None

        if not bool(self.block_cfg.get("enabled", False)):
            raise ValueError("This script requires blockwise_distillation.enabled=True.")

        teacher_checkpoint = str(self.distill_cfg.get("teacher_checkpoint", "")).strip()
        if not teacher_checkpoint:
            raise ValueError("distillation.teacher_checkpoint must be set.")
        teacher_checkpoint = str(Path(teacher_checkpoint).expanduser())
        if not Path(teacher_checkpoint).exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint}")

        self.train_dataset = Dataset(**config["train_dataset"])
        collate_fn = Dataset.collate_fn if hasattr(Dataset, "collate_fn") else None
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            **config["train_dataloader"],
            shuffle=True,
            collate_fn=collate_fn,
        )
        if ("validation_dataset" not in config) or ("validation_dataloader" not in config):
            raise ValueError(
                "train_blockwise_distill.py requires validation_dataset and validation_dataloader "
                "(same split setup as train.py)."
            )
        self.validation_dataset = Dataset(**config["validation_dataset"])
        self.validation_loader = torch.utils.data.DataLoader(
            dataset=self.validation_dataset,
            **config["validation_dataloader"],
            shuffle=False,
            collate_fn=collate_fn,
        )
        self.validation_enabled = True

        self.teacher_model = Model(**config["network_config"]).to(self.device)
        _load_model_weights(self.teacher_model, teacher_checkpoint, self.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

        # Template student that provides initial weights and final assembly skeleton.
        self.student_template = Model(**config["network_config"]).to(self.device)
        self.qat_enabled = bool(self.qat_cfg.get("enabled", False))
        if self.qat_enabled:
            self.student_template.prepare_qat(
                backend=self.qat_cfg.get("backend", "fbgemm"),
                quantize_deconv=bool(self.qat_cfg.get("quantize_deconv", False)),
                per_channel_weights=bool(self.qat_cfg.get("per_channel_weights", False)),
            )

        student_checkpoint = str(self.block_cfg.get("student_checkpoint", "")).strip()
        if student_checkpoint:
            student_checkpoint = str(Path(student_checkpoint).expanduser())
            if not Path(student_checkpoint).exists():
                raise FileNotFoundError(f"Student checkpoint not found: {student_checkpoint}")
            _load_model_weights(self.student_template, student_checkpoint, self.device)

        train_encoder = bool(self.block_cfg.get("train_encoder", True))
        train_decoder = bool(self.block_cfg.get("train_decoder", True))
        requested_blocks = self.block_cfg.get("blocks", "all")
        self.block_names = _resolve_selected_blocks(
            self.teacher_model,
            train_encoder=train_encoder,
            train_decoder=train_decoder,
            requested_blocks=requested_blocks,
        )

        template_modules = dict(self.student_template.named_modules())
        self.initial_block_states = {}
        for name in self.block_names:
            if name not in template_modules:
                raise RuntimeError(f"Block '{name}' is not present in student template.")
            self.initial_block_states[name] = deepcopy(template_modules[name].state_dict())
        if self.progressive_student_input:
            self.progressive_context_model = deepcopy(self.teacher_model).to(self.device)
            if self.qat_enabled:
                self.progressive_context_model.prepare_qat(
                    backend=self.qat_cfg.get("backend", "fbgemm"),
                    quantize_deconv=bool(self.qat_cfg.get("quantize_deconv", False)),
                    per_channel_weights=bool(self.qat_cfg.get("per_channel_weights", False)),
                )
            self.progressive_context_model.eval()
            for parameter in self.progressive_context_model.parameters():
                parameter.requires_grad_(False)

        self.loss_type = str(self.block_cfg.get("loss", "mse")).lower()
        if self.loss_type not in {"mse", "l1", "smooth_l1"}:
            raise ValueError("blockwise_distillation.loss must be one of: mse, l1, smooth_l1")

        configured_epochs = int(
            self.block_cfg.get("epochs_per_block", self.block_cfg.get("epochs", self.trainer_cfg.get("epochs", 1)))
        )
        self.epochs_per_block = epochs_override if epochs_override > 0 else configured_epochs
        self.max_steps_per_epoch = max_steps_override if max_steps_override > 0 else int(
            self.block_cfg.get("max_steps_per_epoch", -1)
        )
        self.max_validation_steps_per_epoch = int(
            self.block_cfg.get("max_validation_steps_per_epoch", -1)
        )
        self.save_checkpoint_interval = int(
            self.block_cfg.get("save_checkpoint_interval", self.trainer_cfg.get("save_checkpoint_interval", 1))
        )
        self.clip_grad_norm_value = float(
            self.block_cfg.get("clip_grad_norm_value", self.trainer_cfg.get("clip_grad_norm_value", 3.0))
        )

        self.optimizer_kwargs = OmegaConf.to_container(config["optimizer"], resolve=True)
        self.scheduler_cfg = config["scheduler"] if "scheduler" in config else {}
        self.scheduler_update_interval = str(self.scheduler_cfg.get("update_interval", "step")).lower()
        self.scheduler_kwargs = None
        if "kwargs" in self.scheduler_cfg and self.scheduler_cfg["kwargs"] is not None:
            self.scheduler_kwargs = OmegaConf.to_container(self.scheduler_cfg["kwargs"], resolve=True)

        if self.qat_enabled:
            self.qat_disable_observer_epoch = int(
                self.qat_cfg.get("disable_observer_epoch", self.epochs_per_block + 1)
            )
            self.qat_freeze_bn_epoch = int(
                self.qat_cfg.get("freeze_bn_epoch", self.epochs_per_block + 1)
            )
        else:
            self.qat_disable_observer_epoch = 10**9
            self.qat_freeze_bn_epoch = 10**9

        exp_root = str(
            self.block_cfg.get("exp_path", self.trainer_cfg.get("exp_path", "./experiments"))
        ).rstrip("/")
        self.exp_path = f"{exp_root}_independent_blocks_{datetime.now().strftime('%Y-%m-%d-%Hh%Mm')}"
        self.log_path = os.path.join(self.exp_path, "logs")
        self.blocks_path = os.path.join(self.exp_path, "blocks")
        self.final_path = os.path.join(self.exp_path, "final_model")
        self.code_path = os.path.join(self.exp_path, "codes")
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.blocks_path, exist_ok=True)
        os.makedirs(self.final_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)

        cfg_obj = OmegaConf.create(config)
        OmegaConf.save(cfg_obj, os.path.join(self.exp_path, "config.yaml"))
        shutil.copy2(__file__, self.exp_path)
        for file in Path(__file__).parent.iterdir():
            if file.is_file():
                shutil.copy2(file, self.code_path)
        shutil.copytree(Path(__file__).parent / "models", Path(self.code_path) / "models", dirs_exist_ok=True)

        self.writer = SummaryWriter(self.log_path)

        print("Independent block-wise distillation setup complete.")
        print(f"Selected {len(self.block_names)} blocks for standalone training:")
        for name in self.block_names:
            print(f"  - {name}")
        print("GTConv blocks are trained as conv-path-only units (target = teacher point_bn2).")
        print("Validation mode: ON (best block checkpoint selected by validation loss).")
        if self.progressive_student_input:
            print("Progressive student input mode: ON (inputs are captured from a progressively updated student).")
        else:
            print("Progressive student input mode: OFF (inputs are captured from the teacher).")

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.smooth_l1_loss(pred, target)

    def _apply_qat_schedule_to_block(self, block: torch.nn.Module, epoch: int) -> None:
        if not self.qat_enabled:
            return

        if epoch >= self.qat_disable_observer_epoch:
            block.apply(quant.disable_observer)

        freeze_bn_stats = getattr(getattr(torch.nn.intrinsic, "qat", None), "freeze_bn_stats", None)
        if (freeze_bn_stats is not None) and (epoch >= self.qat_freeze_bn_epoch):
            block.apply(freeze_bn_stats)

    def _build_training_unit(self, block_name: str) -> tuple[torch.nn.Module, str, str]:
        """
        Build an independent trainable unit for one block.

        Returns:
        - unit module
        - teacher target module name for hook capture
        - payload type used during final assembly
        """
        template_modules = dict(self.student_template.named_modules())
        source_block = template_modules[block_name]

        if isinstance(source_block, GTConvBlock):
            unit = GTConvConvPath(source_block).to(self.device)
            missing_keys, unexpected_keys = unit.load_state_dict(
                self.initial_block_states[block_name],
                strict=False,
            )
            bad_unexpected = [k for k in unexpected_keys if not k.startswith("tra.")]
            if bad_unexpected:
                raise RuntimeError(
                    f"Unexpected keys while initializing GTConv conv-path '{block_name}': "
                    f"{bad_unexpected[:10]}"
                )
            # Missing keys are expected for excluded TRA/shuffle parts.
            _ = missing_keys
            teacher_target_module_name = f"{block_name}.point_bn2"
            payload_type = "gtconv_conv_path"
        else:
            unit = deepcopy(source_block).to(self.device)
            unit.load_state_dict(self.initial_block_states[block_name], strict=True)
            teacher_target_module_name = block_name
            payload_type = "full_block"

        unit.train()
        for parameter in unit.parameters():
            parameter.requires_grad_(True)
        return unit, teacher_target_module_name, payload_type

    @staticmethod
    def _load_payload_into_module(
        module: torch.nn.Module,
        payload: dict[str, Any],
        block_name: str,
        stage: str,
    ) -> None:
        payload_type = payload.get("payload_type", "full_block")
        if payload_type == "gtconv_conv_path":
            missing_keys, unexpected_keys = module.load_state_dict(
                payload["model"],
                strict=False,
            )
            bad_missing = [k for k in missing_keys if not k.startswith("tra.")]
            if bad_missing or unexpected_keys:
                raise RuntimeError(
                    f"Failed partial load for GTConv block '{block_name}' during {stage}. "
                    f"bad_missing={bad_missing[:10]}, unexpected={unexpected_keys[:10]}"
                )
            return

        module.load_state_dict(payload["model"], strict=True)

    @torch.inference_mode()
    def _validation_one_block(
        self,
        block_name: str,
        training_unit: torch.nn.Module,
        capture_teacher: SingleBlockIOCapture,
        capture_student_input: SingleBlockIOCapture | None,
    ) -> float:
        was_training = training_unit.training
        training_unit.eval()
        total_val_loss = 0.0
        val_steps = 0

        try:
            progress = tqdm(self.validation_loader, ncols=120, dynamic_ncols=True)
            for step, (noisy, _clean) in enumerate(progress, 1):
                noisy = noisy.to(self.device)

                capture_teacher.clear()
                if capture_student_input is not None:
                    capture_student_input.clear()

                _ = self.teacher_model(noisy)
                if capture_student_input is not None:
                    _ = self.progressive_context_model(noisy)

                teacher_output = capture_teacher.block_output
                if teacher_output is None:
                    raise RuntimeError(f"Missing teacher output capture during validation for block '{block_name}'.")

                if capture_student_input is not None:
                    block_input = capture_student_input.block_input
                    if block_input is None:
                        raise RuntimeError(
                            f"Missing progressive student input capture during validation for block '{block_name}'."
                        )
                else:
                    block_input = capture_teacher.block_input
                    if block_input is None:
                        raise RuntimeError(f"Missing teacher input capture during validation for block '{block_name}'.")

                pred = _forward_block_standalone(training_unit, block_input, teacher_output)
                val_loss = self._compute_loss(pred, teacher_output)
                total_val_loss += float(val_loss.detach().item())
                val_steps = step

                progress.set_description(f"block[{_sanitize_name(block_name)}] val")
                progress.set_postfix_str(f"loss={total_val_loss / val_steps:.5f}")

                if self.max_validation_steps_per_epoch > 0 and step >= self.max_validation_steps_per_epoch:
                    break
        finally:
            if was_training:
                training_unit.train()

        if val_steps == 0:
            raise RuntimeError(f"No validation steps executed for block '{block_name}'.")
        return total_val_loss / val_steps

    def _train_one_block(self, block_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        training_unit, teacher_target_module_name, payload_type = self._build_training_unit(block_name)
        trainable_params = [p for p in training_unit.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(f"Block '{block_name}' has no trainable parameters.")
        optimizer = torch.optim.Adam(params=trainable_params, **self.optimizer_kwargs)
        scheduler = WarmupLR(optimizer, **self.scheduler_kwargs) if self.scheduler_kwargs else None

        capture_teacher = SingleBlockIOCapture(
            self.teacher_model,
            input_module_name=block_name,
            target_module_name=teacher_target_module_name,
        )
        capture_student_input = None
        if self.progressive_student_input:
            if self.progressive_context_model is None:
                raise RuntimeError("progressive_student_input=True requires a progressive context model.")
            capture_student_input = SingleBlockIOCapture(
                self.progressive_context_model,
                input_module_name=block_name,
                target_module_name=block_name,
            )
        block_tag = _sanitize_name(block_name)
        block_dir = os.path.join(self.blocks_path, block_tag)
        ckpt_dir = os.path.join(block_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        best_train_loss = float("inf")
        best_val_loss = float("inf")

        try:
            for epoch in range(1, self.epochs_per_block + 1):
                self._apply_qat_schedule_to_block(training_unit, epoch)

                if hasattr(self.train_loader.dataset, "sample_data_per_epoch"):
                    self.train_loader.dataset.sample_data_per_epoch()

                progress = tqdm(self.train_loader, ncols=120, dynamic_ncols=True)
                total_loss = 0.0
                steps = 0

                for step, (noisy, _clean) in enumerate(progress, 1):
                    noisy = noisy.to(self.device)

                    capture_teacher.clear()
                    if capture_student_input is not None:
                        capture_student_input.clear()

                    with torch.no_grad():
                        _ = self.teacher_model(noisy)
                        if capture_student_input is not None:
                            _ = self.progressive_context_model(noisy)

                    teacher_output = capture_teacher.block_output
                    if teacher_output is None:
                        raise RuntimeError(f"Missing teacher output capture for block '{block_name}'.")
                    if capture_student_input is not None:
                        block_input = capture_student_input.block_input
                        if block_input is None:
                            raise RuntimeError(
                                f"Missing progressive student input capture for block '{block_name}'."
                            )
                    else:
                        block_input = capture_teacher.block_input
                        if block_input is None:
                            raise RuntimeError(f"Missing teacher input capture for block '{block_name}'.")

                    optimizer.zero_grad(set_to_none=True)
                    pred = _forward_block_standalone(training_unit, block_input, teacher_output)
                    loss = self._compute_loss(pred, teacher_output)
                    loss.backward()

                    if self.clip_grad_norm_value > 0:
                        torch.nn.utils.clip_grad_norm_(training_unit.parameters(), self.clip_grad_norm_value)

                    optimizer.step()
                    if scheduler is not None and self.scheduler_update_interval == "step":
                        scheduler.step()

                    loss_value = float(loss.detach().item())
                    total_loss += loss_value
                    steps = step

                    progress.set_description(
                        f"block[{block_tag}] epoch[{epoch}/{self.epochs_per_block}]"
                    )
                    progress.set_postfix_str(f"loss={total_loss / steps:.5f}")

                    if self.max_steps_per_epoch > 0 and step >= self.max_steps_per_epoch:
                        break

                if steps == 0:
                    raise RuntimeError(f"No steps executed while training block '{block_name}'.")

                epoch_loss = total_loss / steps
                if scheduler is not None and self.scheduler_update_interval == "epoch":
                    scheduler.step()

                val_loss = self._validation_one_block(
                    block_name=block_name,
                    training_unit=training_unit,
                    capture_teacher=capture_teacher,
                    capture_student_input=capture_student_input,
                )
                selection_loss = val_loss

                self.writer.add_scalar(f"blocks/{block_tag}/loss", epoch_loss, epoch)
                self.writer.add_scalar(f"blocks/{block_tag}/val_loss", val_loss, epoch)
                self.writer.add_scalar(f"blocks/{block_tag}/lr", optimizer.param_groups[0]["lr"], epoch)

                ckpt_payload = {
                    "block_name": block_name,
                    "epoch": epoch,
                    "model": training_unit.state_dict(),
                    "payload_type": payload_type,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "selection_loss": selection_loss,
                }
                if epoch % self.save_checkpoint_interval == 0:
                    torch.save(ckpt_payload, os.path.join(ckpt_dir, f"model_{str(epoch).zfill(3)}.tar"))

                if selection_loss < best_loss:
                    best_loss = selection_loss
                    best_epoch = epoch
                    best_state = deepcopy(training_unit.state_dict())
                    best_train_loss = epoch_loss
                    best_val_loss = val_loss
        finally:
            capture_teacher.remove()
            if capture_student_input is not None:
                capture_student_input.remove()
        if best_state is None:
            raise RuntimeError(f"Block '{block_name}' did not produce a best state.")

        best_payload = {
            "block_name": block_name,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "payload_type": payload_type,
            "model": best_state,
        }
        torch.save(best_payload, os.path.join(ckpt_dir, f"best_model_{str(best_epoch).zfill(3)}.tar"))

        summary = {
            "block_name": block_name,
            "block_tag": block_tag,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
        }
        return best_payload, summary

    def _update_progressive_context(self, block_name: str, payload: dict[str, Any]) -> None:
        if not self.progressive_student_input:
            return
        if self.progressive_context_model is None:
            raise RuntimeError("progressive_student_input=True requires a progressive context model.")

        context_modules = dict(self.progressive_context_model.named_modules())
        if block_name not in context_modules:
            raise RuntimeError(
                f"Block '{block_name}' is missing in progressive context model."
            )
        self._load_payload_into_module(
            module=context_modules[block_name],
            payload=payload,
            block_name=block_name,
            stage="progressive context update",
        )
        self.progressive_context_model.eval()

    def _assemble_final_model(
        self,
        trained_block_states: dict[str, dict[str, Any]],
        block_summaries: dict[str, dict[str, Any]],
    ) -> str:
        # Start from a full teacher copy (all non-QAT paths from teacher).
        final_model = deepcopy(self.teacher_model).to(self.device)
        if self.qat_enabled:
            final_model.prepare_qat(
                backend=self.qat_cfg.get("backend", "fbgemm"),
                quantize_deconv=bool(self.qat_cfg.get("quantize_deconv", False)),
                per_channel_weights=bool(self.qat_cfg.get("per_channel_weights", False)),
            )
        final_modules = dict(final_model.named_modules())

        for name, payload in trained_block_states.items():
            self._load_payload_into_module(
                module=final_modules[name],
                payload=payload,
                block_name=name,
                stage="final assembly",
            )

        final_checkpoint = {
            "model": final_model.state_dict(),
            "trained_blocks": self.block_names,
            "block_summaries": block_summaries,
            "qat_enabled": self.qat_enabled,
            "assembled_from_teacher_base": True,
        }
        final_path = os.path.join(self.final_path, "assembled_student_blockwise.tar")
        torch.save(final_checkpoint, final_path)
        return final_path

    def train(self) -> None:
        trained_block_states: dict[str, dict[str, Any]] = {}
        block_summaries: dict[str, dict[str, Any]] = {}

        try:
            num_blocks = len(self.block_names)
            for i, block_name in enumerate(self.block_names, 1):
                print(f"[{i}/{num_blocks}] Training standalone block: {block_name}")
                best_payload, summary = self._train_one_block(block_name)
                trained_block_states[block_name] = best_payload
                block_summaries[block_name] = summary
                self._update_progressive_context(block_name, best_payload)
                print(
                    f"Finished {block_name}: best_val_loss={summary['best_val_loss']:.6f}, "
                    f"best_train_loss={summary['best_train_loss']:.6f} "
                    f"(epoch {summary['best_epoch']})"
                )

            assembled_path = self._assemble_final_model(trained_block_states, block_summaries)
            print(f"Saved assembled student model to: {assembled_path}")
        finally:
            self.writer.close()


def _resolve_device(device_arg: str) -> torch.device:
    arg = str(device_arg).strip().lower()
    if arg in {"cpu", "none", "-1"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return torch.device("cpu")

    gpu_indices = [chunk.strip() for chunk in str(device_arg).split(",") if chunk.strip()]
    if not gpu_indices or any(not idx.isdigit() for idx in gpu_indices):
        raise ValueError("--device must be 'cpu' or GPU indices like '0' or '0,1'.")

    # Single-process script. If multiple GPU indices are passed, first one is used.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices[0]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA runtime is unavailable.")
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device("cuda:0")


def main(args) -> None:
    device = _resolve_device(args.device)
    config = OmegaConf.load(args.config)
    trainer = IndependentBlockwiseDistiller(
        config=config,
        device=device,
        max_steps_override=args.max_steps,
        epochs_override=args.epochs,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", default="configs/cfg_train_dist.yaml")
    parser.add_argument(
        "-D",
        "--device",
        default="0",
        help="GPU index (e.g. 0) or 'cpu'. If multiple indices are provided, only the first is used.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Debug override: max train steps per epoch per block. -1 uses config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=-1,
        help="Debug override: epochs per block. -1 uses config.",
    )
    cli_args = parser.parse_args()
    main(cli_args)
