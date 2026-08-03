"""
Training script for GTCRN.

Supports:
1) Supervised training with clean targets.
2) QAT for Conv/GTConv blocks (existing path).
3) Optional knowledge distillation (KD) from a frozen teacher model where:
   - the student output matches the teacher output;
   - intermediate activations are matched (standard Conv/ConvTranspose taps
     or edge taps at Conv<->GRU boundaries).
"""

import os
import torch
import torch.ao.quantization as quant
import random
import shutil
import argparse
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from pesq import pesq
from joblib import Parallel, delayed
import soundfile as sf
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from distributed_utils import reduce_value

from models.gtcrn_end2end import GTCRN as Model, ConvBlock, GTConvBlock
from loss_factory import HybridLoss as Loss
from dataloader import DNS3Dataset as Dataset
from scheduler import LinearWarmupCosineAnnealingLR as WarmupLR

seed = 43
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True


class ActivationCapture:
    """
    Capture module outputs via forward hooks.

    The captured tensors keep gradients for the student model so feature-level KD
    can backpropagate through the tapped student modules.
    """

    def __init__(self, model, module_names):
        self.outputs = {}
        self.handles = []
        named_modules = dict(model.named_modules())
        missing = [name for name in module_names if name not in named_modules]
        if missing:
            raise RuntimeError(
                f"Cannot register KD hooks. Missing modules: {missing[:5]}"
            )

        for name in module_names:
            module = named_modules[name]
            self.handles.append(module.register_forward_hook(self._build_hook(name)))

    def _build_hook(self, name):
        def hook(_, __, output):
            # Some modules may return tuples. KD currently uses the first output tensor.
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.outputs[name] = output

        return hook

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.outputs.clear()


def _get_conv_deconv_module_names(model):
    """
    Collect activation points for Conv/ConvTranspose distillation.

    We distill the post-convolution normalization outputs:
    - ConvBlock: `bn`
    - GTConvBlock: `point_bn1`, `depth_bn`, `point_bn2`

    This keeps layer matching stable even when QAT fusion replaces
    `conv + bn` with a fused conv module and sets BN to identity.
    """
    layer_names = []
    for module_name, module in model.named_modules():
        if isinstance(module, ConvBlock):
            layer_names.append(f"{module_name}.bn")
        elif isinstance(module, GTConvBlock):
            layer_names.extend(
                [
                    f"{module_name}.point_bn1",
                    f"{module_name}.depth_bn",
                    f"{module_name}.point_bn2",
                ]
            )
    return layer_names


def _get_feature_hook_module_names(model, hook_scope):
    """
    Return module names used for feature-level distillation hooks.

    Supported scopes:
    - standard: Conv/ConvTranspose-adjacent BN taps (legacy behavior).
    - edge: top-level interfaces between convolutional stacks and GRU stacks.
    """
    scope = str(hook_scope).strip().lower()
    if scope == "standard":
        return _get_conv_deconv_module_names(model)

    if scope == "edge":
        # Interfaces:
        # - encoder output -> first DPGRNN input
        # - second DPGRNN output -> decoder input
        edge_modules = ["encoder", "dpgrnn2"]
        named_modules = dict(model.named_modules())
        missing = [name for name in edge_modules if name not in named_modules]
        if missing:
            raise RuntimeError(
                f"Cannot register edge KD hooks. Missing modules: {missing[:5]}"
            )
        return edge_modules

    raise ValueError(
        "distillation.feature_hook_scope must be one of: standard, edge"
    )


def _extract_model_state_dict(checkpoint):
    """Extract model weights from common checkpoint formats."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format. Expected a dict-like checkpoint.")

    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]

    # Bare state_dict format.
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError(
        "Could not find model weights in checkpoint. Expected keys: 'model' or 'state_dict'."
    )


def _load_teacher_weights(model, teacher_checkpoint, device):
    """
    Load teacher weights and remove optional DDP 'module.' prefixes.
    """
    checkpoint = torch.load(teacher_checkpoint, map_location=device)
    state_dict = _extract_model_state_dict(checkpoint)

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

    # Missing keys indicate an incompatible teacher/student architecture.
    if missing_keys:
        raise RuntimeError(
            f"Teacher checkpoint is incompatible. Missing keys: {missing_keys[:10]}"
        )

    # QAT checkpoints can contain observer/fake-quant states; keep training if only those are extra.
    ignored_prefixes = ("activation_post_process", "weight_fake_quant")
    bad_unexpected = [
        k for k in unexpected_keys if not any(pfx in k for pfx in ignored_prefixes)
    ]
    if bad_unexpected:
        raise RuntimeError(
            f"Teacher checkpoint has unexpected keys: {bad_unexpected[:10]}"
        )


def _copy_and_freeze_gru_from_teacher(student_model, teacher_model):
    """
    Copy GRU weights from teacher into student and freeze student GRUs.
    """
    student_grus = {
        name: module
        for name, module in student_model.named_modules()
        if isinstance(module, torch.nn.GRU)
    }
    if not student_grus:
        raise RuntimeError("Student model has no GRU layers to freeze from teacher.")

    teacher_grus = {
        name: module
        for name, module in teacher_model.named_modules()
        if isinstance(module, torch.nn.GRU)
    }
    missing_teacher_grus = [name for name in student_grus if name not in teacher_grus]
    if missing_teacher_grus:
        raise RuntimeError(
            "Teacher is missing GRU layers required by student: "
            f"{missing_teacher_grus[:10]}"
        )

    frozen_params = 0
    for name, student_gru in student_grus.items():
        student_gru.load_state_dict(teacher_grus[name].state_dict(), strict=True)
        for parameter in student_gru.parameters():
            parameter.requires_grad_(False)
            frozen_params += parameter.numel()

    return len(student_grus), frozen_params


def run(rank, config, args):
    use_cuda = (not getattr(args, "force_cpu", False)) and torch.cuda.is_available()

    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12354'
        dist.init_process_group("nccl" if use_cuda else "gloo", rank=rank, world_size=args.world_size)
        if use_cuda:
            torch.cuda.set_device(rank)
        dist.barrier()

    args.rank = rank
    args.device = torch.device(f"cuda:{rank}") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    collate_fn = Dataset.collate_fn if hasattr(Dataset, "collate_fn") else None
    # config['train_dataloader']['batch_size'] = config['train_dataloader']['batch_size'] // args.world_size
    shuffle = False if args.world_size > 1 else True

    train_dataset = Dataset(**config['train_dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.world_size > 1 else None
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    sampler=train_sampler,
                                                    **config['train_dataloader'],
                                                    shuffle=shuffle,
                                                    collate_fn=collate_fn)
    
    validation_dataset = Dataset(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset) if args.world_size > 1 else None
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                        sampler=validation_sampler,
                                                        **config['validation_dataloader'], 
                                                        shuffle=False,
                                                        collate_fn=collate_fn)
        
    model = Model(**config['network_config']).to(args.device)
    qat_config = config['qat'] if 'qat' in config else {}
    qat_enabled = bool(qat_config.get('enabled', False))
    dynamic_quantize_gru = bool(qat_config.get('dynamic_quantize_gru', False))
    if qat_enabled:
        model.prepare_qat(
            backend=qat_config.get('backend', 'fbgemm'),
            quantize_deconv=bool(qat_config.get('quantize_deconv', False)),
            per_channel_weights=bool(qat_config.get('per_channel_weights', False)),
        )
        if dynamic_quantize_gru and (rank == 0):
            print(
                "QAT config: dynamic_quantize_gru=True. "
                "GRUs will be dynamically quantized during int8 conversion."
            )

    distillation_config = config['distillation'] if 'distillation' in config else {}
    distillation_enabled = bool(distillation_config.get('enabled', False))
    if distillation_enabled and not qat_enabled:
        raise ValueError(
            "distillation.enabled=True expects qat.enabled=True so the student is trained with QAT."
        )
    freeze_gru_from_teacher = bool(distillation_config.get('freeze_gru_from_teacher', False))
    if freeze_gru_from_teacher and not distillation_enabled:
        raise ValueError(
            "distillation.freeze_gru_from_teacher=True requires distillation.enabled=True."
        )
    teacher_model = None
    if distillation_enabled:
        teacher_checkpoint = str(distillation_config.get('teacher_checkpoint', '')).strip()
        if not teacher_checkpoint:
            raise ValueError(
                "distillation.enabled=True requires distillation.teacher_checkpoint."
            )
        teacher_checkpoint = Path(teacher_checkpoint).expanduser()
        if not teacher_checkpoint.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint}")

        teacher_model = Model(**config['network_config']).to(args.device)
        _load_teacher_weights(teacher_model, str(teacher_checkpoint), args.device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

        if freeze_gru_from_teacher:
            frozen_gru_modules, frozen_gru_params = _copy_and_freeze_gru_from_teacher(
                student_model=model,
                teacher_model=teacher_model,
            )
            if rank == 0:
                print(
                    f"KD enabled: copied and froze {frozen_gru_modules} GRU modules "
                    f"({frozen_gru_params} parameters) from teacher."
                )

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank]) if use_cuda else torch.nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.Adam(params=model.parameters(), **config['optimizer'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['scheduler']['kwargs'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['scheduler']['kwargs'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['scheduler']['kwargs'])
    scheduler = WarmupLR(optimizer, **config['scheduler']['kwargs'])
    
    loss_func = Loss(**config['loss']).to(args.device)

    trainer = Trainer(config=config, model=model,optimizer=optimizer, scheduler=scheduler, loss_func=loss_func,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      train_sampler=train_sampler, teacher_model=teacher_model, args=args)

    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, teacher_model, args):
        self.config = config
        self.model = model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size

        # training config
        config['DDP']['world_size'] = args.world_size
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']
        self.qat_config = config['qat'] if 'qat' in config else {}
        self.qat_enabled = bool(self.qat_config.get('enabled', False))
        self.qat_disable_observer_epoch = self.qat_config.get('disable_observer_epoch', self.epochs + 1)
        self.qat_freeze_bn_epoch = self.qat_config.get('freeze_bn_epoch', self.epochs + 1)
        self.qat_disable_observer_epoch = (
            int(self.qat_disable_observer_epoch) if self.qat_disable_observer_epoch is not None else self.epochs + 1
        )
        self.qat_freeze_bn_epoch = (
            int(self.qat_freeze_bn_epoch) if self.qat_freeze_bn_epoch is not None else self.epochs + 1
        )
        self.qat_log_observer_formats = bool(self.qat_config.get('log_observer_formats', True))
        self.qat_log_clip_stats = bool(self.qat_config.get('log_clip_stats', True))
        self.qat_clip_stats_topk = max(1, int(self.qat_config.get('clip_stats_topk', 8)))
        clip_batches = self.qat_config.get('clip_stats_batches_per_epoch', 20)
        self.qat_clip_stats_batches_per_epoch = int(clip_batches) if clip_batches is not None else None
        if (self.qat_clip_stats_batches_per_epoch is not None) and (self.qat_clip_stats_batches_per_epoch <= 0):
            self.qat_clip_stats_batches_per_epoch = None
        self._qat_fake_quant_modules = []
        self._qat_clip_handles = []
        self._qat_clip_stats = {}
        self._qat_clip_collect_this_step = False
        self.distillation_config = config['distillation'] if 'distillation' in config else {}
        self.distillation_enabled = bool(self.distillation_config.get('enabled', False)) and (self.teacher_model is not None)
        self.freeze_gru_from_teacher = bool(self.distillation_config.get('freeze_gru_from_teacher', False))
        if self.freeze_gru_from_teacher and not self.distillation_enabled:
            raise ValueError(
                "distillation.freeze_gru_from_teacher=True requires distillation.enabled=True "
                "with a valid teacher model."
            )
        self.supervised_loss_weight = float(self.distillation_config.get('supervised_loss_weight', 1.0))
        self.distill_output_loss_weight = float(self.distillation_config.get('output_loss_weight', 0.0))
        self.distill_feature_loss_weight = float(self.distillation_config.get('feature_loss_weight', 0.0))
        self.distill_feature_loss_type = str(
            self.distillation_config.get('feature_loss_type', 'mse')
        ).strip().lower()
        if self.distill_feature_loss_type not in {'mse', 'cosine'}:
            raise ValueError(
                "distillation.feature_loss_type must be one of: mse, cosine"
            )
        self.distill_feature_hook_scope = str(
            self.distillation_config.get('feature_hook_scope', 'standard')
        ).strip().lower()
        if self.distill_feature_hook_scope not in {'standard', 'edge'}:
            raise ValueError(
                "distillation.feature_hook_scope must be one of: standard, edge"
            )
        self.distill_layer_names = []
        self.teacher_activations = None
        self.student_activations = None

        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
 
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')
        self.code_path = os.path.join(self.exp_path, 'codes')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)
        
        # save the config and codes
        if self.rank == 0:
            data = OmegaConf.create(config)
            OmegaConf.save(data, os.path.join(self.exp_path, 'config.yaml'))

            shutil.copy2(__file__, self.exp_path)
            for file in Path(__file__).parent.iterdir():
                if file.is_file():
                    shutil.copy2(file, self.code_path)
            shutil.copytree(Path(__file__).parent / 'models', Path(self.code_path) / 'models', dirs_exist_ok=True)
            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 0

        if self.distillation_enabled:
            self._setup_distillation()

        if self.resume:
            self._resume_checkpoint()

        if self.qat_enabled:
            self._setup_qat_logging()

    @staticmethod
    def _flag_is_enabled(flag):
        if flag is None:
            return True
        if torch.is_tensor(flag):
            if flag.numel() == 0:
                return False
            return bool(int(flag.reshape(-1)[0].item()))
        return bool(flag)

    def _setup_qat_logging(self):
        model = self._get_model()
        self._qat_fake_quant_modules = [
            (name, module) for name, module in model.named_modules() if isinstance(module, FakeQuantizeBase)
        ]
        if self.rank == 0:
            print(f"QAT logging: discovered {len(self._qat_fake_quant_modules)} fake-quant modules.")

        if (self.rank == 0) and self.qat_log_observer_formats:
            self._log_qat_observer_formats()
        if (self.rank == 0) and self.qat_log_clip_stats:
            self._register_qat_clip_hooks()

    def _cleanup_qat_logging(self):
        for handle in self._qat_clip_handles:
            handle.remove()
        self._qat_clip_handles.clear()
        self._qat_clip_stats.clear()
        self._qat_clip_collect_this_step = False

    def _log_qat_observer_formats(self, *, verbose: bool = True):
        if not self._qat_fake_quant_modules:
            if verbose:
                print("QAT observer report: no fake-quant modules found.")
            return

        def _summarize_tensor(value, *, max_elems: int = 8):
            if value is None:
                return "", "", "", ""

            if not torch.is_tensor(value):
                value = torch.tensor(value)

            if value.numel() == 0:
                shape = "x".join(str(d) for d in value.shape) if value.ndim else "0"
                return shape, "", "", "[]"

            shape = "x".join(str(d) for d in value.shape) if value.ndim else "1"
            flat = value.detach().cpu().reshape(-1)
            if flat.dtype.is_floating_point:
                flat = flat.to(torch.float64)
                min_value = float(flat.min().item())
                max_value = float(flat.max().item())
                sample_values = flat[:max_elems].tolist()
                sample = ",".join(f"{v:.6g}" for v in sample_values)
            else:
                flat = flat.to(torch.int64)
                min_value = int(flat.min().item())
                max_value = int(flat.max().item())
                sample_values = flat[:max_elems].tolist()
                sample = ",".join(str(int(v)) for v in sample_values)

            if flat.numel() > max_elems:
                sample = sample + ",..."
            return shape, f"{min_value}", f"{max_value}", f"[{sample}]"

        header = (
            "module_name\trole\tfake_quant\tobserver\tdtype\tqscheme\tquant_min\tquant_max\tbits\tchannel_axis\t"
            "scale_shape\tscale_min\tscale_max\tscale_sample\t"
            "zero_point_shape\tzero_point_min\tzero_point_max\tzero_point_sample"
        )
        report_lines = [header]
        grouped = {}

        for module_name, module in self._qat_fake_quant_modules:
            observer = getattr(module, "activation_post_process", None)
            role = "weight" if "weight_fake_quant" in module_name else "activation"
            observer_name = type(observer).__name__ if observer is not None else "None"
            dtype = str(getattr(observer, "dtype", None))
            qscheme = str(getattr(observer, "qscheme", None))
            channel_axis = getattr(observer, "ch_axis", None)
            quant_min = getattr(module, "quant_min", None)
            quant_max = getattr(module, "quant_max", None)
            bits = None
            if (quant_min is not None) and (quant_max is not None):
                levels = int(quant_max) - int(quant_min) + 1
                bits = int(np.ceil(np.log2(levels))) if levels > 0 else 0

            scale = getattr(module, "scale", None)
            zero_point = getattr(module, "zero_point", None)
            scale_shape, scale_min, scale_max, scale_sample = _summarize_tensor(scale)
            zp_shape, zp_min, zp_max, zp_sample = _summarize_tensor(zero_point)

            report_lines.append(
                f"{module_name}\t{role}\t{type(module).__name__}\t{observer_name}\t{dtype}\t{qscheme}\t"
                f"{quant_min}\t{quant_max}\t{bits}\t{channel_axis}\t"
                f"{scale_shape}\t{scale_min}\t{scale_max}\t{scale_sample}\t"
                f"{zp_shape}\t{zp_min}\t{zp_max}\t{zp_sample}"
            )
            signature = (role, observer_name, dtype, qscheme, quant_min, quant_max, bits, channel_axis)
            grouped[signature] = grouped.get(signature, 0) + 1

        if verbose:
            print("QAT observer format summary:")
            for signature, count in sorted(grouped.items(), key=lambda item: (-item[1], str(item[0]))):
                role, observer_name, dtype, qscheme, quant_min, quant_max, bits, channel_axis = signature
                print(
                    f"  x{count:02d} role={role}, observer={observer_name}, dtype={dtype}, "
                    f"qscheme={qscheme}, range=[{quant_min}, {quant_max}], bits={bits}, ch_axis={channel_axis}"
                )

        # ConvTranspose2d does not get a weight_fake_quant module in eager QAT, so surface its intended
        # weight qscheme/channel-axis from qconfig for easier verification.
        model = self._get_model()
        convt_rows = []
        for module_name, module in model.named_modules():
            if not isinstance(module, torch.nn.ConvTranspose2d):
                continue
            qconfig = getattr(module, "qconfig", None)
            if qconfig is None:
                continue
            try:
                weight_fq = qconfig.weight()
                weight_obs = getattr(weight_fq, "activation_post_process", None)
                w_qscheme = getattr(weight_obs, "qscheme", getattr(weight_fq, "qscheme", None))
                w_axis = getattr(weight_obs, "ch_axis", getattr(weight_fq, "ch_axis", None))
                convt_rows.append(
                    f"{module_name}\t{module.groups}\t{module.in_channels}\t{module.out_channels}\t{w_qscheme}\t{w_axis}"
                )
            except Exception as exc:
                convt_rows.append(
                    f"{module_name}\t{module.groups}\t{module.in_channels}\t{module.out_channels}\tERROR\t{type(exc).__name__}"
                )
        if convt_rows:
            report_lines.append("")
            report_lines.append("# ConvTranspose2d weight qconfig summary")
            report_lines.append("module_name\tgroups\tin_channels\tout_channels\tweight_qscheme\tweight_channel_axis")
            report_lines.extend(convt_rows)

        report_path = os.path.join(self.exp_path, "qat_observer_formats.txt")
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write("\n".join(report_lines) + "\n")
        if verbose:
            print(f"Saved QAT observer format report: {report_path}")

        if verbose and hasattr(self, "writer"):
            summary_lines = [
                f"x{count:02d} role={sig[0]}, observer={sig[1]}, dtype={sig[2]}, qscheme={sig[3]}, "
                f"range=[{sig[4]}, {sig[5]}], bits={sig[6]}, ch_axis={sig[7]}"
                for sig, count in sorted(grouped.items(), key=lambda item: (-item[1], str(item[0])))
            ]
            self.writer.add_text("qat/observer_formats", "  \n".join(summary_lines), global_step=0)

    def _register_qat_clip_hooks(self):
        self._cleanup_qat_logging()
        self._qat_clip_stats = {
            module_name: {
                "role": "weight" if "weight_fake_quant" in module_name else "activation",
                "numel": 0,
                "clipped_low": 0,
                "clipped_high": 0,
                "calls": 0,
            }
            for module_name, _ in self._qat_fake_quant_modules
        }

        for module_name, module in self._qat_fake_quant_modules:
            handle = module.register_forward_hook(self._build_qat_clip_hook(module_name))
            self._qat_clip_handles.append(handle)

    def _build_qat_clip_hook(self, module_name):
        def hook(module, inputs, _output):
            if not self._qat_clip_collect_this_step:
                return
            if not self._flag_is_enabled(getattr(module, "fake_quant_enabled", None)):
                return
            if not inputs:
                return

            observed = inputs[0]
            if not torch.is_tensor(observed):
                return
            if observed.numel() == 0:
                return

            scale = getattr(module, "scale", None)
            zero_point = getattr(module, "zero_point", None)
            quant_min = getattr(module, "quant_min", None)
            quant_max = getattr(module, "quant_max", None)
            if (scale is None) or (zero_point is None) or (quant_min is None) or (quant_max is None):
                return

            with torch.no_grad():
                observed = observed.detach()
                dtype = observed.dtype if observed.is_floating_point() else torch.float32
                device = observed.device

                scale = scale.detach().to(device=device, dtype=dtype)
                zero_point = zero_point.detach().to(device=device, dtype=dtype)
                min_allowed = (float(quant_min) - zero_point) * scale
                max_allowed = (float(quant_max) - zero_point) * scale

                if (scale.numel() > 1) and (observed.ndim > 0):
                    observer = getattr(module, "activation_post_process", None)
                    raw_channel_axis = getattr(observer, "ch_axis", 0)
                    channel_axis = int(raw_channel_axis) if raw_channel_axis is not None else 0
                    if channel_axis < 0:
                        channel_axis += observed.ndim

                    if (0 <= channel_axis < observed.ndim) and (observed.shape[channel_axis] == scale.numel()):
                        view_shape = [1] * observed.ndim
                        view_shape[channel_axis] = scale.numel()
                        min_allowed = min_allowed.view(view_shape)
                        max_allowed = max_allowed.view(view_shape)
                    else:
                        min_allowed = min_allowed.min()
                        max_allowed = max_allowed.max()

                clipped_low = torch.count_nonzero(observed < min_allowed)
                clipped_high = torch.count_nonzero(observed > max_allowed)
                stats = self._qat_clip_stats.get(module_name)
                if stats is None:
                    return

                stats["numel"] += int(observed.numel())
                stats["clipped_low"] += int(clipped_low.item())
                stats["clipped_high"] += int(clipped_high.item())
                stats["calls"] += 1

        return hook

    def _reset_qat_clip_stats(self):
        if not self._qat_clip_stats:
            return
        for stats in self._qat_clip_stats.values():
            stats["numel"] = 0
            stats["clipped_low"] = 0
            stats["clipped_high"] = 0
            stats["calls"] = 0

    def _summarize_qat_clip_stats(self, epoch):
        entries = []
        overall_numel = 0
        overall_clipped = 0
        act_numel = 0
        act_clipped = 0
        wt_numel = 0
        wt_clipped = 0

        for module_name, stats in self._qat_clip_stats.items():
            numel = stats["numel"]
            if numel <= 0:
                continue
            clipped = stats["clipped_low"] + stats["clipped_high"]
            clip_pct = 100.0 * clipped / max(1, numel)
            low_pct = 100.0 * stats["clipped_low"] / max(1, numel)
            high_pct = 100.0 * stats["clipped_high"] / max(1, numel)
            entries.append((clip_pct, module_name, low_pct, high_pct, numel, stats["calls"], stats["role"]))

            overall_numel += numel
            overall_clipped += clipped
            if stats["role"] == "weight":
                wt_numel += numel
                wt_clipped += clipped
            else:
                act_numel += numel
                act_clipped += clipped

        if not entries:
            print(f"[QAT][epoch {epoch}] No clipping stats collected.")
            return

        overall_pct = 100.0 * overall_clipped / max(1, overall_numel)
        act_pct = 100.0 * act_clipped / max(1, act_numel)
        wt_pct = 100.0 * wt_clipped / max(1, wt_numel)
        sampled_batches = self.qat_clip_stats_batches_per_epoch if self.qat_clip_stats_batches_per_epoch is not None else "all"

        print(
            f"[QAT][epoch {epoch}] clipping: overall={overall_pct:.4f}% "
            f"(activation={act_pct:.4f}%, weight={wt_pct:.4f}%), sampled_batches={sampled_batches}"
        )

        entries.sort(key=lambda item: item[0], reverse=True)
        for clip_pct, module_name, low_pct, high_pct, numel, calls, role in entries[: self.qat_clip_stats_topk]:
            print(
                f"  [{role}] {module_name}: clipped={clip_pct:.4f}% "
                f"(low={low_pct:.4f}%, high={high_pct:.4f}%, elems={numel}, calls={calls})"
            )

        if hasattr(self, "writer"):
            self.writer.add_scalars(
                "qat_clip/overall",
                {
                    "overall_pct": overall_pct,
                    "activation_pct": act_pct,
                    "weight_pct": wt_pct,
                },
                epoch,
            )
            for clip_pct, module_name, _, _, _, _, _ in entries[: self.qat_clip_stats_topk]:
                tag = module_name.replace(".", "_")
                self.writer.add_scalar(f"qat_clip/module_top/{tag}", clip_pct, epoch)

    def _set_train_mode(self):
        self.model.train()
        if self.distillation_enabled:
            # Teacher stays frozen in eval mode during student optimization.
            self.teacher_model.eval()

    def _set_eval_mode(self):
        self.model.eval()

    def _get_model(self):
        return self.model.module if self.world_size > 1 else self.model

    def _setup_distillation(self):
        """
        Prepare layer-wise KD hooks.

        Layer pairing uses module names so teacher and student stay aligned.
        """
        student_model = self._get_model()
        self.teacher_model.eval()

        teacher_layer_names = _get_feature_hook_module_names(
            self.teacher_model,
            self.distill_feature_hook_scope,
        )
        if not teacher_layer_names:
            raise RuntimeError(
                f"No distillation feature hook points were found for scope "
                f"'{self.distill_feature_hook_scope}'."
            )

        student_modules = dict(student_model.named_modules())
        missing_student_layers = [name for name in teacher_layer_names if name not in student_modules]
        if missing_student_layers:
            raise RuntimeError(
                "Student is missing distillation hook modules required for KD: "
                f"{missing_student_layers[:10]}"
            )

        self.distill_layer_names = teacher_layer_names
        self.teacher_activations = ActivationCapture(self.teacher_model, self.distill_layer_names)
        self.student_activations = ActivationCapture(student_model, self.distill_layer_names)

        if self.rank == 0:
            print(
                "KD enabled: matching "
                f"{len(self.distill_layer_names)} activations "
                f"(scope={self.distill_feature_hook_scope}) "
                f"with {self.distill_feature_loss_type} feature loss."
            )

    def _cleanup_distillation(self):
        if self.teacher_activations is not None:
            self.teacher_activations.remove()
            self.teacher_activations = None
        if self.student_activations is not None:
            self.student_activations.remove()
            self.student_activations = None

    def _compute_feature_distillation_loss(self):
        """
        Mean feature loss across matched activation hooks.
        """
        losses = []
        for layer_name in self.distill_layer_names:
            teacher_feat = self.teacher_activations.outputs.get(layer_name)
            student_feat = self.student_activations.outputs.get(layer_name)
            if (teacher_feat is None) or (student_feat is None):
                raise RuntimeError(
                    f"Missing captured activation for KD layer: {layer_name}"
                )
            if self.distill_feature_loss_type == 'mse':
                losses.append(F.mse_loss(student_feat, teacher_feat))
            elif self.distill_feature_loss_type == 'cosine':
                student_flat = student_feat.flatten(1)
                teacher_flat = teacher_feat.flatten(1)
                cosine_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1, eps=1e-8)
                losses.append(1.0 - cosine_sim.mean())
            else:
                raise RuntimeError(
                    "Unsupported distillation.feature_loss_type: "
                    f"{self.distill_feature_loss_type}"
                )

        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def _apply_qat_schedule(self, epoch):
        if not self.qat_enabled:
            return

        model = self._get_model()
        if epoch >= self.qat_disable_observer_epoch:
            model.apply(quant.disable_observer)

        freeze_bn_stats = getattr(getattr(torch.nn.intrinsic, 'qat', None), 'freeze_bn_stats', None)
        if (freeze_bn_stats is not None) and (epoch >= self.qat_freeze_bn_epoch):
            model.apply(freeze_bn_stats)

    def _save_checkpoint(self, epoch, score):
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'model': model_dict}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(3)}.tar'))

        if score > self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

        if self.freeze_gru_from_teacher:
            frozen_gru_modules, frozen_gru_params = _copy_and_freeze_gru_from_teacher(
                student_model=self._get_model(),
                teacher_model=self.teacher_model,
            )
            if self.rank == 0:
                print(
                    "Re-applied teacher GRU freeze after resume: "
                    f"{frozen_gru_modules} modules, {frozen_gru_params} parameters."
                )

    def _train_epoch(self, epoch):
        total_loss = 0
        total_supervised_loss = 0
        total_distill_output_loss = 0
        total_distill_feature_loss = 0
        if (self.rank == 0) and self.qat_enabled and self.qat_log_clip_stats:
            self._reset_qat_clip_stats()
        if hasattr(self.train_dataloader.dataset, "sample_data_per_epoch"):
            self.train_dataloader.dataset.sample_data_per_epoch()
        self.train_bar = tqdm(self.train_dataloader, ncols=110, dynamic_ncols=True)

        for step, (noisy, clean) in enumerate(self.train_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)  

            teacher_enhanced = None
            if self.distillation_enabled:
                self.teacher_activations.clear()
                self.student_activations.clear()
                with torch.no_grad():
                    teacher_enhanced = self.teacher_model(noisy)

            self._qat_clip_collect_this_step = (
                (self.rank == 0)
                and self.qat_enabled
                and self.qat_log_clip_stats
                and (
                    (self.qat_clip_stats_batches_per_epoch is None)
                    or (step <= self.qat_clip_stats_batches_per_epoch)
                )
            )
            enhanced = self.model(noisy)
            supervised_loss = self.loss_func(enhanced, clean)

            if self.distillation_enabled:
                distill_output_loss = F.mse_loss(enhanced, teacher_enhanced)
                distill_feature_loss = self._compute_feature_distillation_loss()
                loss = (
                    self.supervised_loss_weight * supervised_loss
                    + self.distill_output_loss_weight * distill_output_loss
                    + self.distill_feature_loss_weight * distill_feature_loss
                )
            else:
                distill_output_loss = torch.zeros((), device=self.device)
                distill_feature_loss = torch.zeros((), device=self.device)
                loss = supervised_loss

            self.train_bar.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()

            # Keep distributed reduction for logging only.
            if self.world_size > 1:
                loss_for_log = reduce_value(loss.detach().clone())
                supervised_for_log = reduce_value(supervised_loss.detach().clone())
                distill_output_for_log = reduce_value(distill_output_loss.detach().clone())
                distill_feature_for_log = reduce_value(distill_feature_loss.detach().clone())
            else:
                loss_for_log = loss.detach()
                supervised_for_log = supervised_loss.detach()
                distill_output_for_log = distill_output_loss.detach()
                distill_feature_for_log = distill_feature_loss.detach()

            total_loss += loss_for_log.item()
            total_supervised_loss += supervised_for_log.item()
            total_distill_output_loss += distill_output_for_log.item()
            total_distill_feature_loss += distill_feature_for_log.item()

            if self.distillation_enabled:
                sup_avg = total_supervised_loss / step
                kd_out_avg = total_distill_output_loss / step
                kd_feat_avg = total_distill_feature_loss / step
                self.train_bar.postfix = (
                    'loss={:.3f}, sup_w={:.3f}, kd_out_w={:.3f}, kd_feat_w={:.3f}'.format(
                        total_loss / step,
                        self.supervised_loss_weight * sup_avg,
                        self.distill_output_loss_weight * kd_out_avg,
                        self.distill_feature_loss_weight * kd_feat_avg,
                    )
                )
            else:
                self.train_bar.postfix = 'train_loss={:.3f}'.format(total_loss / step)

            if self.config['scheduler']['update_interval'] == 'step':
                self.scheduler.step()

        self._qat_clip_collect_this_step = False
        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            train_metrics = {
                'train_loss': total_loss / step,
                'supervised_loss': total_supervised_loss / step,
            }
            if self.distillation_enabled:
                train_metrics['distill_output_loss'] = total_distill_output_loss / step
                train_metrics['distill_feature_loss'] = total_distill_feature_loss / step
            self.writer.add_scalars('train_loss', train_metrics, epoch)
            if self.qat_enabled and self.qat_log_clip_stats:
                self._summarize_qat_clip_stats(epoch)
            if self.qat_enabled and self.qat_log_observer_formats:
                # Keep an up-to-date snapshot of scales/zero-points as observers evolve.
                self._log_qat_observer_formats(verbose=False)


    @torch.inference_mode()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_pesq_score = 0

        self.validation_bar = tqdm(self.validation_dataloader, ncols=123, dynamic_ncols=True)
        for step, (noisy, clean) in enumerate(self.validation_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)  
            
            enhanced = self.model(noisy)

            loss = self.loss_func(enhanced, clean)
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()

            clean = clean.cpu().numpy()
            enhanced = enhanced.detach().cpu().numpy()
            pesq_score_batch = Parallel(n_jobs=-1)(
                delayed(pesq)(16000, c, e, 'wb') for c, e in zip(clean, enhanced))
            pesq_score = torch.tensor(pesq_score_batch, device=self.device).mean()
            if self.world_size > 1:
                pesq_score = reduce_value(pesq_score)
            total_pesq_score += pesq_score
            
            if self.rank == 0 and (epoch==1 or epoch %10 == 0) and step <= 3:
                noisy_path = os.path.join(self.sample_path, 'sample_{}_noisy.wav'.format(step))
                clean_path = os.path.join(self.sample_path, 'sample_{}_clean.wav'.format(step))
                enhanced_path = os.path.join(self.sample_path, 'sample_{}_enh_epoch{}.wav'.format(step, str(epoch).zfill(3)))
                if not os.path.exists(noisy_path):
                    noisy = noisy.cpu().numpy()
                    sf.write(noisy_path, noisy[0], samplerate=self.config['samplerate'])
                    sf.write(clean_path, clean[0], samplerate=self.config['samplerate'])

                sf.write(enhanced_path, enhanced[0], samplerate=self.config['samplerate'])

            self.validation_bar.desc = 'validate[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.validation_bar.postfix = 'valid_loss={:.3f}, pesq={:.4f}'.format(
                total_loss / step, total_pesq_score / step)

        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars(
                'val_loss', {'val_loss': total_loss / step, 
                             'pesq': total_pesq_score / step}, epoch)

        return total_loss / step, total_pesq_score / step


    def train(self):
        try:
            if self.resume:
                self._resume_checkpoint()

            for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)

                self._set_train_mode()
                self._apply_qat_schedule(epoch)
                self._train_epoch(epoch)

                self._set_eval_mode()
                valid_loss, score = self._validation_epoch(epoch)
                
                if self.config['scheduler']['update_interval'] == 'epoch':
                    if self.config['scheduler']['use_plateau']:
                        self.scheduler.step(score)
                    else:
                        self.scheduler.step()

                if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
                    self._save_checkpoint(epoch, score)

            if self.rank == 0:
                torch.save(self.state_dict_best,
                        os.path.join(self.checkpoint_path,
                        'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(3))))

                print('------------Training for {} epochs is done!------------'.format(self.epochs))
        finally:
            self._cleanup_qat_logging()
            self._cleanup_distillation()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_train.yaml')
    parser.add_argument(
        '-D',
        '--device',
        default='0',
        help="GPU indices (e.g. 0 or 0,1,2,3) or 'cpu' to force CPU",
    )

    args = parser.parse_args()
    device_arg = str(args.device).strip()
    if device_arg.lower() in {"cpu", "none", "-1"}:
        args.force_cpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        args.world_size = 1
    else:
        parts = [p.strip() for p in device_arg.split(",") if p.strip()]
        if not parts or any(not p.isdigit() for p in parts):
            raise ValueError("--device must be GPU indices like '0' or '0,1' or 'cpu'")
        args.force_cpu = False
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(parts)
        args.world_size = len(parts)
    config = OmegaConf.load(args.config)
    
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
