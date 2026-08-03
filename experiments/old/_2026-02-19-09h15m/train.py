"""
Training script for GTCRN.

Supports:
1) Supervised training with clean targets.
2) QAT for Conv/GTConv blocks (existing path).
3) Optional knowledge distillation (KD) from a frozen teacher model where:
   - the student output matches the teacher output;
   - intermediate activations from Conv/ConvTranspose layers are matched.
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
    can backpropagate through student Conv/ConvTranspose layers.
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
    if qat_enabled:
        model.prepare_qat(
            backend=qat_config.get('backend', 'fbgemm'),
            quantize_deconv=bool(qat_config.get('quantize_deconv', False)),
        )

    distillation_config = config['distillation'] if 'distillation' in config else {}
    distillation_enabled = bool(distillation_config.get('enabled', False))
    if distillation_enabled and not qat_enabled:
        raise ValueError(
            "distillation.enabled=True expects qat.enabled=True so the student is trained with QAT."
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
        self.distillation_config = config['distillation'] if 'distillation' in config else {}
        self.distillation_enabled = bool(self.distillation_config.get('enabled', False)) and (self.teacher_model is not None)
        self.supervised_loss_weight = float(self.distillation_config.get('supervised_loss_weight', 1.0))
        self.distill_output_loss_weight = float(self.distillation_config.get('output_loss_weight', 0.0))
        self.distill_feature_loss_weight = float(self.distillation_config.get('feature_loss_weight', 0.0))
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

        We match activations only for Conv2d / ConvTranspose2d layers.
        Layer pairing uses module names so teacher and student stay aligned.
        """
        student_model = self._get_model()
        self.teacher_model.eval()

        teacher_layer_names = _get_conv_deconv_module_names(self.teacher_model)
        if not teacher_layer_names:
            raise RuntimeError("No Conv/ConvTranspose layers were found for distillation.")

        student_modules = dict(student_model.named_modules())
        missing_student_layers = [name for name in teacher_layer_names if name not in student_modules]
        if missing_student_layers:
            raise RuntimeError(
                "Student is missing Conv/ConvTranspose layers required for KD: "
                f"{missing_student_layers[:10]}"
            )

        self.distill_layer_names = teacher_layer_names
        self.teacher_activations = ActivationCapture(self.teacher_model, self.distill_layer_names)
        self.student_activations = ActivationCapture(student_model, self.distill_layer_names)

        if self.rank == 0:
            print(
                f"KD enabled: matching {len(self.distill_layer_names)} Conv/ConvTranspose activations."
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
        Mean MSE across matched Conv/ConvTranspose activations.
        """
        losses = []
        for layer_name in self.distill_layer_names:
            teacher_feat = self.teacher_activations.outputs.get(layer_name)
            student_feat = self.student_activations.outputs.get(layer_name)
            if (teacher_feat is None) or (student_feat is None):
                raise RuntimeError(
                    f"Missing captured activation for KD layer: {layer_name}"
                )
            losses.append(F.mse_loss(student_feat, teacher_feat))

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

    def _train_epoch(self, epoch):
        total_loss = 0
        total_supervised_loss = 0
        total_distill_output_loss = 0
        total_distill_feature_loss = 0
        if hasattr(self.train_dataloader.dataset, "sample_data_per_epoch"):
            self.train_dataloader.dataset.sample_data_per_epoch()
        self.train_bar = tqdm(self.train_dataloader, ncols=110)

        for step, (noisy, clean) in enumerate(self.train_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)  

            teacher_enhanced = None
            if self.distillation_enabled:
                self.teacher_activations.clear()
                self.student_activations.clear()
                with torch.no_grad():
                    teacher_enhanced = self.teacher_model(noisy)

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
                self.train_bar.postfix = (
                    'loss={:.3f}, sup={:.3f}, kd_out={:.3f}, kd_feat={:.3f}'.format(
                        total_loss / step,
                        total_supervised_loss / step,
                        total_distill_output_loss / step,
                        total_distill_feature_loss / step,
                    )
                )
            else:
                self.train_bar.postfix = 'train_loss={:.3f}'.format(total_loss / step)

            if self.config['scheduler']['update_interval'] == 'step':
                self.scheduler.step()

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


    @torch.inference_mode()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_pesq_score = 0

        self.validation_bar = tqdm(self.validation_dataloader, ncols=123)
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
