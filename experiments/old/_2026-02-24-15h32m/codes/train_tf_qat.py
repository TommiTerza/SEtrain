"""
TensorFlow QAT training script for the exact GTCRN architecture.

Important behavior:
1) Builds an exact GTCRN TensorFlow port from `models/gtcrn_tf_exact.py`.
2) Verifies TF-vs-PyTorch parameter count parity before training.
3) Applies QAT annotation + quantize_apply (including deconv layers when enabled).
"""

import argparse
import importlib
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from dataloader import DNS3Dataset as Dataset

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - runtime dependency guard
    tf = None
    _TF_IMPORT_ERROR = exc
else:
    _TF_IMPORT_ERROR = None

try:
    import tensorflow_model_optimization as tfmot
except Exception as exc:  # pragma: no cover - runtime dependency guard
    tfmot = None
    _TFMOT_IMPORT_ERROR = exc
else:
    _TFMOT_IMPORT_ERROR = None

try:
    from pesq import pesq
except Exception:
    pesq = None

if tf is not None:
    tf_function = tf.function
else:
    def tf_function(func):
        return func


SEED = 43
random.seed(SEED)
np.random.seed(SEED)


def _load_gtcrn_tf_exact_module():
    return importlib.import_module("models.gtcrn_tf_exact")


def _require_tf_dependencies(require_tfmot: bool):
    if tf is None:
        raise RuntimeError(
            "TensorFlow is not available in this environment.\n"
            f"Import error: {_TF_IMPORT_ERROR}\n"
            "Install with e.g.: `pip install tensorflow`"
        )
    if require_tfmot and (tfmot is None):
        raise RuntimeError(
            "tensorflow-model-optimization is required for QAT but is not available.\n"
            f"Import error: {_TFMOT_IMPORT_ERROR}\n"
            "Install with e.g.: `pip install tensorflow-model-optimization`"
        )


def _configure_device(device_arg):
    device_arg = str(device_arg).strip().lower()
    if device_arg in {"cpu", "none", "-1"}:
        tf.config.set_visible_devices([], "GPU")
        return "cpu"

    if "," in device_arg:
        raise ValueError("--device must be a single GPU index (e.g. 0) or 'cpu'")
    if not device_arg.isdigit():
        raise ValueError("--device must be a single GPU index (e.g. 0) or 'cpu'")

    gpu_index = int(device_arg)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "cpu"
    if gpu_index >= len(gpus):
        raise ValueError(f"Requested GPU index {gpu_index}, but only {len(gpus)} GPU(s) are visible.")

    tf.config.set_visible_devices(gpus[gpu_index], "GPU")
    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
    return f"gpu:{gpu_index}"


class HybridLossTF:
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        compress_factor=0.3,
        eps=1e-12,
        lamda_ri=30.0,
        lamda_mag=70.0,
    ):
        self.n_fft = int(n_fft)
        self.hop_len = int(hop_len)
        self.win_len = int(win_len)
        self.c = float(compress_factor)
        self.eps = float(eps)
        self.lamda_ri = float(lamda_ri)
        self.lamda_mag = float(lamda_mag)

    def __call__(self, y_true, y_pred):
        eps = tf.cast(self.eps, y_pred.dtype)

        pred_stft = tf.signal.stft(
            y_pred,
            frame_length=self.win_len,
            frame_step=self.hop_len,
            fft_length=self.n_fft,
            window_fn=tf.signal.hann_window,
        )
        true_stft = tf.signal.stft(
            y_true,
            frame_length=self.win_len,
            frame_step=self.hop_len,
            fft_length=self.n_fft,
            window_fn=tf.signal.hann_window,
        )

        pred_mag = tf.maximum(tf.abs(pred_stft), eps)
        true_mag = tf.maximum(tf.abs(true_stft), eps)

        pred_scale = tf.cast(tf.pow(pred_mag, 1.0 - self.c), pred_stft.dtype)
        true_scale = tf.cast(tf.pow(true_mag, 1.0 - self.c), true_stft.dtype)
        pred_stft_c = pred_stft / pred_scale
        true_stft_c = true_stft / true_scale

        real_loss = tf.reduce_mean(tf.square(tf.math.real(pred_stft_c) - tf.math.real(true_stft_c)))
        imag_loss = tf.reduce_mean(tf.square(tf.math.imag(pred_stft_c) - tf.math.imag(true_stft_c)))
        mag_loss = tf.reduce_mean(tf.square(tf.pow(pred_mag, self.c) - tf.pow(true_mag, self.c)))

        y_norm = (
            tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True) * y_true
            / (tf.reduce_sum(tf.square(y_true), axis=-1, keepdims=True) + 1e-8)
        )
        sisnr_ratio = tf.norm(y_norm, axis=-1, keepdims=True) / tf.maximum(
            tf.norm(y_pred - y_norm, axis=-1, keepdims=True), eps
        )
        log10 = tf.math.log(tf.constant(10.0, dtype=y_pred.dtype))
        sisnr = -2.0 * tf.reduce_mean(tf.math.log(sisnr_ratio + eps) / log10)

        return self.lamda_ri * (real_loss + imag_loss) + self.lamda_mag * mag_loss + sisnr


def _iter_numpy_batches(dataset, batch_size, shuffle, drop_last):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if drop_last and len(batch_indices) < batch_size:
            continue

        noisy_batch = []
        clean_batch = []
        for idx in batch_indices:
            noisy, clean = dataset[idx]
            noisy_batch.append(noisy.astype(np.float32))
            clean_batch.append(clean.astype(np.float32))
        yield np.stack(noisy_batch, axis=0), np.stack(clean_batch, axis=0)


def _compute_pesq_batch(clean_np, enhanced_np, samplerate):
    if pesq is None:
        return None
    scores = []
    for c, e in zip(clean_np, enhanced_np):
        try:
            scores.append(float(pesq(samplerate, c, e, "wb")))
        except Exception:
            continue
    if not scores:
        return None
    return float(np.mean(scores))


class TFTrainer:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        tf.keras.utils.set_random_seed(SEED)

        gtcrn_tf = _load_gtcrn_tf_exact_module()
        self._build_gtcrn_tf_model = gtcrn_tf.build_gtcrn_tf_model
        self._apply_qat_to_annotated_model = gtcrn_tf.apply_qat_to_annotated_model
        self._compare_tf_torch_param_counts = gtcrn_tf.compare_tf_torch_param_counts
        self._summarize_qat_wrappers = gtcrn_tf.summarize_qat_wrappers

        self.qat_cfg = config.get("qat", {})
        self.qat_enabled = bool(self.qat_cfg.get("enabled", True))
        self.quantize_deconv = bool(self.qat_cfg.get("quantize_deconv", True))
        self.log_qat_wrapper_summary = bool(self.qat_cfg.get("log_qat_wrapper_summary", True))

        self.network_cfg = dict(config["network_config"])

        self.train_dataset = Dataset(**config["train_dataset"])
        self.validation_dataset = Dataset(**config["validation_dataset"])
        self.train_batch_size = int(config["train_dataloader"]["batch_size"])
        self.val_batch_size = int(config["validation_dataloader"]["batch_size"])
        self.train_drop_last = bool(config["train_dataloader"].get("drop_last", False))
        self.val_drop_last = bool(config["validation_dataloader"].get("drop_last", False))

        self.trainer_cfg = config["trainer"]
        self.epochs = int(self.trainer_cfg["epochs"])
        self.save_checkpoint_interval = int(self.trainer_cfg.get("save_checkpoint_interval", 1))
        self.clip_grad_norm_value = float(self.trainer_cfg.get("clip_grad_norm_value", 0.0))
        self.compute_pesq = bool(self.trainer_cfg.get("compute_pesq", False))
        self.samplerate = int(config.get("samplerate", 16000))

        self.max_train_steps = int(args.max_train_steps)
        self.max_val_steps = int(args.max_val_steps)

        self.resume = bool(self.trainer_cfg.get("resume", False))
        if not self.resume:
            self.exp_path = self.trainer_cfg["exp_path"] + "_" + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
        else:
            self.exp_path = self.trainer_cfg["exp_path"] + "_" + str(self.trainer_cfg.get("resume_datetime", ""))

        self.log_path = os.path.join(self.exp_path, "logs")
        self.checkpoint_path = os.path.join(self.exp_path, "checkpoints")
        self.code_path = os.path.join(self.exp_path, "codes")
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)

        data = OmegaConf.create(config)
        OmegaConf.save(data, os.path.join(self.exp_path, "config.yaml"))
        shutil.copy2(__file__, self.exp_path)
        shutil.copytree(Path(__file__).parent / "models", Path(self.code_path) / "models", dirs_exist_ok=True)
        for file in Path(__file__).parent.iterdir():
            if file.is_file() and file.suffix in {".py", ".yaml", ".yml"}:
                shutil.copy2(file, self.code_path)

        # Build exact float GTCRN first and validate parity against PyTorch.
        float_model = self._build_gtcrn_tf_model(
            **self.network_cfg, qat_annotate=False, quantize_deconv=self.quantize_deconv
        )
        torch_params, tf_params, diff = self._compare_tf_torch_param_counts(float_model, self.network_cfg)
        print(f"Parameter count parity check: torch={torch_params}, tf={tf_params}, diff={diff}")
        if diff != 0:
            raise RuntimeError(
                "TF GTCRN parameter count mismatch with PyTorch. "
                f"Expected {torch_params}, got {tf_params} (diff={diff})."
            )

        if self.qat_enabled:
            annotated_model = self._build_gtcrn_tf_model(
                **self.network_cfg,
                qat_annotate=True,
                quantize_deconv=self.quantize_deconv,
            )
            self.model = self._apply_qat_to_annotated_model(annotated_model)
            if self.log_qat_wrapper_summary:
                summary = self._summarize_qat_wrappers(self.model)
                print(
                    "QAT wrappers summary: "
                    f"total={summary.total_wrappers}, conv2d={summary.conv2d_wrappers}, "
                    f"deconv={summary.deconv_wrappers}"
                )
                report_path = os.path.join(self.exp_path, "qat_wrapper_layers.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    for line in summary.wrapper_lines:
                        f.write(line + "\n")
                print(f"Saved QAT wrapper report: {report_path}")
                if self.quantize_deconv and (summary.deconv_wrappers <= 0):
                    raise RuntimeError(
                        "QAT deconv was requested but no Conv2DTranspose quantization wrappers were found."
                    )
        else:
            self.model = float_model

        self.loss_fn = HybridLossTF(**config["loss"])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=float(config["optimizer"]["lr"]))

        self.summary_writer = tf.summary.create_file_writer(self.log_path)
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=10)

        self.start_epoch = 1
        self.best_score = -np.inf
        self.best_epoch = 0

        if self.resume:
            self._resume_if_possible()

    def _resume_if_possible(self):
        latest = self.ckpt_manager.latest_checkpoint
        if not latest:
            print("Resume requested but no TensorFlow checkpoint was found. Starting from scratch.")
            return
        self.ckpt.restore(latest).expect_partial()
        basename = os.path.basename(latest)
        try:
            self.start_epoch = int(basename.split("-")[-1]) + 1
        except Exception:
            self.start_epoch = 1
        print(f"Resumed from checkpoint: {latest} (start_epoch={self.start_epoch})")

    @tf_function
    def _train_step(self, noisy, clean):
        with tf.GradientTape() as tape:
            enhanced = self.model(noisy, training=True)
            loss = self.loss_fn(clean, enhanced)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None]
        if self.clip_grad_norm_value > 0.0 and grads_and_vars:
            grads_only = [g for g, _ in grads_and_vars]
            grads_only, _ = tf.clip_by_global_norm(grads_only, self.clip_grad_norm_value)
            grads_and_vars = [(g, v) for g, (_, v) in zip(grads_only, grads_and_vars)]
        self.optimizer.apply_gradients(grads_and_vars)
        return loss

    @tf_function
    def _val_step(self, noisy, clean):
        enhanced = self.model(noisy, training=False)
        loss = self.loss_fn(clean, enhanced)
        return loss, enhanced

    def _train_epoch(self, epoch):
        if hasattr(self.train_dataset, "sample_data_per_epoch"):
            self.train_dataset.sample_data_per_epoch()

        iterator = _iter_numpy_batches(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=self.train_drop_last,
        )

        total_loss = 0.0
        steps = 0
        pbar = tqdm(iterator, ncols=110, dynamic_ncols=True)
        for step, (noisy_np, clean_np) in enumerate(pbar, start=1):
            noisy = tf.convert_to_tensor(noisy_np, dtype=tf.float32)
            clean = tf.convert_to_tensor(clean_np, dtype=tf.float32)
            loss = self._train_step(noisy, clean)

            steps = step
            total_loss += float(loss.numpy())
            pbar.set_description(f"   train[{epoch}/{self.epochs + self.start_epoch - 1}]")
            pbar.set_postfix_str(f"train_loss={total_loss / max(1, steps):.3f}")

            if self.max_train_steps > 0 and step >= self.max_train_steps:
                break

        train_loss = total_loss / max(1, steps)
        with self.summary_writer.as_default():
            tf.summary.scalar("train/loss", train_loss, step=epoch)
            tf.summary.scalar("train/lr", self.optimizer.learning_rate, step=epoch)
        return train_loss

    def _validation_epoch(self, epoch):
        iterator = _iter_numpy_batches(
            self.validation_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=self.val_drop_last,
        )

        total_loss = 0.0
        total_pesq = 0.0
        pesq_steps = 0
        steps = 0

        pbar = tqdm(iterator, ncols=123, dynamic_ncols=True)
        for step, (noisy_np, clean_np) in enumerate(pbar, start=1):
            noisy = tf.convert_to_tensor(noisy_np, dtype=tf.float32)
            clean = tf.convert_to_tensor(clean_np, dtype=tf.float32)
            loss, enhanced = self._val_step(noisy, clean)

            steps = step
            total_loss += float(loss.numpy())

            if self.compute_pesq:
                pesq_score = _compute_pesq_batch(clean_np, enhanced.numpy(), self.samplerate)
                if pesq_score is not None:
                    total_pesq += pesq_score
                    pesq_steps += 1

            postfix = f"valid_loss={total_loss / max(1, steps):.3f}"
            if self.compute_pesq and pesq_steps > 0:
                postfix += f", pesq={total_pesq / pesq_steps:.4f}"
            pbar.set_description(f"validate[{epoch}/{self.epochs + self.start_epoch - 1}]")
            pbar.set_postfix_str(postfix)

            if self.max_val_steps > 0 and step >= self.max_val_steps:
                break

        val_loss = total_loss / max(1, steps)
        val_pesq = (total_pesq / pesq_steps) if pesq_steps > 0 else None
        with self.summary_writer.as_default():
            tf.summary.scalar("val/loss", val_loss, step=epoch)
            if val_pesq is not None:
                tf.summary.scalar("val/pesq", val_pesq, step=epoch)
        return val_loss, val_pesq

    def _save_checkpoint(self, epoch):
        self.ckpt_manager.save(checkpoint_number=epoch)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            train_loss = self._train_epoch(epoch)
            val_loss, val_pesq = self._validation_epoch(epoch)

            if epoch % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            current_score = val_pesq if val_pesq is not None else -val_loss
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_epoch = epoch
                best_path = os.path.join(self.checkpoint_path, f"best_model_{str(epoch).zfill(3)}.weights.h5")
                self.model.save_weights(best_path)

            print(
                f"[epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"best_epoch={self.best_epoch}"
            )

        print(f"------------TensorFlow GTCRN QAT training for {self.epochs} epochs is done!------------")


def main(args):
    config = OmegaConf.load(args.config)
    qat_cfg = config.get("qat", {})
    qat_enabled = bool(qat_cfg.get("enabled", True))
    _require_tf_dependencies(require_tfmot=qat_enabled)
    _ = _configure_device(args.device)
    trainer = TFTrainer(config=config, args=args)
    if args.check_only:
        print("Model build/parity/QAT-wrapper checks completed (--check_only).")
        return
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", default="configs/cfg_train_tf_qat.yaml")
    parser.add_argument("-D", "--device", default="0", help="GPU index (e.g. 0) or 'cpu' to force CPU")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=-1,
        help="If > 0, cap train steps per epoch for quick smoke testing.",
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=-1,
        help="If > 0, cap validation steps per epoch for quick smoke testing.",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Build model, run parity/QAT checks, then exit without training.",
    )
    args = parser.parse_args()
    main(args)
