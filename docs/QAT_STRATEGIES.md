# QAT Strategies In This Repo

This document summarizes all quantization-aware training (QAT) strategies currently supported in this codebase.

## 1) End-to-End QAT (No KD)

Use full-model supervised training with QAT and no distillation.

- Script: `train.py`
- Typical config base: `configs/cfg_train.yaml` (or equivalent custom config)
- Key knobs:
  - `qat.enabled: True`
  - `qat.backend: fbgemm | qnnpack`
  - `qat.per_channel_weights: True | False` (forces per-channel weight quantization; per-channel deconvs require `qnnpack`)
  - `qat.quantize_deconv: True | False`
  - `distillation.enabled: False`

## 2) End-to-End QAT + KD

Train the full student model end-to-end with a frozen teacher.

- Script: `train.py`
- Typical config: `configs/cfg_train_dist.yaml`
- Key knobs:
  - `qat.enabled: True`
  - `distillation.enabled: True`
  - `distillation.teacher_checkpoint: <path>`
  - `distillation.output_loss_weight`
  - `distillation.feature_loss_weight`
  - `distillation.feature_loss_type: mse | cosine`
  - `distillation.supervised_loss_weight`
  - `distillation.freeze_gru_from_teacher: True | False`

Notes:
- Feature KD is done on Conv/ConvTranspose intermediate activations.
- `freeze_gru_from_teacher=True` keeps student GRUs fixed to teacher GRUs.

## 3) Blockwise QAT Distillation (Teacher-Input Mode)

Train Conv/GTConv blocks independently, each against teacher block I/O, then assemble.

- Script: `train_blockwise_distill.py`
- Typical config: `configs/cfg_train_dist_convonly.yaml`
- Key knobs:
  - `blockwise_distillation.enabled: True`
  - `blockwise_distillation.blocks: all` (or selected block names)
  - `blockwise_distillation.epochs_per_block`
  - `blockwise_distillation.loss: mse | l1 | smooth_l1`
  - `blockwise_distillation.progressive_student_input: False`

Important behavior:
- Each block is optimized independently.
- GTConv blocks are trained as conv-path-only units (targeting `point_bn2`), then injected into a full model for final assembly.

## 4) Blockwise QAT Distillation (Progressive Student-Input Mode)

Same blockwise pipeline, but block input is captured from a progressively updated student context model (not always from teacher input).

- Script: `train_blockwise_distill.py`
- Config knob:
  - `blockwise_distillation.progressive_student_input: True`

What changes:
- Block targets still come from teacher outputs.
- Inputs for later blocks reflect errors/distribution from earlier trained student blocks.
- This reduces teacher-input vs assembled-student mismatch.

## 5) QAT Inference Modes (Evaluation Strategy)

QAT checkpoints can be evaluated in two modes:

- Float fake-quant path:
  - `network.use_int8: False`
- Converted int8 path:
  - `network.use_int8: True`
  - conversion happens in `infer.py` via `convert_qat(...)`
  - optional GRU dynamic quantization at conversion: `qat.dynamic_quantize_gru: True`

Why this matters:
- Always compare both modes while debugging quality.
- If float is already low, training strategy is the main bottleneck.
- If float is good but int8 drops hard, quantization/deployment settings are the bottleneck.

## Scheduler/QAT Timing Reminder

For blockwise runs, tune scheduler and QAT freeze points to actual steps-per-epoch:

- Effective steps per epoch are driven by dataloader length and `max_steps_per_epoch`.
- If LR reaches `min_lr` too early, training may appear to stall.
- If observer/BN freeze happens too early, optimization may flatten prematurely.
