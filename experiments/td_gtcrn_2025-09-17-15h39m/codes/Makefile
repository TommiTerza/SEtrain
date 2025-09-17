## TD-GTCRN helper Makefile
## Inspired by the tstnn profiler helpers.

.PHONY: help prepare-voicebank train infer profile-infer profile-train clean-experiments cuda-check

PYTHON ?= python
CONFIG ?= configs/cfg_train.yaml
INFER_CONFIG ?= configs/cfg_infer.yaml
DATA_DIR ?= data/voicebank-demand-16k
PROFILE_DIR ?= runs/profiles

help:
	@echo "Available targets:"
	@echo "  prepare-voicebank  - Download + cache VoiceBank-DEMAND-16k under $(DATA_DIR)"
	@echo "  train              - Launch training with $(CONFIG)"
	@echo "  infer              - Run inference with $(INFER_CONFIG)"
	@echo "  profile-infer      - Capture inference profiler trace to $(PROFILE_DIR)/infer.json"
	@echo "  profile-train      - Capture one training step profiler trace to $(PROFILE_DIR)/train.json"
	@echo "  clean-experiments  - Remove experiments/, runs/profiles traces"
	@echo "  cuda-check         - Print CUDA availability for $(PYTHON)"

prepare-voicebank:
	@echo "[td-gtcrn] preparing VoiceBank-DEMAND under $(DATA_DIR)"
	$(PYTHON) scripts/prepare_voicebank.py --out "$(DATA_DIR)"
	@echo "[td-gtcrn] dataset ready"

train:
	@echo "[td-gtcrn] training with config $(CONFIG)"
	$(PYTHON) train.py --config $(CONFIG)

infer:
	@echo "[td-gtcrn] inference with config $(INFER_CONFIG)"
	$(PYTHON) infer.py --config $(INFER_CONFIG)

profile-infer:
	@echo "[td-gtcrn] profiling inference"
	$(PYTHON) scripts/profile_td_gtcrn.py inference --out $(PROFILE_DIR)/infer.json

profile-train:
	@echo "[td-gtcrn] profiling one training step"
	$(PYTHON) scripts/profile_td_gtcrn.py train --out $(PROFILE_DIR)/train.json

clean-experiments:
	@echo "[td-gtcrn] cleaning experiments and profile traces"
	rm -rf experiments "$(PROFILE_DIR)"

cuda-check:
	@echo "[td-gtcrn] checking CUDA availability"
	$(PYTHON) -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available());\
	print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
