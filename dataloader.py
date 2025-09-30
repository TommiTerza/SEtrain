import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

VOICEBANK_DATASET_ID = "JacobLinCool/VoiceBank-DEMAND-16k"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data" / "voicebank-demand-16k"
DEFAULT_FS = 16000


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_id(identifier: str) -> str:
    return str(identifier).replace("/", "_").replace("\\", "_")


def _write_wav(path: Path, audio: np.ndarray, sr: int = DEFAULT_FS) -> None:
    _ensure_dir(path.parent)
    sf.write(path, audio.astype(np.float32), sr)


def _export_split(ds, split_root: Path, *, desc: str) -> None:
    noisy_dir = split_root / "noisy"
    clean_dir = split_root / "clean"
    _ensure_dir(noisy_dir)
    _ensure_dir(clean_dir)

    for example in tqdm(ds, desc=desc, unit="file"):
        sid = _sanitize_id(example.get("id", "sample"))
        noisy_path = noisy_dir / f"{sid}.wav"
        clean_path = clean_dir / f"{sid}.wav"
        if noisy_path.exists() and clean_path.exists():
            continue
        noisy = np.asarray(example["noisy"]["array"], dtype=np.float32)
        clean = np.asarray(example["clean"]["array"], dtype=np.float32)
        _write_wav(noisy_path, noisy)
        _write_wav(clean_path, clean)


def prepare_voicebank_dataset(root: Path) -> None:
    root = root.expanduser().resolve()
    train_noisy = root / "train" / "noisy"
    validation_noisy = root / "validation" / "noisy"
    test_noisy = root / "test" / "noisy"

    def _has_audio(directory: Path) -> bool:
        return directory.exists() and any(directory.glob("*.wav"))

    if _has_audio(train_noisy) and _has_audio(validation_noisy) and _has_audio(test_noisy):
        return

    legacy_valid_noisy = root / "valid" / "noisy"
    legacy_valid_clean = root / "valid" / "clean"
    if _has_audio(legacy_valid_noisy) and not (_has_audio(validation_noisy) and _has_audio(test_noisy)):
        clean_lookup = {p.stem: p for p in legacy_valid_clean.glob("*.wav")}
        noisy_files = sorted(legacy_valid_noisy.glob("*.wav"))
        if noisy_files:
            if len(noisy_files) < 2:
                splits = [("validation", noisy_files), ("test", noisy_files)]
            else:
                split_idx = max(1, min(len(noisy_files) - 1, len(noisy_files) // 2))
                splits = [
                    ("validation", noisy_files[:split_idx]),
                    ("test", noisy_files[split_idx:]),
                ]
            for subset, subset_files in splits:
                noisy_target = root / subset / "noisy"
                clean_target = root / subset / "clean"
                noisy_target.mkdir(parents=True, exist_ok=True)
                clean_target.mkdir(parents=True, exist_ok=True)
                for noisy_path in subset_files:
                    clean_path = clean_lookup.get(noisy_path.stem)
                    if clean_path is None:
                        continue
                    shutil.move(str(noisy_path), noisy_target / noisy_path.name)
                    if clean_path.exists():
                        shutil.move(str(clean_path), clean_target / clean_path.name)
        if legacy_valid_noisy.parent.exists() and not any(legacy_valid_noisy.glob("*.wav")):
            shutil.rmtree(legacy_valid_noisy.parent)

    if _has_audio(train_noisy) and _has_audio(validation_noisy) and _has_audio(test_noisy):
        return

    print(f"Preparing VoiceBank-DEMAND dataset in {root} ...")
    ds_train = load_dataset(VOICEBANK_DATASET_ID, split="train")
    ds_test = load_dataset(VOICEBANK_DATASET_ID, split="test")

    audio_feature = Audio(sampling_rate=DEFAULT_FS)
    ds_train = ds_train.cast_column("noisy", audio_feature)
    ds_train = ds_train.cast_column("clean", audio_feature)
    ds_test = ds_test.cast_column("noisy", audio_feature)
    ds_test = ds_test.cast_column("clean", audio_feature)

    for subset in ("train", "validation", "test"):
        subset_root = root / subset
        if subset_root.exists():
            shutil.rmtree(subset_root)

    _export_split(ds_train, root / "train", desc="VoiceBank train")

    num_test = len(ds_test)
    if num_test >= 2:
        split_idx = max(1, min(num_test - 1, num_test // 2))
        validation_indices = list(range(split_idx))
        test_indices = list(range(split_idx, num_test))
        ds_validation = ds_test.select(validation_indices)
        ds_evaluation = ds_test.select(test_indices)
    else:
        ds_validation = ds_test
        ds_evaluation = ds_test

    _export_split(ds_validation, root / "validation", desc="VoiceBank validation")
    _export_split(ds_evaluation, root / "test", desc="VoiceBank test")
    print("VoiceBank-DEMAND preparation complete.")


class VoiceBankDemandDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fs: int = DEFAULT_FS,
        length_in_seconds: float = 10.0,
        num_data_tot: int = -1,
        num_data_per_epoch: int = -1,
        random_start_point: bool = True,
        train: bool = True,
        dataset_root: Path | None = None,
        split: str | None = None,
    ):
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.L = int(round(length_in_seconds * fs))
        self.random_start_point = random_start_point
        self.train = train
        self.dataset_root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATA_ROOT
        self.split = (split or ("train" if train else "validation")).lower()
        valid_splits = {"train", "validation", "test"}
        if self.split not in valid_splits:
            raise ValueError(f"Unknown split '{self.split}'. Expected one of {sorted(valid_splits)}")

        prepare_voicebank_dataset(self.dataset_root)

        noisy_dir = self.dataset_root / self.split / "noisy"
        clean_dir = self.dataset_root / self.split / "clean"

        noisy_paths = {p.stem: p for p in sorted(noisy_dir.glob("*.wav"))}
        clean_paths = {p.stem: p for p in sorted(clean_dir.glob("*.wav"))}
        keys = sorted(set(noisy_paths) & set(clean_paths))

        if num_data_tot is not None and num_data_tot > 0:
            keys = keys[: min(num_data_tot, len(keys))]

        self.examples: List[Tuple[Path, Path]] = [(noisy_paths[k], clean_paths[k]) for k in keys]
        if not self.examples:
            raise RuntimeError(f"No paired files found in {noisy_dir} and {clean_dir}")

        self.num_data_per_epoch = num_data_per_epoch if num_data_per_epoch and num_data_per_epoch > 0 else len(self.examples)
        self.indices: List[int] = list(range(len(self.examples)))
        if self.train:
            self.sample_data_per_epoch()

    def sample_data_per_epoch(self) -> None:
        count = min(self.num_data_per_epoch, len(self.examples))
        self.indices = random.sample(range(len(self.examples)), count)

    def _crop_or_pad(self, audio: np.ndarray) -> np.ndarray:
        length = audio.shape[0]
        if length >= self.L:
            if self.train and self.random_start_point and length > self.L:
                start = random.randint(0, length - self.L)
            elif self.train:
                start = 0
            else:
                start = (length - self.L) // 2
            return audio[start : start + self.L]

        pad_needed = self.L - length
        if self.train and self.random_start_point:
            pad_left = random.randint(0, pad_needed)
        else:
            pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        return np.pad(audio, (pad_left, pad_right), mode="constant")

    def __getitem__(self, idx: int):
        if self.train:
            example_idx = self.indices[idx]
        else:
            example_idx = idx

        noisy_path, clean_path = self.examples[example_idx]
        noisy, _ = sf.read(noisy_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")

        noisy = self._crop_or_pad(noisy.astype(np.float32))
        clean = self._crop_or_pad(clean.astype(np.float32))

        return noisy, clean

    def __len__(self) -> int:
        if self.train:
            return len(self.indices)
        return len(self.examples)


DNS3Dataset = VoiceBankDemandDataset


if __name__ == "__main__":
    from torch.utils import data
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/cfg_train.yaml")

    train_dataset = VoiceBankDemandDataset(**config["train_dataset"])
    train_dataloader = data.DataLoader(train_dataset, **config["train_dataloader"])
    train_dataloader.dataset.sample_data_per_epoch()

    validation_dataset = VoiceBankDemandDataset(**config["validation_dataset"])
    validation_dataloader = data.DataLoader(validation_dataset, **config["validation_dataloader"])

    print(len(train_dataloader), len(validation_dataloader))

    for noisy, clean in train_dataloader:
        print(noisy.shape, clean.shape)
        break

    for noisy, clean in validation_dataloader:
        print(noisy.shape, clean.shape)
        break
