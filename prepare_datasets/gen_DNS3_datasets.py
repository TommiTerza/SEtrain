import argparse
import os
import random
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from tqdm import tqdm


DEFAULT_CLEAN_URL = "https://dns3public.blob.core.windows.net/datasets/datasets_fullband/clean_fullband.tar.bz2"
DEFAULT_NOISE_URL = "https://dns3public.blob.core.windows.net/datasets/datasets_fullband/noise_fullband.tar.bz2"
DEFAULT_RIR_URL = "https://dns3public.blob.core.windows.net/datasets/datasets_fullband/impulse_responses.zip"

DEFAULT_CLEAN_SUBDIR = "clean_fullband"
DEFAULT_NOISE_SUBDIR = "noise_fullband"
DEFAULT_RIR_SUBDIR = "impulse_responses"

def add_pyreverb(clean_speech, rir):
    # max_index = np.argmax(np.abs(rir))
    # rir = rir[max_index:]
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[: clean_speech.shape[0]]

    return reverb_speech


def mk_mixture(s1, s2, s1_ref, snr, eps=1e-8):
    """
    s1: reverbrant speech
    s2: reverbrant speech or noise
    s1_ref: s1 with low reverbration, as target in training
    """
    amp = 0.5 * np.random.rand() + 0.01
    s1_ref = amp * s1_ref / (np.max(np.abs(s1)) + eps) 
    s1 = amp * s1 / (np.max(np.abs(s1)) + eps) 
    norm_sig1 = s1

    norm_sig2 = s2 * np.math.sqrt(np.sum(s1 ** 2) + eps) / np.math.sqrt(np.sum(s2 ** 2) + eps)
    alpha = 10**(-snr*1.5 / 20)
    # freq_num = np.random.randint(0, 4)
    # sins = np.zeros(len(s1))
    # if freq_num > 1:
    #     freq = np.random.choice(range(50, 8000), freq_num)
    #     for f in freq:
    #         s_sin = np.sin(2*np.pi * f * np.arange(len(s1)) / 16000)
    #         sins = sins + (0.5*np.random.rand() + 0.5) * alpha * s_sin * np.math.sqrt(np.sum(s1 ** 2) + eps) / np.math.sqrt(np.sum(s_sin ** 2) + eps)
    
    # mix = norm_sig1 + alpha * norm_sig2 + sins
    mix = norm_sig1 + alpha * norm_sig2
    
    M = max(np.max(abs(mix)), np.max(abs(norm_sig1)), np.max(abs(alpha*norm_sig2))) + eps
    if M > 1.0:    
        mix = mix / M
        norm_sig1 = norm_sig1 / M
        norm_sig2 = norm_sig2 / M
        s1_ref = s1_ref / M

    return mix, s1_ref


def stream_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response, open(destination, "wb") as f:
        total = response.headers.get("Content-Length")
        total = int(total) if total is not None else None
        chunk_size = 1024 * 1024

        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {destination.name}") as pbar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(destination)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def ensure_component(download_root: Path, url: str, expected_subdir: str, force_download: bool) -> Path:
    target_dir = download_root / expected_subdir
    if target_dir.exists() and any(target_dir.rglob("*.wav")) and not force_download:
        return target_dir

    archive_name = url.split("/")[-1]
    archive_path = download_root / archive_name

    if not archive_path.exists() or force_download:
        stream_download(url, archive_path)

    extract_archive(archive_path, download_root)

    if not target_dir.exists():
        raise FileNotFoundError(f"Expected directory {target_dir} not found after extracting {archive_path}")

    return target_dir


def write_csv(list_path: Path, file_paths) -> None:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"file_dir": [str(path) for path in file_paths]})
    df.to_csv(list_path, index=False)


def collect_wavs(directory: Path):
    wavs = sorted(directory.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found under {directory}")
    return wavs


def prepare_and_load_lists(args):
    download_root = Path(args.dataset_root).expanduser().resolve()

    if args.download:
        print(f"Downloading DNS3 components into {download_root}")
        clean_dir = ensure_component(download_root, args.clean_url, args.clean_subdir, args.force_download)
        noise_dir = ensure_component(download_root, args.noise_url, args.noise_subdir, args.force_download)
        rir_dir = ensure_component(download_root, args.rir_url, args.rir_subdir, args.force_download)
    else:
        clean_dir = (download_root / args.clean_subdir).expanduser().resolve()
        noise_dir = (download_root / args.noise_subdir).expanduser().resolve()
        rir_dir = (download_root / args.rir_subdir).expanduser().resolve()

    for directory in (clean_dir, noise_dir, rir_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Expected directory {directory} does not exist. Use --download or adjust the *_subdir arguments.")

    clean_wavs = collect_wavs(clean_dir)
    noise_wavs = collect_wavs(noise_dir)
    rir_wavs = collect_wavs(rir_dir)

    csv_root = Path(args.csv_root).expanduser().resolve()
    clean_csv = csv_root / f"{args.flag}_clean_dir.csv"
    noise_csv = csv_root / f"{args.flag}_noise_dir.csv"
    rir_csv = csv_root / f"{args.flag}_rir_dir.csv"

    if args.regenerate_csv or not clean_csv.exists():
        print(f"Writing clean list to {clean_csv}")
        write_csv(clean_csv, clean_wavs)
    if args.regenerate_csv or not noise_csv.exists():
        print(f"Writing noise list to {noise_csv}")
        write_csv(noise_csv, noise_wavs)
    if args.regenerate_csv or not rir_csv.exists():
        print(f"Writing RIR list to {rir_csv}")
        write_csv(rir_csv, rir_wavs)

    clean_list = pd.read_csv(clean_csv)["file_dir"].tolist()
    noise_list = pd.read_csv(noise_csv)["file_dir"].tolist()
    rir_list = pd.read_csv(rir_csv)["file_dir"].tolist()

    num_available = min(len(clean_list), len(noise_list), len(rir_list))
    if args.num_tot is not None:
        num_tot = min(args.num_tot, num_available)
    else:
        num_tot = num_available

    if num_tot <= 0:
        raise ValueError("No data available to generate mixtures. Check the CSV contents.")

    return clean_list[:num_tot], noise_list[:num_tot], rir_list[:num_tot], num_tot


def parse_arguments():
    parser = argparse.ArgumentParser(description="Download DNS3 data, generate metadata, and create mixtures.")
    parser.add_argument("--dataset-root", default=Path(__file__).parent / "downloads", help="Root directory to store/download DNS3 components.")
    parser.add_argument("--csv-root", default=Path(__file__).parent, help="Directory where metadata CSV files will be written/read.")
    parser.add_argument("--save-root", default=Path(__file__).parent / "data" / "DNS3", help="Directory where generated mixtures will be saved.")
    parser.add_argument("--flag", default="train", help="Dataset split name to use for CSV naming and INFO files.")
    parser.add_argument("--num-tot", type=int, default=None, help="Number of mixtures to generate. Defaults to the minimum available based on the CSVs.")
    parser.add_argument("--download", action="store_true", help="Download DNS3 components before generating CSVs.")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of DNS3 components even if they already exist.")
    parser.add_argument("--regenerate-csv", action="store_true", help="Rewrite CSV lists even if they already exist.")
    parser.add_argument("--clean-url", default=DEFAULT_CLEAN_URL)
    parser.add_argument("--noise-url", default=DEFAULT_NOISE_URL)
    parser.add_argument("--rir-url", default=DEFAULT_RIR_URL)
    parser.add_argument("--clean-subdir", default=DEFAULT_CLEAN_SUBDIR, help="Relative folder under dataset-root containing clean wav files.")
    parser.add_argument("--noise-subdir", default=DEFAULT_NOISE_SUBDIR, help="Relative folder under dataset-root containing noise wav files.")
    parser.add_argument("--rir-subdir", default=DEFAULT_RIR_SUBDIR, help="Relative folder under dataset-root containing RIR wav files.")
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate of the dataset.")
    parser.add_argument("--wav-len", type=int, default=10, help="Length of the generated samples in seconds.")
    parser.add_argument("--random-start", action="store_true", help="Randomly choose start offsets when reading clean/noise files.")
    parser.add_argument("--snr-min", type=float, default=-5., help="Minimum SNR in dB for mixture generation.")
    parser.add_argument("--snr-max", type=float, default=15., help="Maximum SNR in dB for mixture generation.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for reproducibility.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_list, noise_list, rir_list, num_tot = prepare_and_load_lists(args)
    nfill = len(str(num_tot))

    fs = args.fs
    wav_len = args.wav_len  # in seconds
    random_start = args.random_start
    snr_range = [args.snr_min, args.snr_max]

    save_root = Path(args.save_root).expanduser().resolve()
    os.makedirs(save_root / f"{args.flag}_noisy", exist_ok=True)
    os.makedirs(save_root / f"{args.flag}_clean", exist_ok=True)

    snr_list = np.random.uniform(snr_range[0], snr_range[1], size=num_tot)

    info = pd.DataFrame([str(idx + 1).zfill(nfill) + ".wav" for idx in range(num_tot)], columns=["file_name"])
    info["clean"] = clean_list[:num_tot]
    info["noise"] = noise_list[:num_tot]
    info["snr"] = snr_list

    info.to_csv(save_root / f"{args.flag}_INFO.csv", index=None)

    for idx in tqdm(range(num_tot)):
        if random_start:
            start_s = int(np.random.uniform(0, max(0, 15 - wav_len))) * fs
            start_n = int(np.random.uniform(0, max(0, 30 - wav_len))) * fs
        else:
            start_s = 0
            start_n = 0

        clean = sf.read(clean_list[idx], dtype="float32", start=start_s, stop=start_s + wav_len * fs)[0]
        noise = sf.read(noise_list[idx], dtype="float32", start=start_n, stop=start_n + wav_len * fs)[0]
        rir = sf.read(rir_list[idx], dtype="float32")[0]

        if len(rir.shape) > 1:
            rir = rir[:, 0]
        max_index = np.argmax(np.abs(rir))
        rir = rir[max_index:]
        rir_e = rir[: min(int(100 * fs / 1000), len(rir))]

        rev_clean = add_pyreverb(clean, rir)  # reverberant clean speech
        drb_clean = add_pyreverb(clean, rir_e)  # clean speech with low reverberation

        mixture, target = mk_mixture(rev_clean, noise, drb_clean, snr_list[idx], eps=1e-8)

        sf.write(save_root / f"{args.flag}_noisy" / (str(idx + 1).zfill(nfill) + ".wav"), mixture, fs)
        sf.write(save_root / f"{args.flag}_clean" / (str(idx + 1).zfill(nfill) + ".wav"), target, fs)
