import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import torch
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf
from models.gtcrn_end2end import GTCRN as Model
from dataloader import VoiceBankDemandDataset as Dataset


def _resolve_device(device_arg: str) -> torch.device:
    device_lower = device_arg.lower()
    if device_lower in ("cpu", "none"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_arg}")
    return torch.device("cpu")


def _load_model(cfg_infer_path: str, device_str: str, log_gru_inputs: bool, log_file: str, custom_gru: bool, amp: bool):
    torch.backends.cudnn.benchmark = True
    cfg_infer = OmegaConf.load(cfg_infer_path)
    cfg_network = OmegaConf.load(cfg_infer.network.config)
    device = _resolve_device(device_str)
    amp_enabled = amp and device.type == 'cuda'

    net_conf = dict(cfg_network['network_config'])
    for key in ("use_delta_gru", "delta_gru_threshold_x", "delta_gru_threshold_h", "delta_gru_thresholds",
                "log_gru_inputs", "log_file"):
        override_val = cfg_infer.network.get(key, None)
        if override_val is not None:
            net_conf[key] = override_val
    if log_gru_inputs:
        net_conf['log_gru_inputs'] = True
        net_conf['log_file'] = log_file
    if custom_gru:
        net_conf['use_custom_gru'] = True
    model = Model(**net_conf).to(device)
    checkpoint = torch.load(cfg_infer.network.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    enh_folder = cfg_infer.network.enh_folder
    os.makedirs(enh_folder, exist_ok=True)
    return model, device, amp_enabled, enh_folder, cfg_infer


def _process_batch(batch, cfg_infer_path: str, device_str: str, log_gru_inputs: bool, log_file: str,
                   custom_gru: bool, no_copy: bool, amp: bool):
    model, device, amp_enabled, enh_folder, _ = _load_model(
        cfg_infer_path, device_str, log_gru_inputs, log_file, custom_gru, amp
    )
    results = []
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=amp_enabled)
    for task in batch:
        noisy_path = task["noisy_path"]
        clean_path = task["clean_path"]
        uid = task["uid"]
        noisy, fs = sf.read(noisy_path, dtype='float32')
        input = torch.as_tensor(noisy, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.inference_mode(), autocast_ctx:
            output = model(input)
        enhanced = output.cpu().detach().numpy().squeeze()
        enh_path = os.path.join(enh_folder, uid + f"_enh.wav")
        sf.write(enh_path, enhanced, fs)
        if not no_copy:
            noisy_copy_path = os.path.join(enh_folder, uid + "_noisy.wav")
            clean_copy_path = os.path.join(enh_folder, uid + "_clean.wav")
            shutil.copy2(noisy_path, noisy_copy_path)
            if os.path.isfile(clean_path):
                shutil.copy2(clean_path, clean_copy_path)
        results.append((uid, enh_path, clean_path))
    return results

def main(args):
    torch.backends.cudnn.benchmark = True
    cfg_infer = OmegaConf.load(args.config)
    cfg_network = OmegaConf.load(cfg_infer.network.config)
    
    noisy_folder = cfg_infer.test_dataset.noisy_dir
    clean_folder = cfg_infer.test_dataset.clean_dir
    dataset_cfg = cfg_infer.get('dataset', None)
    enh_folder = cfg_infer.network.enh_folder
    os.makedirs(enh_folder, exist_ok=True)
    
    # Device and AMP setup only needed for single-worker path; multi-worker loads inside workers
    device = _resolve_device(args.device)
    amp_enabled = args.amp and device.type == 'cuda'
    
    inf_scp_list = []
    ref_scp_list = []

    if dataset_cfg is not None and (not noisy_folder):
        # Use the dataloader to prepare/load dataset (same logic as training)
        ds_conf = dict(dataset_cfg)
        # Convert null to None
        if ds_conf.get('dataset_root', None) is None:
            ds_conf['dataset_root'] = None
        test_dataset = Dataset(**ds_conf)
        examples = test_dataset.examples
        fs = test_dataset.fs
        iterator = examples
    else:
        # fall back to explicit folders
        noisy_wavs = sorted(list(filter(lambda x: x.endswith("wav"), os.listdir(noisy_folder))))
        iterator = [(n, os.path.join(clean_folder, n)) for n in noisy_wavs]

    if args.max_files is not None and args.max_files > 0:
        iterator = list(iterator)[:args.max_files]
    total = len(iterator) if hasattr(iterator, "__len__") else None

    tasks = []
    for item in iterator:
        if dataset_cfg is not None and (not noisy_folder):
            noisy_path, clean_path = item
        else:
            noisy_fname, clean_path = item
            noisy_path = os.path.join(noisy_folder, noisy_fname)
        tasks.append(
            {
                "noisy_path": noisy_path,
                "clean_path": clean_path,
                "uid": Path(noisy_path).stem,
            }
        )

    # Prevent CUDA + forked workers; fall back to single worker if GPU selected
    worker_count = args.workers or 1
    if device.type == "cuda" and worker_count > 1:
        print("[infer] CUDA with workers>1 not supported; falling back to workers=1", flush=True)
        worker_count = 1

    if worker_count > 1:
        # Run batches in parallel; each worker loads its own model
        batch_size = math.ceil(len(tasks) / worker_count)
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            futures = [
                ex.submit(
                    _process_batch,
                    batch,
                    args.config,
                    args.device,
                    args.log_gru_inputs,
                    args.log_file,
                    args.custom_gru,
                    args.no_copy,
                    args.amp if device.type == "cuda" else False,
                )
                for batch in batches
            ]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                results = fut.result()
                for uid, enh_path, ref_path in results:
                    inf_scp_list.append([uid, enh_path])
                    ref_scp_list.append([uid, ref_path])
    else:
        # Single worker path reuses one model instance
        # Allow overriding the network config to enable GRU input logging during inference
        net_conf = dict(cfg_network['network_config'])
        # Optional DeltaGRU overrides from cfg_infer.yaml
        for key in ("use_delta_gru", "delta_gru_threshold_x", "delta_gru_threshold_h", "delta_gru_thresholds",
                    "log_gru_inputs", "log_file"):
            override_val = cfg_infer.network.get(key, None)
            if override_val is not None:
                net_conf[key] = override_val
        if args.log_gru_inputs:
            net_conf['log_gru_inputs'] = True
            net_conf['log_file'] = args.log_file
        if args.custom_gru:
            net_conf['use_custom_gru'] = True
        model = Model(**net_conf).to(device)
        checkpoint = torch.load(cfg_infer.network.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=amp_enabled)
        for task in tqdm(tasks, total=total):
            noisy_path = task["noisy_path"]
            clean_path = task["clean_path"]
            uid = task["uid"]

            noisy, fs = sf.read(noisy_path, dtype='float32')

            input = torch.as_tensor(noisy, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.inference_mode(), autocast_ctx:
                output = model(input)
            enhanced = output.cpu().detach().numpy().squeeze()

            enh_path = os.path.join(enh_folder, uid + f"_enh.wav")
            ref_path = clean_path

            inf_scp_list.append([uid, enh_path])
            ref_scp_list.append([uid, ref_path])

            sf.write(enh_path, enhanced, fs)

            if not args.no_copy:
                noisy_copy_path = os.path.join(enh_folder, uid + "_noisy.wav")
                clean_copy_path = os.path.join(enh_folder, uid + "_clean.wav")
                shutil.copy2(noisy_path, noisy_copy_path)
                if os.path.isfile(clean_path):
                    shutil.copy2(clean_path, clean_copy_path)
    
    # Save paths into scp file for evaluation
    with open(os.path.join(enh_folder, "inf.scp"), "w") as f:
        for uid, audio_path in inf_scp_list:
            f.write(f"{uid} {audio_path}\n")

    with open(os.path.join(enh_folder, "ref.scp"), "w") as f:
        for uid, audio_path in ref_scp_list:
            f.write(f"{uid} {audio_path}\n")
            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_infer.yaml')
    parser.add_argument('-D', '--device', default='0', help='Index of the gpu device')
    parser.add_argument('--log-gru-inputs', action='store_true', dest='log_gru_inputs',
                        help='Enable dumping of GRU inputs during inference')
    parser.add_argument('--log-file', type=str, default='gru_inputs.pkl',
                        help='Path to output pickle file for GRU inputs')
    parser.add_argument('--custom-gru', action='store_true',
                        help='Use the handcrafted GRU implementation for GTCRN inference.')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Process at most this many files (for quick sweeps/debug).')
    parser.add_argument('--no-copy', action='store_true',
                        help='Skip copying noisy/clean wavs to the enhancement folder (reduces I/O).')
    parser.add_argument('--amp', action='store_true',
                        help='Enable autocast mixed precision on CUDA for faster inference.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel inference workers (each loads its own model).')

    args = parser.parse_args()
    main(args)
