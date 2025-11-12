import os
import shutil
import torch
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf
from models.gtcrn_end2end import GTCRN as Model
from dataloader import VoiceBankDemandDataset as Dataset

def main(args):
    cfg_infer = OmegaConf.load(args.config)
    cfg_network = OmegaConf.load(cfg_infer.network.config)
    
    noisy_folder = cfg_infer.test_dataset.noisy_dir
    clean_folder = cfg_infer.test_dataset.clean_dir
    dataset_cfg = cfg_infer.get('dataset', None)
    enh_folder = cfg_infer.network.enh_folder
    os.makedirs(enh_folder, exist_ok=True)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Allow overriding the network config to enable GRU input logging during inference
    net_conf = dict(cfg_network['network_config'])
    if args.log_gru_inputs:
        net_conf['log_gru_inputs'] = True
        net_conf['log_file'] = args.log_file
    if args.custom_gru:
        net_conf['use_custom_gru'] = True
    model = Model(**net_conf).to(device)
    checkpoint = torch.load(cfg_infer.network.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
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

    for item in tqdm(iterator):
        if dataset_cfg is not None and (not noisy_folder):
            noisy_path, clean_path = item
        else:
            noisy_fname, clean_path = item
            noisy_path = os.path.join(noisy_folder, noisy_fname)

        noisy, fs = sf.read(noisy_path, dtype='float32')

        input = torch.FloatTensor(noisy).unsqueeze(0).to(device)
        with torch.inference_mode():
            output  = model(input)
        enhanced = output.cpu().detach().numpy().squeeze()

        uid = Path(noisy_path).stem
        enh_path = os.path.join(enh_folder, uid + f"_enh.wav")
        ref_path = clean_path

        inf_scp_list.append([uid, enh_path])
        ref_scp_list.append([uid, ref_path])

        sf.write(enh_path, enhanced, fs)

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

    args = parser.parse_args()
    main(args)
