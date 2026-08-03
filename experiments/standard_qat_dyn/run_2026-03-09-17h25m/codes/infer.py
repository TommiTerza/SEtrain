import os
import shutil
import torch
import soundfile as sf
from collections.abc import Mapping
from tqdm import tqdm
from omegaconf import OmegaConf
from models.gtcrn_end2end import GTCRN as Model

def main(args):
    cfg_infer = OmegaConf.load(args.config)
    cfg_network = OmegaConf.load(cfg_infer.network.config)
    qat_config = cfg_network['qat'] if 'qat' in cfg_network else {}
    qat_enabled = bool(qat_config.get('enabled', False))
    qat_gru = bool(qat_config.get('qat_gru', False))
    dynamic_quantize_gru = bool(qat_config.get('dynamic_quantize_gru', False))
    static_quantize_gru = bool(qat_config.get('static_quantize_gru', False))
    if dynamic_quantize_gru and static_quantize_gru:
        raise ValueError("qat.dynamic_quantize_gru and qat.static_quantize_gru are mutually exclusive.")
    scale_constraint_cfg = qat_config.get("scale_constraint", {})
    if isinstance(scale_constraint_cfg, Mapping):
        scale_constraint_mode = scale_constraint_cfg.get("mode", qat_config.get("scale_constraint_mode", "none"))
        scale_constraint_frac_bits = scale_constraint_cfg.get(
            "frac_bits", qat_config.get("scale_constraint_frac_bits", None)
        )
        scale_constraint_pow2_rounding = scale_constraint_cfg.get(
            "pow2_rounding", qat_config.get("scale_constraint_pow2_rounding", "nearest")
        )
    else:
        scale_constraint_mode = scale_constraint_cfg if scale_constraint_cfg is not None else qat_config.get("scale_constraint_mode", "none")
        scale_constraint_frac_bits = qat_config.get("scale_constraint_frac_bits", None)
        scale_constraint_pow2_rounding = qat_config.get("scale_constraint_pow2_rounding", "nearest")
    use_int8 = bool(cfg_infer.network.get('use_int8', False))
    if use_int8 and not qat_enabled:
        raise ValueError("network.use_int8=True requires a QAT-trained config with qat.enabled=True.")
    
    noisy_folder = cfg_infer.test_dataset.noisy_dir
    clean_folder = cfg_infer.test_dataset.clean_dir
    enh_folder = cfg_infer.network.enh_folder
    os.makedirs(enh_folder, exist_ok=True)
    
    device_arg = str(args.device).strip().lower()
    force_cpu = device_arg in {"cpu", "none", "-1"}
    if use_int8 or force_cpu:
        device = torch.device("cpu")
    else:
        if "," in device_arg:
            raise ValueError("--device must be a single GPU index (e.g. 0) or 'cpu'")
        if not device_arg.isdigit():
            raise ValueError("--device must be a single GPU index (e.g. 0) or 'cpu'")
        device = torch.device(f"cuda:{int(device_arg)}" if torch.cuda.is_available() else "cpu")

    model = Model(**cfg_network['network_config']).to(device)
    if qat_enabled:
        model.prepare_qat(
            backend=qat_config.get('backend', 'fbgemm'),
            quantize_deconv=bool(qat_config.get('quantize_deconv', False)),
            per_channel_weights=bool(qat_config.get('per_channel_weights', False)),
            quantize_linear=bool(qat_config.get("quantize_linear", False)),
            quantize_gru=qat_gru,
            scale_constraint_mode=scale_constraint_mode,
            scale_constraint_frac_bits=scale_constraint_frac_bits,
            scale_constraint_pow2_rounding=scale_constraint_pow2_rounding,
        )
    checkpoint = torch.load(cfg_infer.network.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if use_int8:
        model = model.cpu().convert_qat(
            inplace=True,
            dynamic_quantize_gru=dynamic_quantize_gru,
            static_quantize_gru=static_quantize_gru,
            dynamic_gru_scale_constraint_mode=scale_constraint_mode,
            dynamic_gru_scale_constraint_frac_bits=scale_constraint_frac_bits,
            dynamic_gru_scale_constraint_pow2_rounding=scale_constraint_pow2_rounding,
        )
    model.eval()
    
    noisy_wavs = sorted(list(filter(lambda x: x.endswith("wav"), os.listdir(noisy_folder))))

    inf_scp_list = []
    ref_scp_list = []
    for wav_name in tqdm(noisy_wavs):
        noisy, fs = sf.read(os.path.join(noisy_folder, wav_name), dtype='float32')
        
        input = torch.FloatTensor(noisy).unsqueeze(0).to(device)
        with torch.inference_mode():
            output  = model(input)
        enhanced = output.cpu().detach().numpy().squeeze()
        
        uid = wav_name.split(".wav")[0]
        enh_path = os.path.join(enh_folder, uid + f"_enh.wav")
        ref_path = os.path.join(clean_folder, wav_name)
        
        inf_scp_list.append([uid, enh_path])
        ref_scp_list.append([uid, ref_path])
        
        sf.write(enh_path, enhanced, fs)
    
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
    parser.add_argument('-D', '--device', default='0', help="GPU index (e.g. 0) or 'cpu' to force CPU")

    args = parser.parse_args()
    main(args)
