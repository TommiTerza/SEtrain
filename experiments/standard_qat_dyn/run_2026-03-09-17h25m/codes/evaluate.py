import os
from omegaconf import OmegaConf


def main(args):
    config = OmegaConf.load(args.config)
    enh_folder = config.network.enh_folder
    # enh_folder = '/data/ssd0/xiaobin.rong/Datasets/DNS3/test_noisy/'
    
    if args.metric == 'dnsmos':
        dnsmos_device = "cpu" if getattr(args, "force_cpu", False) else "cuda"
        os.system(
            ('python ./evaluation/calculate_nonintrusive_dnsmos.py '
                f'--inf_scp {enh_folder}/inf.scp '
                f'--output_dir {enh_folder}/scoring_dnsmos '
                f'--device {dnsmos_device} '
                '--job 1 '
                '--convert_to_torch True '
                '--primary_model ./DNSMOS/DNSMOS/sig_bak_ovr.onnx '
                '--p808_model ./DNSMOS/DNSMOS/model_v8.onnx'
            )
        )    
    elif args.metric == 'intrusive':
        os.system(
            ('python ./evaluation/calculate_intrusive_se_metrics.py '
             f'--ref_scp {enh_folder}/ref.scp '
             f'--inf_scp {enh_folder}/inf.scp '
             f'--output_dir {enh_folder}/scoring_intrusive '
             '--nj 8 '
             '--chunksize 1000'
            )
        )

    else:
        raise ValueError
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', required=True, help="Metric to be calculated")
    parser.add_argument('--config', default='configs/cfg_infer.yaml')
    parser.add_argument('--device', default='0', help="GPU indices (e.g. 0 or 0,1) or 'cpu' to force CPU")
    
    args = parser.parse_args()
    device_arg = str(args.device).strip()
    if device_arg.lower() in {"cpu", "none", "-1"}:
        args.force_cpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        parts = [p.strip() for p in device_arg.split(",") if p.strip()]
        if not parts or any(not p.isdigit() for p in parts):
            raise ValueError("--device must be GPU indices like '0' or '0,1' or 'cpu'")
        args.force_cpu = False
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(parts)
    main(args)
