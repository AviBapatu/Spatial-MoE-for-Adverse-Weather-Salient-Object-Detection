import argparse
import os

import torch
from torch.amp import autocast
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.diagnostics import ExpertSimilarityAnalyzer, MoEDiagnosticsEngine
from src.log import get_logger
from src.model import SpatialMoESODNet

log = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/WXSOD_data')
    parser.add_argument('--out_dir', type=str, default='evaluation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for diagnostics dataloader (default 4 — safe on 16 GB GPU)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_cfg = checkpoint.get('config', {}).get('model', {})

    use_deep_supervision = model_cfg.get('deep_supervision', False)
    num_experts = model_cfg.get('num_experts', 6)
    window_size = model_cfg.get('window_size', 8)
    router_noise_enabled = model_cfg.get('router_noise_enabled', True)
    router_noise_scale = model_cfg.get('router_noise_scale', 1.0)
    router_noise_min_std = model_cfg.get('router_noise_min_std', 0.05)

    log.info("--- CHECKPOINT CONFIG ---")
    log.info(f"num_experts: {num_experts}")
    log.info(f"window_size: {window_size}")
    log.info(f"deep_supervision: {use_deep_supervision}")
    log.info(f"router_noise_enabled: {router_noise_enabled}")
    log.info(f"router_noise_scale: {router_noise_scale}")
    log.info(f"router_noise_min_std: {router_noise_min_std}")
    log.info("-------------------------\n")

    model = SpatialMoESODNet(
        use_deep_supervision=use_deep_supervision,
        num_experts=num_experts,
        window_size=window_size,
        router_noise_enabled=router_noise_enabled,
        router_noise_scale=router_noise_scale,
        router_noise_min_std=router_noise_min_std
    ).to(device)

    assert num_experts == model.moe_4.num_experts, f"Mismatch: config num_experts={num_experts}, model.moe_4.num_experts={model.moe_4.num_experts}"

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    torch.cuda.empty_cache()  # free fragmented reserved memory before inference loop

    _, _, test_sys_loader, test_real_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        distributed=False
    )

    datasets_to_run = [('test_real', test_real_loader), ('test_sys', test_sys_loader)]

    all_warnings = []

    for ds_name, loader in datasets_to_run:
        log.info(f"Running diagnostics on {ds_name}...")
        ds_out_dir = os.path.join(args.out_dir, f"diagnostics_{ds_name}")
        os.makedirs(ds_out_dir, exist_ok=True)

        diag_engine = MoEDiagnosticsEngine(
            ds_out_dir,
            num_experts=num_experts,
            top_k=model_cfg.get('top_k', 2)
        )

        last_v_images = None
        with torch.no_grad():
            for d_idx, v_batch in enumerate(tqdm(loader, desc=f"{ds_name} Routing")):
                v_images = v_batch['image'].to(device)
                last_v_images = v_images
                with autocast(device_type='cuda', dtype=torch.float16):
                    out, moe_outputs = model(v_images)

                # 'name' is a top-level batch key, not inside 'meta' — merge it in
                meta_list = dict(v_batch['meta'])
                meta_list['name'] = v_batch['name']  # list of stems, e.g. ["scene1_fog", ...]
                diag_engine.update(v_images, moe_outputs, meta_list, num_visual_samples=0)

        stats = diag_engine.finalize(epoch=0)
        torch.cuda.empty_cache()  # release activations before similarity analyzer

        # Print warnings to stdout per scale
        for scale in ['moe_4', 'moe_8', 'moe_16']:
            warnings = stats[scale].get('warnings', [])
            if warnings:
                log.info(f"[{ds_name}] {scale} WARNINGS: {warnings}")
                all_warnings.extend([f"[{ds_name}] {scale}: {w}" for w in warnings])

        # Run ExpertSimilarityAnalyzer
        log.info(f"Running ExpertSimilarityAnalyzer on {ds_name} (using last batch)...")
        # Get intermediate features for the last batch
        with torch.no_grad():
            with autocast(device_type='cuda', dtype=torch.float16):
                feats = model.backbone(last_v_images)

        for scale_name, moe_layer, feat_tensor in zip(
            ['moe_4', 'moe_8', 'moe_16'],
            [model.moe_4, model.moe_8, model.moe_16],
            [feats['res_4'].float(), feats['res_8'].float(), feats['res_16'].float()]
        ):
            analyzer = ExpertSimilarityAnalyzer(moe_layer, num_experts=num_experts)
            sim_out_path = os.path.join(ds_out_dir, f"expert_similarity_{scale_name}.json")
            res = analyzer.analyze(feat_tensor, sim_out_path)

            for w in res.get('warnings', []):
                all_warnings.append(f"[{ds_name}] {scale_name}: {w}")

    log.info("\n--- DIAGNOSTICS SUMMARY ---")
    for w in all_warnings:
        log.info(w)
    if not all_warnings:
        log.info("No WARNINGS or REDUNDANT_PAIR found.")

if __name__ == '__main__':
    main()

