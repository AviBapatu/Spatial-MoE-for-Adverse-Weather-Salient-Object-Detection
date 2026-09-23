"""Standalone MoE routing diagnostics runner.

Runs two diagnostic passes over each test split in a single forward loop:

1. **padded-baseline** — all spatial tokens included (reproduces the historical
   ``routing_stats_ep*`` behaviour).
2. **content-only** — padding tokens excluded via a downsampled ``pad_mask``
   derived from each image's aspect-preserving resize/pad metadata.

Output files (per dataset split under ``evaluation/diagnostics_{ds_name}/``):

    routing_stats_padded_baseline.json / .csv
    routing_stats_contentonly.json / .csv
    expert_similarity_padded_baseline_moe_{4,8,16}.json
    expert_similarity_contentonly_moe_{4,8,16}.json

After the content-only pass the script logs the **masked-token fraction** at
each MoE stage as a sanity check that the mask is being built at the correct
spatial resolution.  Expected range: 0.15–0.50 (matching the pixel-level
padded-pixel-fraction already measured in ``evaluation/padding_stats.json``).
"""

import argparse
import json
import os

import torch
from torch.amp import autocast
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.diagnostics import (
    ExpertSimilarityAnalyzer,
    MoEDiagnosticsEngine,
    build_scale_pad_mask,
)
from src.log import get_logger
from src.model import SpatialMoESODNet

log = get_logger(__name__)

# Sanity-check bounds: masked fraction must fall in this range at every scale.
_MASK_FRAC_LO = 0.15
_MASK_FRAC_HI = 0.50


def _run_dataset(
    ds_name: str,
    loader: torch.utils.data.DataLoader,
    model: SpatialMoESODNet,
    device: torch.device,
    num_experts: int,
    top_k: int,
    ds_out_dir: str,
) -> list[str]:
    """Run both diagnostic passes on one dataset split and return warnings.

    Args:
        ds_name: human-readable split label (used only for logging).
        loader: DataLoader for the split.
        model: model in eval mode.
        device: inference device.
        num_experts: expert count (from checkpoint config).
        top_k: router top-k (from checkpoint config).
        ds_out_dir: directory where outputs are written.

    Returns:
        List of accumulated warning strings (routing collapse + redundant pairs).
    """
    os.makedirs(ds_out_dir, exist_ok=True)

    # Two accumulators — same forward pass feeds both.
    diag_baseline = MoEDiagnosticsEngine(ds_out_dir, num_experts=num_experts, top_k=top_k)
    diag_content = MoEDiagnosticsEngine(ds_out_dir, num_experts=num_experts, top_k=top_k)

    last_v_images: torch.Tensor | None = None
    last_v_pad_masks: torch.Tensor | None = None

    with torch.no_grad():
        for v_batch in tqdm(loader, desc=f"{ds_name} Routing"):
            v_images = v_batch["image"].to(device)
            v_pad_masks = v_batch["pad_mask"].to(device)  # [B, 1, 384, 384]

            last_v_images = v_images
            last_v_pad_masks = v_pad_masks

            with autocast(device_type="cuda", dtype=torch.float16):
                out, moe_outputs = model(v_images)

            meta_list = dict(v_batch["meta"])
            meta_list["name"] = v_batch["name"]

            # Baseline: all tokens (current behaviour)
            diag_baseline.update(
                v_images, moe_outputs, meta_list,
                num_visual_samples=0,
                pad_masks=None,
            )
            # Content-only: valid tokens only
            diag_content.update(
                v_images, moe_outputs, meta_list,
                num_visual_samples=0,
                pad_masks=v_pad_masks,
            )

    # --- Finalize both runs ---
    baseline_stats = diag_baseline.finalize(run_tag="padded_baseline")
    content_stats = diag_content.finalize(run_tag="contentonly")

    # --- Sanity check: masked-token fraction per scale ---
    log.info(f"\n[{ds_name}] === MASKED-TOKEN FRACTION SANITY CHECK ===")
    sanity_ok = True
    for scale in ["moe_4", "moe_8", "moe_16"]:
        tracker = diag_content.trackers[scale]
        observed = tracker.total_tokens + tracker.masked_tokens
        frac = tracker.masked_tokens / observed if observed > 0 else 0.0
        in_range = _MASK_FRAC_LO <= frac <= _MASK_FRAC_HI
        status = "✓" if in_range else "✗ OUT OF RANGE"
        log.info(
            f"  {scale}  masked_fraction={frac:.4f}  "
            f"(expected {_MASK_FRAC_LO:.2f}–{_MASK_FRAC_HI:.2f})  {status}"
        )
        if not in_range:
            sanity_ok = False
    if not sanity_ok:
        log.warning(
            f"[{ds_name}] Masked-token fraction is outside the expected range at "
            f"one or more scales — the pad_mask may be built at the wrong resolution. "
            f"Content-only outputs may not be trustworthy until this is investigated."
        )

    # --- Collect routing-collapse warnings ---
    all_warnings: list[str] = []
    for tag, stats in [("padded_baseline", baseline_stats), ("contentonly", content_stats)]:
        for scale in ["moe_4", "moe_8", "moe_16"]:
            warnings = stats[scale].get("warnings", [])
            for w in warnings:
                msg = f"[{ds_name}][{tag}] {scale}: {w}"
                log.info(msg)
                all_warnings.append(msg)

    # --- Expert similarity analysis ---
    # Build backbone features from the last batch (same as original runner).
    log.info(f"Running ExpertSimilarityAnalyzer on {ds_name} (using last batch)...")
    assert last_v_images is not None and last_v_pad_masks is not None

    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.float16):
            feats = model.backbone(last_v_images)

    scale_configs = [
        ("moe_4",  model.moe_4,  feats["res_4"].float(),  4),
        ("moe_8",  model.moe_8,  feats["res_8"].float(),  8),
        ("moe_16", model.moe_16, feats["res_16"].float(), 16),
    ]

    for scale_name, moe_layer, feat_tensor, stride in scale_configs:
        if getattr(moe_layer, 'is_dense', False):
            continue

        analyzer = ExpertSimilarityAnalyzer(moe_layer, num_experts=num_experts)

        # Build per-scale mask from the last batch's pad_mask.
        scale_pm = build_scale_pad_mask(last_v_pad_masks, stride=stride)

        # Baseline: all tokens (no mask)
        baseline_path = os.path.join(
            ds_out_dir, f"expert_similarity_padded_baseline_{scale_name}.json"
        )
        res_base = analyzer.analyze(feat_tensor, baseline_path, pad_mask=None)
        for w in res_base.get("warnings", []):
            all_warnings.append(f"[{ds_name}][padded_baseline] {scale_name}: {w}")

        # Content-only: filter padding tokens
        content_path = os.path.join(
            ds_out_dir, f"expert_similarity_contentonly_{scale_name}.json"
        )
        res_cont = analyzer.analyze(feat_tensor, content_path, pad_mask=scale_pm)
        for w in res_cont.get("warnings", []):
            all_warnings.append(f"[{ds_name}][contentonly] {scale_name}: {w}")

        # Log the delta so it's immediately visible in the run log.
        base_vals = [
            res_base["activation_similarity"][i][j]
            for i in range(num_experts)
            for j in range(i + 1, num_experts)
        ]
        cont_vals = [
            res_cont["activation_similarity"][i][j]
            for i in range(num_experts)
            for j in range(i + 1, num_experts)
        ]
        mean_base = sum(base_vals) / len(base_vals) if base_vals else 0.0
        mean_cont = sum(cont_vals) / len(cont_vals) if cont_vals else 0.0
        log.info(
            f"  {scale_name}  mean_activation_sim: "
            f"padded_baseline={mean_base:.4f}  contentonly={mean_cont:.4f}  "
            f"delta={mean_cont - mean_base:+.4f}"
        )

    return all_warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/WXSOD_data")
    parser.add_argument("--out_dir", type=str, default="evaluation")
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for diagnostics dataloader (default 4 — safe on 16 GB GPU)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("config", {}).get("model", {})

    use_deep_supervision = model_cfg.get("deep_supervision", False)
    num_experts = model_cfg.get("num_experts", 6)
    window_size = model_cfg.get("window_size", 8)
    router_noise_enabled = model_cfg.get("router_noise_enabled", True)
    router_noise_scale = model_cfg.get("router_noise_scale", 1.0)
    router_noise_min_std = model_cfg.get("router_noise_min_std", 0.05)
    top_k = model_cfg.get("top_k", 2)

    log.info("--- CHECKPOINT CONFIG ---")
    log.info(f"num_experts: {num_experts}")
    log.info(f"top_k: {top_k}")
    log.info(f"window_size: {window_size}")
    log.info(f"deep_supervision: {use_deep_supervision}")
    log.info(f"router_noise_enabled: {router_noise_enabled}")
    log.info(f"router_noise_scale: {router_noise_scale}")
    log.info(f"router_noise_min_std: {router_noise_min_std}")
    log.info("-------------------------\n")

    model = SpatialMoESODNet(
        use_deep_supervision=use_deep_supervision,
        num_experts=num_experts,
        k=top_k,
        gate_mode=model_cfg.get("gate_mode", "renormalized"),
        window_size=window_size,
        moe_16_mode=model_cfg.get("moe_16_mode", "sparse"),
        moe_type=model_cfg.get("moe_type", "sparse"),
        router_noise_enabled=router_noise_enabled,
        router_noise_scale=router_noise_scale,
        router_noise_min_std=router_noise_min_std,
    ).to(device)

    # Router-less arms (moe_type dense/none) have no per-expert contract to check.
    if not getattr(model.moe_4, "is_dense", False):
        assert num_experts == model.moe_4.num_experts, (
            f"Mismatch: config num_experts={num_experts}, "
            f"model.moe_4.num_experts={model.moe_4.num_experts}"
        )
        assert top_k == model.moe_4.k, (
            f"Mismatch: config top_k={top_k}, model.moe_4.k={model.moe_4.k}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.cuda.empty_cache()

    _, _, test_sys_loader, test_real_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        distributed=False,
    )

    datasets_to_run = [("test_real", test_real_loader), ("test_sys", test_sys_loader)]

    all_warnings: list[str] = []
    for ds_name, loader in datasets_to_run:
        log.info(f"\n{'='*60}")
        log.info(f"Running diagnostics on {ds_name}...")
        ds_out_dir = os.path.join(args.out_dir, f"diagnostics_{ds_name}")

        warnings = _run_dataset(
            ds_name=ds_name,
            loader=loader,
            model=model,
            device=device,
            num_experts=num_experts,
            top_k=top_k,
            ds_out_dir=ds_out_dir,
        )
        all_warnings.extend(warnings)
        torch.cuda.empty_cache()

    log.info("\n--- DIAGNOSTICS SUMMARY ---")
    for w in all_warnings:
        log.info(w)
    if not all_warnings:
        log.info("No WARNINGS or REDUNDANT_PAIR found.")

    # Write a manifest of output files for traceability.
    manifest = {
        "checkpoint": args.checkpoint,
        "output_files": []
    }
    for ds_name, _ in datasets_to_run:
        ds_out_dir = os.path.join(args.out_dir, f"diagnostics_{ds_name}")
        for fname in sorted(os.listdir(ds_out_dir)):
            if fname.endswith((".json", ".csv")):
                manifest["output_files"].append(os.path.join(ds_out_dir, fname))

    manifest_path = os.path.join(args.out_dir, "diagnostics_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    log.info(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
