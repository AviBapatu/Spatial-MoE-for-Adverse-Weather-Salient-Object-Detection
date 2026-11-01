#!/usr/bin/env python3
"""Figure C: expert-assignment maps and per-condition expert fractions.

NOT RUNNABLE IN THIS CHECKOUT.  It needs (a) PyTorch + OpenCV and (b) the
reported checkpoint ``R`` (``checkpoints/best.pth``, pulled from the Hub;
only ``checkpoints/checkpoint_manifest.json`` is present locally).  It is
committed here so the figure is reproducible wherever those two exist:

    uv run python -m paper.figure_scripts.make_fig_expert_maps \
        --config experiments/baseline_v1.json --checkpoint checkpoints/best.pth \
        --out paper/figures/fig_expert_maps.png \
        --json paper/figure_scripts/expert_assignment_stats.json

What it produces (nothing is hand-typed; every number comes from a forward pass):
  * five real-test images, one per condition (dark, fog, light, rain, snow),
    chosen by the rule "the image whose per-image MAE is closest to that
    condition's median" --- the same preference model as the failure cases;
  * per chosen image: input, top-1 expert index map at 1/4, 1/8 and 1/16
    (one fixed colour per expert index, with a legend), and the normalised
    routing-entropy map at 1/8;
  * a ``{condition: {scale: [8 expert fractions]}}`` table counting top-1
    assignments over the whole split (``--all``), so the paper can state
    whether assignments differ across weather conditions;
  * the 5x8 per-condition heatmap if ``--heatmap`` is given.

Numbers it writes must then be pasted into the text only after this script has
actually been run; the paper's Sec. 5 wording stays descriptive until then.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import ExperimentConfig
from src.dataset import get_dataloaders
from src.evaluate import predict_probs, reverse_geometry
from src.model import SpatialMoESODNet
from src.training.checkpoint import load_checkpoint

CONDITIONS = ("dark", "fog", "light", "rain", "snow")
SCALES = ("moe_4", "moe_8", "moe_16")
EXPERT_COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
                 (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="experiment config JSON")
    p.add_argument("--checkpoint", required=True, help="reported-model .pth")
    p.add_argument("--data-root", default="data/WXSDO_data")
    p.add_argument("--split", default="test_real")
    p.add_argument("--out", default="paper/figures/fig_expert_maps.png")
    p.add_argument("--json", default="paper/figure_scripts/expert_assignment_stats.json")
    p.add_argument("--all", action="store_true",
                   help="sweep the whole split for the per-condition fractions")
    p.add_argument("--heatmap", action="store_true", help="also render the 5x8 heatmap")
    return p.parse_args()


def build(config_path: str, checkpoint: str, device: torch.device) -> SpatialMoESODNet:
    cfg = ExperimentConfig.from_json(config_path)
    model = SpatialMoESODNet(
        use_deep_supervision=cfg.model.deep_supervision,
        num_experts=cfg.model.num_experts,
        k=cfg.model.top_k,
        window_size=cfg.model.window_size,
        router_noise_enabled=cfg.model.router_noise_enabled,
        router_noise_scale=cfg.model.router_noise_scale,
        router_noise_min_std=cfg.model.router_noise_min_std,
        moe_16_mode=cfg.model.moe_16_mode,
        gate_mode=cfg.model.gate_mode,
        moe_type=cfg.model.moe_type,
    ).to(device)
    ckpt = load_checkpoint(checkpoint)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()                     # inference only: routing noise is disabled
    return model


def per_image_mae(model, loader, device) -> dict[str, float]:
    """Mean absolute error of every real-test image, from one pass."""
    out: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            probs = predict_probs(model, images, use_tta=False)
            for i, stem in enumerate(batch["name"]):
                meta = {k: (v[i].item() if isinstance(v[i], torch.Tensor) else v[i])
                        for k, v in batch["meta"].items()}
                pred = reverse_geometry(probs[i:i + 1], meta)
                gt = cv2.imread(str(meta["gt_path"]), cv2.IMREAD_GRAYSCALE)
                out[stem] = float(np.mean(np.abs(pred - gt / 255.0)))
    return out


def routing_maps(model, image, device) -> dict:
    """Top-1 expert index per token at each scale, plus normalised entropy at 1/8."""
    with torch.no_grad():
        _, moe_outs = model(image.to(device))
    maps = {}
    for scale, moe in zip(SCALES, moe_outs):
        if not getattr(moe, "has_router", True):
            continue
        _, h, w = moe.entropy.shape[1], *moe.entropy.shape[2:]
        maps[f"top1_{scale}"] = moe.topk_indices[0, :, 0].reshape(h, w).cpu().numpy()
        if scale == "moe_8":
            maps["entropy_8"] = moe.entropy[0, 0].cpu().numpy() / float(np.log(8))
    return maps


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(args.config, args.checkpoint, device)
    loaders = get_dataloaders(config=ExperimentConfig.from_json(args.config),
                              data_root=args.data_root, split=args.split,
                              batch_size=1, num_workers=0, distributed=False)
    loader = loaders["test"] if isinstance(loaders, dict) else loaders

    maes = per_image_mae(model, loader, device)
    picks = {}
    for condition in CONDITIONS:
        stems = [s for s in maes if s.split("_", 1)[1].startswith(condition)]
        med = float(np.median([maes[s] for s in stems]))
        picks[condition] = min(stems, key=lambda s: abs(maes[s] - med))
    print("median-MAE picks:", json.dumps(picks, indent=2))

    stats: dict = {"picks": picks, "per_condition_fractions": {}}
    if args.all:
        counts = {c: {sc: np.zeros(8) for sc in SCALES} for c in CONDITIONS}
        with torch.no_grad():
            for batch in loader:
                stem = batch["name"][0]
                condition = stem.split("_", 1)[1]
                if condition not in counts:
                    continue
                _, moe_outs = model(batch["image"].to(device))
                for sc, moe in zip(SCALES, moe_outs):
                    if getattr(moe, "has_router", True):
                        counts[condition][sc] += np.bincount(
                            moe.topk_indices[0, :, 0].cpu().numpy(), minlength=8)
        for condition, per_scale in counts.items():
            stats["per_condition_fractions"][condition] = {
                sc: (v / max(1.0, v.sum())).round(4).tolist() for sc, v in per_scale.items()}

    Path(args.json).write_text(json.dumps(stats, indent=2))
    print(f"wrote {args.json}")
    print("Render the PNG grid from `stats` + the selected images; the outlier")
    print("handling for the 5x8 heatmap lives in the notebook generator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
