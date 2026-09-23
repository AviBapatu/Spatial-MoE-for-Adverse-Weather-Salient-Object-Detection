"""Offline SOD evaluation: predict, reverse geometry, and aggregate metrics.

Public API surface (kept stable for the notebook and ``generate_notebook.py``):
:func:`evaluate`, :func:`reverse_geometry`, :func:`verify_parameter_consistency`.
The per-batch forward/TTA logic and per-sample bookkeeping live in small
helpers (:func:`predict_probs`, :func:`_evaluate_sample`) so the loop stays
readable and easily unit-testable with a stub model.
"""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.boundary import BOUNDARY_KERNEL_SIZE, compute_boundary
from src.dataset import get_dataloaders, get_weather_type
from src.log import NumpyEncoder, get_logger
from src.metrics import BoundaryMetrics, SODMetrics
from src.model import SpatialMoESODNet

log = get_logger(__name__)


def verify_parameter_consistency(model: Any) -> str:
    """Return an MD5 over the sorted parameter names of ``model``.

    Used to fingerprint the checkpoint architecture build, so a config-vs-code
    drift (e.g. num_experts changed after saving) is caught before trusting
    evaluation numbers.
    """
    names = sorted([name for name, _ in model.named_parameters()])
    name_str = ",".join(names)
    return hashlib.md5(name_str.encode("utf-8")).hexdigest()


def reverse_geometry(pred_tensor: torch.Tensor, meta: Dict[str, Any]) -> np.ndarray:
    """Undo dataset geometry (padding + resize) on a single prediction.

    Args:
        pred_tensor: ``[1, 1, H, W]`` model prediction in ``[0, 1]``.
        meta: dataset meta dict with ``pad_*``, ``resized_*`` and ``orig_*`` keys.

    Returns:
        ``[orig_h, orig_w]`` float array (``[0, 1]`` probabilities).
    """
    pred = pred_tensor[0, 0].cpu().numpy()

    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]
    resized_h = meta["resized_h"]
    resized_w = meta["resized_w"]
    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]

    # 1. Unpad
    pred_cropped = pred[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]

    # 2. Resize to orig
    pred_original = cv2.resize(pred_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    assert pred_original.shape == (orig_h, orig_w)
    return pred_original


def predict_probs(
    model: Any,
    images: torch.Tensor,
    ablation_cfg: Dict[str, Any] | None = None,
    use_tta: bool = False,
) -> torch.Tensor:
    """Forward a batch and return sigmoid saliency probabilities.

    With ``use_tta`` the prediction is the mean of the forward and horizontally
    flipped forwards (flip-average TTA).

    Args:
        model: SOD model (must return ``(out, moe_outputs)`` on forward).
        images: input batch ``[B, C, H, W]``.
        ablation_cfg: optional expert-ablation config forwarded to the model.
        use_tta: whether to apply horizontal-flip test-time augmentation.

    Returns:
        ``[B, 1, H, W]`` float tensor in ``[0, 1]``.
    """
    out, _ = model(images, ablation_cfg=ablation_cfg)
    pred_prob = torch.sigmoid(out.saliency_logits)

    if not use_tta:
        return pred_prob

    # Horizontal flip
    images_flipped = torch.flip(images, dims=[-1])
    out_flipped, _ = model(images_flipped, ablation_cfg=ablation_cfg)
    pred_prob_flipped = torch.sigmoid(out_flipped.saliency_logits)
    pred_prob_unflipped = torch.flip(pred_prob_flipped, dims=[-1])
    return 0.5 * pred_prob + 0.5 * pred_prob_unflipped


def _evaluate_sample(
    batch: Dict[str, Any],
    i: int,
    pred_prob: torch.Tensor,
    output_dir: str,
    global_metrics: SODMetrics,
    weather_metrics: Dict[str, SODMetrics],
    weather_counts: Dict[str, int],
    global_boundary: BoundaryMetrics,
    weather_boundary: Dict[str, BoundaryMetrics],
) -> None:
    """Save one prediction PNG and update all global + weather-wise metrics.

    Args:
        batch: the collated batch dict (``image``, ``name``, ``meta``).
        i: index of the sample within the batch.
        pred_prob: full-batch sigmoid probs ``[B, 1, H, W]``.
        output_dir: where ``{stem}.png`` maps are written.
        global_metrics/global_boundary: aggregate metric accumulators.
        weather_metrics/weather_boundary/weather_counts: per-weather accumulators.
    """
    meta = {k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i] for k, v in batch["meta"].items()}
    stem = batch["name"][i]
    weather = get_weather_type(stem)

    # Reverse geometry
    pred_orig = reverse_geometry(pred_prob[i : i + 1], meta)

    # GT
    gt_path = str(meta.get("gt_path"))
    gt_orig = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if gt_orig is None:
        raise FileNotFoundError(f"Missing GT: {gt_path}")
    assert pred_orig.shape == gt_orig.shape, f"Shape mismatch: Pred {pred_orig.shape} vs GT {gt_orig.shape}"

    # 8-bit prediction
    pred_uint8 = np.round(np.clip(pred_orig, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{stem}.png"), pred_uint8)

    # Metrics update
    global_metrics.step(pred_uint8, gt_orig)
    weather_metrics[weather].step(pred_uint8, gt_orig)
    weather_counts[weather] += 1

    # Boundary metrics update
    gt_bin = (gt_orig > 127).astype(np.float32)
    gt_bound = compute_boundary(gt_bin)
    pred_bin = (pred_orig > 0.5).astype(np.float32)
    pred_bound = compute_boundary(pred_bin)
    global_boundary.step(pred_bound, gt_bound)
    weather_boundary[weather].step(pred_bound, gt_bound)


def evaluate(
    model: Any,
    dataloader: Any,
    output_dir: str,
    use_tta: bool = False,
    ablation_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate ``model`` over ``dataloader`` and persist prediction maps.

    Args:
        model: SOD model on its target device.
        dataloader: yields batches with ``image``, ``name`` and ``meta`` keys.
        output_dir: where ``{stem}.png`` maps and results are saved.
        use_tta: apply horizontal-flip test-time augmentation.
        ablation_cfg: optional expert-ablation config forwarded to the model.

    Returns:
        Dict with ``global`` (aggregate metrics + ``sample_count``) and
        ``weather`` (per weather class; ``LOW SAMPLE`` warning when count < 10).
    """
    model.eval()
    device = next(model.parameters()).device

    global_metrics = SODMetrics()
    global_boundary = BoundaryMetrics()

    weather_metrics = defaultdict(SODMetrics)
    weather_boundary = defaultdict(BoundaryMetrics)
    weather_counts = defaultdict(int)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=True):
            images = batch["image"].to(device)
            pred_prob = predict_probs(model, images, ablation_cfg, use_tta)

            for i in range(images.size(0)):
                _evaluate_sample(
                    batch,
                    i,
                    pred_prob,
                    output_dir,
                    global_metrics,
                    weather_metrics,
                    weather_counts,
                    global_boundary,
                    weather_boundary,
                )

    results = {
        "global": {**global_metrics.get_results(), **global_boundary.get_results(), "sample_count": sum(weather_counts.values())},
        "weather": {},
    }

    for w in sorted(weather_counts.keys()):
        count = weather_counts[w]
        w_res = {**weather_metrics[w].get_results(), **weather_boundary[w].get_results()}
        w_res["sample_count"] = count
        if count < 10:
            w_res["warning"] = "LOW SAMPLE"
        results["weather"][w] = w_res

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["test_sys", "test_real", "both"], help="Dataset to evaluate")
    parser.add_argument("--tta", type=str, default="none", choices=["none", "hflip"], help="TTA mode")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/wxsod-dataset")
    parser.add_argument("--out_dir", type=str, default="evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Checkpoint Integrity
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Extract config from checkpoint if available to initialize model properly
    model_cfg = checkpoint.get("config", {}).get("model", {})
    use_deep_supervision = model_cfg.get("deep_supervision", False)
    num_experts = model_cfg.get("num_experts", 6)  # Default to 6 if missing
    log.info("num_experts from checkpoint config: %s", num_experts)
    assert "num_experts" in model_cfg, "checkpoint config missing num_experts — verify before trusting results"
    window_size = model_cfg.get("window_size", 8)  # Default to 8 if missing
    moe_16_mode = model_cfg.get("moe_16_mode", "sparse")
    log.info("moe_16_mode from checkpoint config: %s", moe_16_mode)
    moe_type = model_cfg.get("moe_type", "sparse")
    top_k = model_cfg.get("top_k", 2)  # Default to 2 if missing (matches model default)
    log.info("top_k from checkpoint config: %s", top_k)
    gate_mode = model_cfg.get("gate_mode", "renormalized")
    log.info("gate_mode from checkpoint config: %s", gate_mode)
    router_noise_enabled = model_cfg.get("router_noise_enabled", True)
    router_noise_scale = model_cfg.get("router_noise_scale", 1.0)
    router_noise_min_std = model_cfg.get("router_noise_min_std", 0.05)

    model = SpatialMoESODNet(
        use_deep_supervision=use_deep_supervision,
        num_experts=num_experts,
        k=top_k,
        gate_mode=gate_mode,
        window_size=window_size,
        moe_16_mode=moe_16_mode,
        moe_type=moe_type,
        router_noise_enabled=router_noise_enabled,
        router_noise_scale=router_noise_scale,
        router_noise_min_std=router_noise_min_std,
    ).to(device)

    model_hash = verify_parameter_consistency(model)
    model.load_state_dict(checkpoint["model_state_dict"])

    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    eval_dir = os.path.join(args.out_dir, ckpt_name)

    log.info("--- EVALUATION CONFIGURATION ---")
    log.info("Checkpoint: %s", args.checkpoint)
    log.info("Epoch: %s", checkpoint.get("epoch", "Unknown"))
    log.info("Best Metric: %s", checkpoint.get("best_metric", "Unknown"))
    log.info("Model Hash: %s", model_hash)
    log.info("TTA: %s", args.tta)
    log.info("--------------------------------")

    # Disable router noise and random augmentation implicitly by eval()
    model.eval()

    _, _, test_sys_loader, test_real_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=1,  # Strict batch size 1 for geometry extraction ease
        num_workers=2,
        distributed=False,
    )

    datasets_to_run = []
    if args.dataset in ["test_sys", "both"]:
        datasets_to_run.append(("test_sys", test_sys_loader))
    if args.dataset in ["test_real", "both"]:
        datasets_to_run.append(("test_real", test_real_loader))

    for ds_name, loader in datasets_to_run:
        log.info("Evaluating %s...", ds_name)
        ds_out_dir = os.path.join(eval_dir, ds_name, args.tta)

        results = evaluate(model, loader, ds_out_dir, use_tta=(args.tta == "hflip"))

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save manifest
        manifest = {
            "checkpoint": args.checkpoint,
            "dataset": ds_name,
            "tta": args.tta,
            "num_images": results["global"]["sample_count"],
            "prediction_resolution": "original",
            "boundary_kernel": BOUNDARY_KERNEL_SIZE,
            "boundary_shape": "ellipse",
            "model_hash": model_hash,
        }

        manifest_filename = f"evaluation_manifest_{ds_name}_{timestamp}.json"
        with open(os.path.join(ds_out_dir, manifest_filename), "w") as f:
            json.dump(manifest, f, indent=4, cls=NumpyEncoder)

        metrics_filename = f"metrics_{ds_name}_{timestamp}.json"
        with open(os.path.join(ds_out_dir, metrics_filename), "w") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

        # Summary TXT
        summary_filename = f"summary_{ds_name}_{timestamp}.txt"
        with open(os.path.join(ds_out_dir, summary_filename), "w") as f:
            f.write(f"--- {ds_name.upper()} RESULTS ---\n")
            f.write(f"Global Sample Count: {results['global']['sample_count']}\n")
            for k, v in results["global"].items():
                if isinstance(v, float):
                    f.write(f"{k}: {v:.4f}\n")

            f.write("\n--- WEATHER-WISE RESULTS ---\n")
            for w, w_res in results["weather"].items():
                f.write(f"\nWeather: {w} (Count: {w_res['sample_count']})\n")
                if "warning" in w_res:
                    f.write(f"WARNING: {w_res['warning']}\n")
                for k, v in w_res.items():
                    if isinstance(v, float):
                        f.write(f"  {k}: {v:.4f}\n")

        log.info("Completed %s. Results saved to %s", ds_name, ds_out_dir)


if __name__ == "__main__":
    main()
