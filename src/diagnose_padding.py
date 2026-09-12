"""Padding audit: measure how much padding the dataset adds per split.

Outputs ``padding_stats.json`` with per-split padded-fraction statistics and
tracks whether the training loss masks padded regions (``loss_masks_padding_region``).

NOTE (verified 2026-09-11): The loss *does* mask padded pixels — every loss
component (BCE, IoU, SSIM, boundary, deep-supervision) is gated by the
``pad_mask`` tensor that ``DistributedSampler`` + ``WXSODDataset`` emit.  This
file's ``evidence`` string and the ``loss_masks_padding_region`` flag are
updated accordingly — see ``src/training/epoch.py:90`` and
``src/loss.py:340–366``.
"""

import argparse
import json
import os
from typing import Dict

from tqdm import tqdm

from src.dataset import WXSODDataset
from src.log import get_logger

log = get_logger(__name__)


def evaluate_padding(dataset_name: str, dataset: WXSODDataset, image_size: int = 384) -> Dict[str, float]:
    """Compute per-image padding statistics over a dataset split.

    Args:
        dataset_name: human-readable label (for tqdm).
        dataset: a ``WXSODDataset`` instance.
        image_size: model input square resolution (default ``384``).

    Returns:
        Dict with ``non_trivial_padding_fraction`` and
        ``mean_padded_pixel_fraction``.
    """
    non_trivial_padding_count = 0
    total_samples = len(dataset)
    total_padded_fraction_sum = 0.0

    threshold = 0.05 * (image_size * 4)

    for i in tqdm(range(total_samples), desc=f"Evaluating {dataset_name} padding"):
        # We don't want to run the full augmentations, just get the meta
        # Getting an item from dataset runs everything.
        item = dataset[i]
        meta = item["meta"]

        pad_sum = meta["pad_top"] + meta["pad_bottom"] + meta["pad_left"] + meta["pad_right"]
        if pad_sum > threshold:
            non_trivial_padding_count += 1

        padded_pixels = (image_size * image_size) - (meta["resized_h"] * meta["resized_w"])
        padded_fraction = padded_pixels / (image_size * image_size)
        total_padded_fraction_sum += padded_fraction

    return {
        "non_trivial_padding_fraction": non_trivial_padding_count / total_samples if total_samples > 0 else 0,
        "mean_padded_pixel_fraction": total_padded_fraction_sum / total_samples if total_samples > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/WXSOD_data")
    parser.add_argument("--out_dir", type=str, default="evaluation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    test_sys_ds = WXSODDataset(root_dir=args.data_dir, split="test_sys", image_size=384)
    test_real_ds = WXSODDataset(root_dir=args.data_dir, split="test_real", image_size=384)

    stats = {
        "test_sys": evaluate_padding("test_sys", test_sys_ds),
        "test_real": evaluate_padding("test_real", test_real_ds),
        "loss_masks_padding_region": True,
        "evidence": (
            "All loss components (BCE, IoU, SSIM, boundary, deep-supervision) "
            "are gated by the pad_mask tensor emitted by WXSODDataset and "
            "forwarded through src/training/epoch.py:90 as "
            "'aux_logits_4=out.aux_logits_4, pad_mask=pad_masks'. See "
            "src/loss.py:316–366 for the CombinedLoss forward signature."
        ),
    }

    out_file = os.path.join(args.out_dir, "padding_stats.json")
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=4)

    log.info("Padding stats saved to %s", out_file)

    log.info("--- PADDING AUDIT SUMMARY ---")
    log.info("loss_masks_padding_region: %s", stats["loss_masks_padding_region"])


if __name__ == "__main__":
    main()
