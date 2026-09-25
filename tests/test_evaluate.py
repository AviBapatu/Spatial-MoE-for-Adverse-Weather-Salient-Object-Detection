"""End-to-end tests for ``src.evaluate`` using a stub model + tiny synthetic data.

The stub answers ``[B, 1, H, W]`` zero maps and ignores ``ablation_cfg``, so the
test exercises the full evaluate pipeline (batch collation, reverse geometry,
PNG saving, aggregate + weather-wise metrics) without touching real checkpoints.
"""

import os
from types import SimpleNamespace
from typing import Any, Dict

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.evaluate import evaluate, predict_probs, reverse_geometry, verify_parameter_consistency

WEATHER_STEMS = ["1000_fog", "1001_rain", "1002_snow", "1003_fog"]
IMG_SIZE = 8
METRIC_KEYS = [
    "MAE",
    "S_measure",
    "E_adaptive",
    "E_mean",
    "E_max",
    "F_adaptive",
    "F_mean",
    "F_max",
    "boundary_MAE",
    "boundary_F1",
]


class _StubSODModel(nn.Module):
    """Minimal SOD model whose output is all-zero saliency at IMG_SIZE."""

    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self, images: torch.Tensor, ablation_cfg: Dict[str, Any] | None = None
    ) -> tuple:
        B = images.shape[0]
        logits = torch.zeros(B, 1, IMG_SIZE, IMG_SIZE, device=images.device)
        out = SimpleNamespace(saliency_logits=logits)
        return out, []


class _SyntheticDataset(Dataset):
    """Synthetic dataset backed by real GT PNG files on disk (tmp_path)."""

    def __init__(self, root: str, stems: [str]) -> None:
        self.items = []
        for stem in stems:
            gt_path = os.path.join(root, f"{stem}.png")
            gt = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            gt[2:6, 2:6] = 255
            cv2.imwrite(gt_path, gt)
            self.items.append((stem, gt_path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem, gt_path = self.items[idx]
        return {
            "image": torch.rand(3, IMG_SIZE, IMG_SIZE),
            "name": stem,
            "meta": {
                "orig_h": IMG_SIZE,
                "orig_w": IMG_SIZE,
                "resized_h": IMG_SIZE,
                "resized_w": IMG_SIZE,
                "pad_top": 0,
                "pad_bottom": 0,
                "pad_left": 0,
                "pad_right": 0,
                "gt_path": gt_path,
            },
        }


@pytest.fixture()
def synthetic_loader(tmp_path) -> DataLoader:
    ds = _SyntheticDataset(str(tmp_path), WEATHER_STEMS)
    return DataLoader(ds, batch_size=2, shuffle=False)


def test_evaluate_end_to_end(synthetic_loader, tmp_path):
    model = _StubSODModel()
    out_dir = str(tmp_path / "out")

    results = evaluate(model, synthetic_loader, out_dir, use_tta=False)

    assert set(results.keys()) == {"global", "weather"}
    assert results["global"]["sample_count"] == len(WEATHER_STEMS)
    for key in METRIC_KEYS + ["sample_count"]:
        assert key in results["global"], f"missing global key {key}"
        assert np.isfinite(results["global"][key]), f"non-finite {key}"

    # Per-weather aggregation (fog appears twice, rain/snow once).
    assert set(results["weather"].keys()) == {"fog", "rain", "snow"}
    assert results["weather"]["fog"]["sample_count"] == 2
    assert results["weather"]["rain"]["sample_count"] == 1
    assert results["weather"]["snow"]["sample_count"] == 1
    for w in results["weather"]:
        assert results["weather"][w]["warning"] == "LOW SAMPLE"  # all < 10 samples

    # Every stem got a saved prediction map.
    for stem in WEATHER_STEMS:
        assert os.path.exists(os.path.join(out_dir, f"{stem}.png")), f"missing {stem}.png"


def test_evaluate_tta_and_ablation_cfg(synthetic_loader, tmp_path):
    model = _StubSODModel()

    results_tta = evaluate(model, synthetic_loader, str(tmp_path / "out_tta"), use_tta=True)
    assert results_tta["global"]["sample_count"] == len(WEATHER_STEMS)

    cfg = {"scale": 4, "expert_id": 3}
    results_abl = evaluate(model, synthetic_loader, str(tmp_path / "out_abl"), ablation_cfg=cfg)
    assert results_abl["global"]["sample_count"] == len(WEATHER_STEMS)


def test_reverse_geometry_identity():
    pred = torch.full((1, 1, IMG_SIZE, IMG_SIZE), 0.5)
    meta = {
        "pad_top": 0,
        "pad_left": 0,
        "resized_h": IMG_SIZE,
        "resized_w": IMG_SIZE,
        "orig_h": IMG_SIZE,
        "orig_w": IMG_SIZE,
    }
    out = reverse_geometry(pred, meta)
    assert out.shape == (IMG_SIZE, IMG_SIZE)
    assert np.allclose(out, 0.5)


def test_predict_probs_tta_averaging():
    model = _StubSODModel()
    images = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)

    base = predict_probs(model, images, use_tta=False)
    tta = predict_probs(model, images, use_tta=True)

    assert base.shape == (2, 1, IMG_SIZE, IMG_SIZE)
    assert tta.shape == base.shape
    # Stub is flip-invariant (all zeros), so TTA must equal no-TTA.
    assert torch.allclose(tta, base)


def _StubSODModelWithExtraParam():
    """Model build that would correspond to a different config (drift proxy)."""

    class _Drifted(_StubSODModel):
        def __init__(self) -> None:
            super().__init__()
            self.extra = nn.Parameter(torch.zeros(1))

    return _Drifted()


def test_verify_parameter_consistency_detects_drift():
    h1 = verify_parameter_consistency(_StubSODModel())
    h2 = verify_parameter_consistency(_StubSODModel())
    assert h1 == h2  # deterministic across identical builds

    h3 = verify_parameter_consistency(_StubSODModelWithExtraParam())
    assert h3 != h1, "different parameter names must yield a different hash"


def test_legacy_router_noise_keys_are_restored() -> None:
    """Checkpoints from before the RouterNoise refactor must load strictly.

    The learned router noise moved from ``moe_4.noise_linear`` to
    ``moe_4.router_noise.noise_linear``. Renaming restores the same weights under the
    current layout; keys already in the current layout pass through untouched.
    """
    from src.evaluate import _restore_legacy_parameter_names

    legacy = {
        "moe_4.noise_linear.weight": 1,
        "moe_4.noise_linear.bias": 2,
        "moe_8.noise_linear.weight": 3,
        "moe_16.noise_linear.bias": 4,
        "backbone.x": 5,
        "moe_4.router.weight": 6,
    }
    out = _restore_legacy_parameter_names(legacy)

    assert out["moe_4.router_noise.noise_linear.weight"] == 1
    assert out["moe_4.router_noise.noise_linear.bias"] == 2
    assert out["moe_8.router_noise.noise_linear.weight"] == 3
    assert out["moe_16.router_noise.noise_linear.bias"] == 4
    assert out["backbone.x"] == 5
    assert out["moe_4.router.weight"] == 6
    assert len(out) == len(legacy)


def test_current_layout_keys_are_not_renamed() -> None:
    """The remap is a no-op for checkpoints written by the current code."""
    from src.evaluate import _restore_legacy_parameter_names

    current = {
        "moe_4.router_noise.noise_linear.weight": 1,
        "moe_8.router_noise.noise_linear.bias": 2,
    }
    assert _restore_legacy_parameter_names(current) == current

