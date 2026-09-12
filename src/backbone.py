"""Hierarchical multi-scale backbone feature extraction.

Wraps a PVTv2 backbone (via ``timm``) and projects each extracted scale to a
common working dimensionality.
"""
from typing import Dict

import timm
import torch
import torch.nn as nn

from src.log import get_logger

log = get_logger(__name__)


class MultiScaleBackbone(nn.Module):
    """PVTv2-based hierarchical features at 1/4, 1/8, and 1/16 scale.

    The backbone extracts ``out_indices=(0, 1, 2)``, which correspond to the
    1/4, 1/8, and 1/16 scales of PVTv2, and 1x1-convolves each scale down to a
    common width ``d``.
    """

    def __init__(self, model_name: str = "pvt_v2_b4", pretrained: bool = True, d: int = 256) -> None:
        """Initialize the backbone and per-scale projection layers.

        Args:
            model_name: ``timm`` PVTv2 model name (e.g. ``"pvt_v2_b4"``).
            pretrained: Whether to load pre-trained weights.
            d: Common channel dimensionality for all three output scales.
        """
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2),
        )

        feature_channels = self.backbone.feature_info.channels()

        self.proj_4 = nn.Conv2d(feature_channels[0], d, kernel_size=1)
        self.proj_8 = nn.Conv2d(feature_channels[1], d, kernel_size=1)
        self.proj_16 = nn.Conv2d(feature_channels[2], d, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract and project the three backbone scales.

        Args:
            x: Input image ``[B, 3, H, W]``.

        Returns:
            A dict with keys ``res_4``, ``res_8``, ``res_16`` mapping to the
            extracted features at 1/4, 1/8, and 1/16 scale, each projected to
            dimensionality ``d``.
        """
        feats = self.backbone(x)

        res_4 = self.proj_4(feats[0])
        res_8 = self.proj_8(feats[1])
        res_16 = self.proj_16(feats[2])

        return {
            "res_4": res_4,
            "res_8": res_8,
            "res_16": res_16,
        }


if __name__ == "__main__":
    log.info("Testing MultiScaleBackbone...")

    model = MultiScaleBackbone(model_name="pvt_v2_b4", pretrained=True, d=256)

    dummy_input = torch.randn(1, 3, 384, 384)
    log.info(f"Input shape: {dummy_input.shape}")

    outputs = model(dummy_input)

    log.info("\nOutput shapes:")
    log.info(f"  res_4: {outputs['res_4'].shape}")
    log.info(f"  res_8: {outputs['res_8'].shape}")
    log.info(f"  res_16: {outputs['res_16'].shape}")

    assert outputs["res_4"].shape == (1, 256, 96, 96), (
        f"Expected (1, 256, 96, 96), got {outputs['res_4'].shape}"
    )
    assert outputs["res_8"].shape == (1, 256, 48, 48), (
        f"Expected (1, 256, 48, 48), got {outputs['res_8'].shape}"
    )
    assert outputs["res_16"].shape == (1, 256, 24, 24), (
        f"Expected (1, 256, 24, 24), got {outputs['res_16'].shape}"
    )

    log.info("All shapes verified successfully!")
