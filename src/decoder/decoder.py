"""Top-level Spatial-MoE decoder: multi-scale fusion + prediction heads.

Assembles the reusable building blocks from :mod:`src.decoder.blocks` into
the full saliency decoder, including optional deep-supervision aux heads.
"""
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.decoder.blocks import (
    EntropyFusionBlock,
    GlobalCrossAttentionBlock,
    RefinementBlock,
    WindowedCrossAttentionBlock,
)


class DecoderOutput(NamedTuple):
    """Container bundling the decoder's prediction maps.

    Attributes:
        saliency_logits: Full-resolution saliency map ``[B, 1, 384, 384]``.
        boundary_logits: Full-resolution boundary map ``[B, 1, 384, 384]``.
        aux_logits_16: Deep-supervision logits at 1/16 scale (or ``None``).
        aux_logits_8: Deep-supervision logits at 1/8 scale (or ``None``).
        aux_logits_4: Deep-supervision logits at 1/4 scale (or ``None``).
    """

    saliency_logits: torch.Tensor
    boundary_logits: torch.Tensor
    aux_logits_16: Optional[torch.Tensor]
    aux_logits_8: Optional[torch.Tensor]
    aux_logits_4: Optional[torch.Tensor]


class SpatialMoEDecoder(nn.Module):
    """Multi-scale cross-attention decoder for saliency prediction.

    Pipeline (top-down):

    1. Per-scale entropy fusion of MoE features with routing entropy.
    2. Global cross-attention fusing the 1/16-scale context stream into the
       1/8-scale features, injected into the 1/4-scale stream via a global
       context pool.
    3. Windowed cross-attention fusing the 1/8-scale stream into the 1/4-scale
       features.
    4. Residual refinement and bilinear upsampling back to input resolution.
    5. Saliency / boundary prediction heads, plus optional aux heads at each
       scale for deep supervision.
    """

    def __init__(
        self, dim: int = 256, window_size: int = 8, use_deep_supervision: bool = True
    ) -> None:
        """Initialize the decoder.

        Args:
            dim: Feature dimensionality throughout the decoder.
            window_size: Window size for the 1/8 -> 1/4 windowed attention.
            use_deep_supervision: Whether to attach per-scale aux heads.
        """
        super().__init__()
        self.dim = dim
        self.use_deep_supervision = use_deep_supervision

        self.fuse_16 = EntropyFusionBlock(dim, dim)
        self.fuse_8 = EntropyFusionBlock(dim, dim)
        self.fuse_4 = EntropyFusionBlock(dim, dim)

        self.global_context_pool = nn.AdaptiveAvgPool2d(1)
        self.global_context_proj = nn.Conv2d(dim, dim, 1)

        self.cross_attn_16_to_8 = GlobalCrossAttentionBlock(dim)
        self.cross_attn_8_to_4 = WindowedCrossAttentionBlock(dim, window_size=window_size)

        self.refine_4 = RefinementBlock(dim)
        self.refine_2 = RefinementBlock(dim)
        self.refine_1 = RefinementBlock(dim)

        self.saliency_head = nn.Conv2d(dim, 1, 1)
        self.edge_head = nn.Conv2d(dim, 1, 1)

        if self.use_deep_supervision:
            self.aux_head_16 = nn.Conv2d(dim, 1, 1)
            self.aux_head_8 = nn.Conv2d(dim, 1, 1)
            self.aux_head_4 = nn.Conv2d(dim, 1, 1)

    def forward(
        self,
        Y_4: torch.Tensor,
        Y_8: torch.Tensor,
        Y_16: torch.Tensor,
        H_4: torch.Tensor,
        H_8: torch.Tensor,
        H_16: torch.Tensor,
        disable_entropy: bool = False,
    ) -> DecoderOutput:
        """Fuse the three MoE scales and produce prediction maps.

        Args:
            Y_4: MoE features at 1/4 scale ``[B, dim, H_4, W_4]``.
            Y_8: MoE features at 1/8 scale ``[B, dim, H_8, W_8]``.
            Y_16: MoE features at 1/16 scale ``[B, dim, H_16, W_16]``.
            H_4: Routing entropy at 1/4 scale ``[B, 1, H_4, W_4]``.
            H_8: Routing entropy at 1/8 scale ``[B, 1, H_8, W_8]``.
            H_16: Routing entropy at 1/16 scale ``[B, 1, H_16, W_16]``.
            disable_entropy: If ``True`` (ablation), skip entropy channels.

        Returns:
            A :class:`DecoderOutput` with full-resolution saliency/boundary
            maps and (optionally) per-scale aux maps.
        """
        F16 = self.fuse_16(Y_16, H_16, disable_entropy=disable_entropy)
        F8_local = self.fuse_8(Y_8, H_8, disable_entropy=disable_entropy)
        F4_local = self.fuse_4(Y_4, H_4, disable_entropy=disable_entropy)

        F16_up = F.interpolate(F16, size=F8_local.shape[-2:], mode="bilinear", align_corners=False)
        F8 = self.cross_attn_16_to_8(q_x=F16_up, kv_x=F8_local)

        g_ctx = self.global_context_proj(self.global_context_pool(F8))
        F4_ctx = F4_local + g_ctx

        F8_up = F.interpolate(F8, size=F4_ctx.shape[-2:], mode="bilinear", align_corners=False)
        F4 = self.cross_attn_8_to_4(q_x=F8_up, kv_x=F4_ctx)

        x4 = self.refine_4(F4)
        x2 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x2 = self.refine_2(x2)
        x1 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
        x1 = self.refine_1(x1)

        saliency_logits = self.saliency_head(x1)
        boundary_logits = self.edge_head(x1)

        aux_16 = self.aux_head_16(F16) if self.use_deep_supervision else None
        aux_8 = self.aux_head_8(F8) if self.use_deep_supervision else None
        aux_4 = self.aux_head_4(F4) if self.use_deep_supervision else None

        return DecoderOutput(
            saliency_logits=saliency_logits,
            boundary_logits=boundary_logits,
            aux_logits_16=aux_16,
            aux_logits_8=aux_8,
            aux_logits_4=aux_4,
        )
