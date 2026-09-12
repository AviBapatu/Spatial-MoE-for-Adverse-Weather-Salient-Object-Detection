"""Spatial-MoE decoder package.

Re-exports the decoder building blocks (:mod:`src.decoder.blocks`) and the
top-level decoder (:mod:`src.decoder.decoder`) so that the pre-split import
paths (e.g. ``from src.decoder import SpatialMoEDecoder``) keep working.
"""
from src.decoder.blocks import (
    EntropyFusionBlock,
    GlobalCrossAttentionBlock,
    RefinementBlock,
    WindowedCrossAttention,
    WindowedCrossAttentionBlock,
    window_partition,
    window_reverse,
)
from src.decoder.decoder import DecoderOutput, SpatialMoEDecoder

__all__ = [
    "SpatialMoEDecoder",
    "DecoderOutput",
    "EntropyFusionBlock",
    "GlobalCrossAttentionBlock",
    "WindowedCrossAttention",
    "WindowedCrossAttentionBlock",
    "RefinementBlock",
    "window_partition",
    "window_reverse",
]
