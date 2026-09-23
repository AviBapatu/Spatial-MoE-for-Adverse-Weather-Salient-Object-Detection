"""Top-level Spatial-MoE SOD network.

Wires the multi-scale backbone, per-scale MoE layers, and the fusion decoder
into a single end-to-end saliency-detection model.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.backbone import MultiScaleBackbone
from src.decoder import DecoderOutput, SpatialMoEDecoder
from src.log import get_logger
from src.moe_layer import DenseMoE16Adapter, MoEOutput, PassthroughMoELayer, SpatialMoELayer

log = get_logger(__name__)


class SpatialMoESODNet(nn.Module):
    """Spatially-dynamic Mixture-of-Experts saliency-detection network.

    Architecture:

    1. A hierarchical ``MultiScaleBackbone`` extracts features at 1/4, 1/8,
       and 1/16 scale.
    2. An independent ``SpatialMoELayer`` routes and dispatches each scale's
       spatial tokens to specialized experts.
    3. The ``SpatialMoEDecoder`` fuses the routed features (with routing
       entropy) top-down and predicts saliency/boundary maps.

    The ``__init__`` signature takes only primitives (not an
    ``ExperimentConfig``) so the model stays usable in inference-only scripts.
    """

    def __init__(
        self,
        dim: int = 256,
        num_experts: int = 6,
        k: int = 2,
        window_size: int = 8,
        use_deep_supervision: bool = True,
        router_noise_enabled: bool = True,
        router_noise_scale: float = 1.0,
        router_noise_min_std: float = 0.05,
        pretrained_backbone: bool = True,
        moe_16_mode: str = "sparse",
        gate_mode: str = "renormalized",
        moe_type: str = "sparse",
    ) -> None:
        """Initialize the full network.

        Args:
            dim: Feature dimensionality of the MoE layers and decoder.
            num_experts: Number of experts in each scale's MoE layer.
            k: Number of experts selected per token (top-``k``).
            window_size: Window size for the decoder's windowed attention.
            use_deep_supervision: Whether to attach per-scale aux heads.
            router_noise_enabled: Whether noisy routing is active in training.
            router_noise_scale: Multiplier for the learned noise std.
            router_noise_min_std: Lower clamp on the noise std.
            gate_mode: ``"renormalized"`` or ``"dense"`` — how the top-k gates are
                formed (see ``SpatialMoELayer``).
            moe_type: ``"sparse"`` (routed experts), ``"dense"`` (one shared expert
                on every token) or ``"none"`` (no MoE).
            pretrained_backbone: Whether the PVTv2 backbone loads pretrained
                ImageNet weights from the HF Hub. Leave ``True`` for real
                training runs; set ``False`` only for offline/CI smoke tests
                where network access to the Hub isn't available and weight
                values don't matter (construction/shape checks, etc.).
        """
        super().__init__()

        # 1. Hierarchical multi-scale backbone.
        self.backbone = MultiScaleBackbone(d=dim, pretrained=pretrained_backbone)

        # 2. Independent MoE layers for each scale.  `moe_type` selects the
        #    ablation arm:
        #      "sparse" — routed top-k experts: the model under study.
        #      "dense"  — one shared expert applied to every token: the same
        #                 expert machinery with routing AND sparsity removed, so
        #                 this is the capacity-matched control for "does learned
        #                 routing beat just applying an expert to everything?".
        #      "none"   — no MoE at all (features pass through); a lighter control
        #                 that also removes the expert parameters.
        def _moe_layer() -> nn.Module:
            if moe_type == "none":
                return PassthroughMoELayer(dim=dim)
            if moe_type == "dense":
                return DenseMoE16Adapter(dim=dim)
            return SpatialMoELayer(
                dim=dim,
                num_experts=num_experts,
                k=k,
                gate_mode=gate_mode,
                router_noise_enabled=router_noise_enabled,
                router_noise_scale=router_noise_scale,
                router_noise_min_std=router_noise_min_std,
            )

        self.moe_4 = _moe_layer()
        self.moe_8 = _moe_layer()
        # moe_16_mode="dense" keeps its own single-expert 16-scale arm.
        self.moe_16 = DenseMoE16Adapter(dim=dim) if moe_16_mode == "dense" else _moe_layer()

        # 3. Spatial MoE decoder for feature fusion and prediction.
        self.decoder = SpatialMoEDecoder(
            dim=dim,
            window_size=window_size,
            use_deep_supervision=use_deep_supervision,
        )

    def forward(
        self, x: torch.Tensor, ablation_cfg: Optional[Dict[str, Any]] = None
    ) -> Tuple[DecoderOutput, List[MoEOutput]]:
        """Run the full forward pass.

        Args:
            x: Input image ``[B, 3, H, W]``.
            ablation_cfg: Optional counterfactual-ablation settings:

                - ``"scale"``: 4, 8, or 16 — forces the MoE at that scale to
                  route every token to ``"expert_id"``.
                - ``"expert_id"``: Expert index to force when ``scale`` is set.
                - ``"random_routing"``: Replace learned routing with random
                  logits (ablation).
                - ``"disable_entropy"``: Skip the entropy channel in the
                  decoder (ablation).

        Returns:
            A tuple ``(decoder_out, moe_outputs)`` where ``decoder_out`` is a
            :class:`src.decoder.DecoderOutput` and ``moe_outputs`` is the list
            of per-scale :class:`src.moe_layer.MoEOutput`.
        """
        feats = self.backbone(x)
        res_4, res_8, res_16 = feats["res_4"], feats["res_8"], feats["res_16"]

        # Counterfactual ablation setup.
        force_4 = force_8 = force_16 = None
        random_routing = False
        disable_entropy = False
        if ablation_cfg is not None:
            if ablation_cfg.get("scale") == 4:
                force_4 = ablation_cfg.get("expert_id")
            if ablation_cfg.get("scale") == 8:
                force_8 = ablation_cfg.get("expert_id")
            if ablation_cfg.get("scale") == 16:
                force_16 = ablation_cfg.get("expert_id")
            random_routing = ablation_cfg.get("random_routing", False)
            disable_entropy = ablation_cfg.get("disable_entropy", False)

        # Route + dispatch each scale through its MoE layer.
        out_4 = self.moe_4(res_4, force_expert_id=force_4, random_routing=random_routing)
        out_8 = self.moe_8(res_8, force_expert_id=force_8, random_routing=random_routing)
        out_16 = self.moe_16(res_16, force_expert_id=force_16, random_routing=random_routing)

        # Fuse the routed features in the decoder.
        decoder_out = self.decoder(
            out_4.features,
            out_8.features,
            out_16.features,
            out_4.entropy,
            out_8.entropy,
            out_16.entropy,
            disable_entropy=disable_entropy,
        )

        return decoder_out, [out_4, out_8, out_16]


if __name__ == "__main__":
    log.info("Testing SpatialMoESODNet end-to-end...")
    model = SpatialMoESODNet(dim=256)

    dummy_input = torch.randn(1, 3, 384, 384)
    log.info(f"Input image shape: {dummy_input.shape}")

    decoder_out, moe_outputs = model(dummy_input)

    log.info("\nOutput shapes:")
    log.info(f"  Saliency Logits: {decoder_out.saliency_logits.shape}")
    log.info(f"  Edge Logits: {decoder_out.boundary_logits.shape}")
    log.info(f"  Aux 16: {decoder_out.aux_logits_16.shape}")
    log.info(f"  Aux 8: {decoder_out.aux_logits_8.shape}")
    log.info(f"  Aux 4: {decoder_out.aux_logits_4.shape}")
    for i, out in enumerate(moe_outputs):
        log.info(f"  Routing Prob Level {i+1}: {out.routing_probs.shape}")

    assert decoder_out.saliency_logits.shape == (1, 1, 384, 384)
    assert decoder_out.boundary_logits.shape == (1, 1, 384, 384)
    assert decoder_out.aux_logits_16.shape == (1, 1, 24, 24)
    assert decoder_out.aux_logits_8.shape == (1, 1, 48, 48)
    assert decoder_out.aux_logits_4.shape == (1, 1, 96, 96)
    log.info("All integration tests passed!")
