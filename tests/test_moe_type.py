"""``moe_type`` must actually change the architecture.

It used to be a dead config field: "sparse", "dense" and "none" all produced the
identical 62.9M-parameter MoE model, so the "no MoE" ablation row was a duplicate
of the sparse one.  These tests pin the three arms apart.
"""
from __future__ import annotations

import pytest
import torch

from src.config import LossConfig
from src.loss import CombinedLoss
from src.model import SpatialMoESODNet

ARMS = ("sparse", "dense", "none")
E, K, DIM = 4, 2, 64


def _model(moe_type: str) -> SpatialMoESODNet:
    return SpatialMoESODNet(
        dim=DIM, num_experts=E, k=K, gate_mode="dense", moe_16_mode="sparse",
        moe_type=moe_type, use_deep_supervision=True, pretrained_backbone=False,
    )


def _batch():
    return (
        torch.randn(2, 3, 64, 64),
        (torch.rand(2, 1, 64, 64) > 0.5).float(),
        (torch.rand(2, 1, 64, 64) > 0.8).float(),
    )


def _criterion(**overrides) -> CombinedLoss:
    cfg = dict(bce_weight=1.0, iou_weight=1.0, ssim_weight=0.0, boundary_weight=1.0,
               load_balance_weight=0.01, importance_weight=0.1, z_loss_weight=0.0,
               aux_boundary_weight=0.5, deep_supervision_weight=0.4)
    cfg.update(overrides)
    return CombinedLoss(LossConfig(**cfg))


@pytest.mark.parametrize("moe_type", ARMS)
def test_each_arm_builds_a_different_architecture(moe_type: str) -> None:
    counts = {t: sum(p.numel() for p in _model(t).parameters()) for t in ARMS}
    # 4 experts/scale (sparse) > 1 shared expert/scale (dense) > no experts (none)
    assert counts["sparse"] > counts["dense"] > counts["none"], counts


@pytest.mark.parametrize("moe_type", ARMS)
def test_router_less_arms_are_flagged(moe_type: str) -> None:
    model = _model(moe_type)
    expected_dense = moe_type != "sparse"
    assert bool(getattr(model.moe_4, "is_dense", False)) is expected_dense
    assert bool(getattr(model.moe_16, "is_dense", False)) is expected_dense


@pytest.mark.parametrize("moe_type", ARMS)
def test_forward_backward_is_finite_for_every_arm(moe_type: str) -> None:
    model = _model(moe_type)
    images, target, edge = _batch()
    out, moe_outputs = model(images)
    loss, components = _criterion()(
        saliency_logits=out.saliency_logits, edge_logits=out.boundary_logits,
        moe_outputs=moe_outputs, target=target, gt_boundary=edge,
        aux_logits_16=out.aux_logits_16, aux_logits_8=out.aux_logits_8,
        aux_logits_4=out.aux_logits_4,
    )
    loss.backward()
    assert torch.isfinite(loss), f"{moe_type}: loss not finite"
    for name, value in components.items():
        assert torch.isfinite(torch.as_tensor(value)), f"{moe_type}: {name} not finite"


def test_outputs_advertise_whether_they_have_a_router() -> None:
    images, _, _ = _batch()
    with torch.no_grad():
        _, sparse_outs = _model("sparse")(images)
        _, dense_outs = _model("dense")(images)
        _, none_outs = _model("none")(images)
    assert all(o.has_router for o in sparse_outs)
    assert not any(o.has_router for o in dense_outs)
    assert not any(o.has_router for o in none_outs)


@pytest.mark.parametrize("moe_type", ("dense", "none"))
def test_routing_losses_are_skipped_without_a_router(moe_type: str) -> None:
    """Sentinel routing fields must not leak into the balance/importance terms."""
    model = _model(moe_type)
    images, target, edge = _batch()
    with torch.no_grad():
        out, moe_outputs = model(images)
        _, components = _criterion()(
            saliency_logits=out.saliency_logits, edge_logits=out.boundary_logits,
            moe_outputs=moe_outputs, target=target, gt_boundary=edge,
            aux_logits_16=out.aux_logits_16, aux_logits_8=out.aux_logits_8,
            aux_logits_4=out.aux_logits_4,
        )
    assert float(components["L_lb"]) == 0.0
    assert float(components["L_importance"]) == 0.0


def test_entropy_confidence_term_is_the_normalised_routing_entropy() -> None:
    model = _model("sparse")
    images, target, edge = _batch()
    with torch.no_grad():
        out, moe_outputs = model(images)
        _, comp_off = _criterion(entropy_confidence_weight=0.0)(
            saliency_logits=out.saliency_logits, edge_logits=out.boundary_logits,
            moe_outputs=moe_outputs, target=target, gt_boundary=edge,
            aux_logits_16=out.aux_logits_16, aux_logits_8=out.aux_logits_8,
            aux_logits_4=out.aux_logits_4,
        )
        _, comp_on = _criterion(entropy_confidence_weight=1.0)(
            saliency_logits=out.saliency_logits, edge_logits=out.boundary_logits,
            moe_outputs=moe_outputs, target=target, gt_boundary=edge,
            aux_logits_16=out.aux_logits_16, aux_logits_8=out.aux_logits_8,
            aux_logits_4=out.aux_logits_4,
        )
    import math
    expected = sum(
        float((o.entropy / math.log(o.clean_logits.shape[1])).mean()) for o in moe_outputs
    ) / len(moe_outputs)
    assert float(comp_off["L_routing_conf"]) == pytest.approx(expected, rel=1e-4)
    # normalized entropy lives in [0, 1]
    assert 0.0 <= float(comp_on["L_routing_conf"]) <= 1.0
    # and the weight actually moves the total
    assert float(comp_on["L_total"]) > float(comp_off["L_total"]) - 1e-9
    assert float(comp_on["L_total"]) >= float(comp_on["L_routing_conf"]) - 1e-6
