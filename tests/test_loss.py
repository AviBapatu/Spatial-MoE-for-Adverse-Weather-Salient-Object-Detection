import torch
import torch.nn as nn
import pytest
from src.config import LossConfig
from src.loss import (
    CombinedLoss,
    SpatialMoELoss,
    bce_loss,
    image_gradient_magnitude_loss,
    moe_routing_losses,
    soft_iou_loss,
)
from src.moe_layer import MoEOutput

SHARED_CONTRACT_KEYS = [
    "L_total",
    "L_bce",
    "L_iou",
    "L_ssim",
    "L_boundary",
    "L_lb",
    "L_importance",
    "L_z",
    "L_aux_boundary",
    "L_deep_supervision",
]


def get_dummy_moe_output(B, N, K, E, requires_grad=True):
    clean_logits = torch.randn(B, E, 4, 4, requires_grad=requires_grad)

    topk_indices = torch.randint(0, E, (B, N, K))
    topk_gates = torch.softmax(torch.randn(B, N, K), dim=-1)
    if requires_grad:
        topk_gates.requires_grad = True

    return MoEOutput(
        features=torch.randn(B, 1, 4, 4),
        routing_probs=torch.randn(B, E, 4, 4),
        topk_indices=topk_indices,
        topk_gates=topk_gates,
        entropy=torch.randn(B, 1, 4, 4),
        clean_logits=clean_logits,
    )


def combined_with(overrides=None):
    defaults = dict(
        bce_weight=1.0,
        iou_weight=1.0,
        ssim_weight=1.0,
        boundary_weight=0.5,
        load_balance_weight=0.01,
        importance_weight=0.01,
        z_loss_weight=0.0,
        aux_boundary_weight=0.0,
        deep_supervision_weight=0.4,
    )
    if overrides:
        defaults.update(overrides)
    return CombinedLoss(LossConfig(**defaults))


def test_contract_returns_tuple_and_all_keys():
    loss_fn = combined_with()
    B, C, H, W = 2, 1, 8, 8
    saliency_logits = torch.randn(B, C, H, W)
    edge_logits = torch.randn(B, C, H, W)
    target = torch.randint(0, 2, (B, C, H, W)).float()
    gt_boundary = torch.randint(0, 2, (B, C, H, W)).float()

    out = loss_fn(
        saliency_logits, edge_logits, [get_dummy_moe_output(2, 8, 2, 8)], target,
        gt_boundary=gt_boundary,
        aux_logits_16=torch.randn(B, C, 2, 2),
        aux_logits_8=torch.randn(B, C, 4, 4),
        aux_logits_4=torch.randn(B, C, 8, 8),
        pad_mask=None,
    )

    assert isinstance(out, tuple) and len(out) == 2
    total, components = out
    assert set(components.keys()) == set(SHARED_CONTRACT_KEYS)
    assert torch.isfinite(total).all()


def test_total_is_weighted_sum_of_components():
    cfg = LossConfig(
        bce_weight=1.0, iou_weight=1.5, ssim_weight=0.3, boundary_weight=0.7,
        load_balance_weight=0.02, importance_weight=0.03, z_loss_weight=0.4,
        aux_boundary_weight=0.5, deep_supervision_weight=0.6,
    )
    loss_fn = CombinedLoss(cfg)
    B, C, H, W = 2, 1, 8, 8
    saliency_logits = torch.randn(B, C, H, W)
    edge_logits = torch.randn(B, C, H, W)
    target = torch.randint(0, 2, (B, C, H, W)).float()
    gt_boundary = torch.randint(0, 2, (B, C, H, W)).float()

    _, comps = loss_fn(
        saliency_logits, edge_logits, [get_dummy_moe_output(2, 8, 2, 8)], target,
        gt_boundary=gt_boundary,
        aux_logits_16=torch.randn(B, C, 2, 2),
        aux_logits_8=torch.randn(B, C, 4, 4),
        aux_logits_4=torch.randn(B, C, 8, 8),
    )

    expected = (
        cfg.bce_weight * comps["L_bce"]
        + cfg.iou_weight * comps["L_iou"]
        + cfg.ssim_weight * comps["L_ssim"]
        + cfg.boundary_weight * comps["L_boundary"]
        + cfg.load_balance_weight * comps["L_lb"]
        + cfg.importance_weight * comps["L_importance"]
        + cfg.z_loss_weight * comps["L_z"]
        + cfg.aux_boundary_weight * comps["L_aux_boundary"]
        + cfg.deep_supervision_weight * comps["L_deep_supervision"]
    )
    assert torch.allclose(comps["L_total"], expected, atol=1e-6)


def test_saliency_gradients():
    loss_fn = combined_with()
    B, C, H, W = 2, 1, 16, 16

    saliency_logits = torch.randn(B, C, H, W, requires_grad=True)
    edge_logits = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randint(0, 2, (B, C, H, W)).float()
    gt_boundary = torch.randint(0, 2, (B, C, H, W)).float()

    _, components = loss_fn(
        saliency_logits=saliency_logits,
        edge_logits=edge_logits,
        moe_outputs=[],
        target=target,
        gt_boundary=gt_boundary,
    )

    for key in ["L_bce", "L_iou", "L_ssim", "L_boundary"]:
        if saliency_logits.grad is not None:
            saliency_logits.grad.zero_()

        components[key].backward(retain_graph=True)
        assert saliency_logits.grad is not None
        assert torch.isfinite(saliency_logits.grad).all()
        assert saliency_logits.grad.abs().sum() > 0


def test_router_gradients():
    loss_fn = combined_with(overrides=dict(z_loss_weight=1.0))
    B, N, K, E = 2, 16, 2, 8

    saliency_logits = torch.randn(B, 1, 4, 4)
    edge_logits = torch.randn(B, 1, 4, 4)
    target = torch.ones(B, 1, 4, 4)

    out = get_dummy_moe_output(B, N, K, E, requires_grad=True)

    _, components = loss_fn(
        saliency_logits=saliency_logits,
        edge_logits=edge_logits,
        moe_outputs=[out],
        target=target,
    )

    for key in ["L_lb", "L_importance"]:
        if out.topk_gates.grad is not None:
            out.topk_gates.grad.zero_()

        components[key].backward(retain_graph=True)
        assert out.topk_gates.grad is not None
        assert torch.isfinite(out.topk_gates.grad).all()
        assert out.topk_gates.grad.abs().sum() > 0

    if out.clean_logits.grad is not None:
        out.clean_logits.grad.zero_()

    components["L_z"].backward(retain_graph=True)
    assert out.clean_logits.grad is not None
    assert torch.isfinite(out.clean_logits.grad).all()
    assert out.clean_logits.grad.abs().sum() > 0


def test_special_cases():
    loss_fn = combined_with()
    B, C, H, W = 2, 1, 16, 16

    cases = {
        "zero_target": (torch.randn(B, C, H, W), torch.zeros(B, C, H, W)),
        "all_one_target": (torch.randn(B, C, H, W), torch.ones(B, C, H, W)),
        "perfect_pred": (torch.full((B, C, H, W), 10.0), torch.ones(B, C, H, W)),
        "all_zero_pred": (torch.full((B, C, H, W), -10.0), torch.ones(B, C, H, W)),
        "random_pred": (torch.randn(B, C, H, W), torch.randint(0, 2, (B, C, H, W)).float()),
        "constant_pred": (torch.zeros(B, C, H, W), torch.randint(0, 2, (B, C, H, W)).float()),
    }

    for name, (pred, target) in cases.items():
        _, components = loss_fn(
            saliency_logits=pred, edge_logits=pred, moe_outputs=[], target=target,
            gt_boundary=target,
        )
        for k, v in components.items():
            assert torch.isfinite(v).all(), f"Loss {k} was not finite in case {name}: {v}"


def test_soft_iou_independent_per_image():
    B, C, H, W = 2, 1, 8, 8

    pred_0 = torch.full((1, C, H, W), 10.0)
    tgt_0 = torch.ones(1, C, H, W)

    pred_1 = torch.full((1, C, H, W), 10.0)
    tgt_1 = torch.zeros(1, C, H, W)

    pred = torch.cat([pred_0, pred_1], dim=0)
    tgt = torch.cat([tgt_0, tgt_1], dim=0)

    P = torch.sigmoid(pred)
    l_iou = soft_iou_loss(P, tgt)

    assert torch.isclose(l_iou, torch.tensor(0.5), atol=1e-3)


def test_load_balancing_correctness():
    B, N, K, E = 1, 4, 2, 4

    topk_indices = torch.tensor([[[0, 1], [0, 2], [2, 3], [1, 3]]])
    topk_gates = torch.full((1, 4, 2), 0.5)

    out = MoEOutput(
        features=torch.zeros(1),
        routing_probs=None,
        topk_indices=topk_indices,
        topk_gates=topk_gates,
        entropy=None,
        clean_logits=torch.zeros(B, E, 1, 1),
    )

    l_lb, l_imp, _ = moe_routing_losses([out])

    assert torch.isclose(l_lb, torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(l_imp, torch.tensor(0.0), atol=1e-5)


def test_boundary_gradients_and_correctness():
    B, C, H, W = 1, 1, 8, 8

    pred_1 = torch.full((B, C, H, W), -10.0)
    pred_1[:, :, 2:6, 2:6] = 10.0
    P_1 = torch.sigmoid(pred_1)

    gx = torch.zeros_like(P_1)
    gy = torch.zeros_like(P_1)
    gx[:, :, :, :-1] = P_1[:, :, :, 1:] - P_1[:, :, :, :-1]
    gy[:, :, :-1, :] = P_1[:, :, 1:, :] - P_1[:, :, :-1, :]
    gt_boundary = torch.sqrt(gx**2 + gy**2 + 1e-6)

    pred_2 = torch.full((B, C, H, W), -10.0)
    pred_2[:, :, 3:7, 3:7] = 10.0
    P_2 = torch.sigmoid(pred_2)

    loss_1 = image_gradient_magnitude_loss(P_1, gt_boundary)
    loss_2 = image_gradient_magnitude_loss(P_2, gt_boundary)

    assert loss_1 < loss_2, "Displaced prediction should have higher boundary loss"


def test_boundary_masking_uses_pad_mask():
    B, C, H, W = 1, 1, 8, 8
    P = torch.zeros(B, C, H, W)
    gt_boundary = torch.ones(B, C, H, W)
    mask = torch.zeros(B, C, H, W)

    masked = image_gradient_magnitude_loss(P, gt_boundary, mask=mask)
    unmasked = image_gradient_magnitude_loss(P, gt_boundary)
    assert float(masked) == 0.0
    assert float(unmasked) > 0.0


def test_bce_masking():
    logits = torch.randn(2, 1, 8, 8)
    target = torch.randint(0, 2, (2, 1, 8, 8)).float()
    pad = torch.zeros(2, 1, 8, 8)
    assert float(bce_loss(logits, target, pad_mask=pad)) == 0.0
    assert float(bce_loss(logits, target)) > 0.0


def test_deep_supervision_gradients():
    loss_fn = combined_with()
    B, C, H, W = 2, 1, 96, 96

    saliency_logits = torch.randn(B, C, H, W, requires_grad=True)
    edge_logits = torch.randn(B, C, H, W, requires_grad=True)

    aux_logits_16 = torch.randn(B, C, 24, 24, requires_grad=True)
    aux_logits_8 = torch.randn(B, C, 48, 48, requires_grad=True)
    aux_logits_4 = torch.randn(B, C, 96, 96, requires_grad=True)

    target = torch.randint(0, 2, (B, C, H, W)).float()

    _, components = loss_fn(
        saliency_logits=saliency_logits,
        edge_logits=edge_logits,
        moe_outputs=[],
        target=target,
        gt_boundary=None,
        aux_logits_16=aux_logits_16,
        aux_logits_8=aux_logits_8,
        aux_logits_4=aux_logits_4,
    )

    assert components["L_deep_supervision"] > 0

    components["L_total"].backward()

    for aux in [aux_logits_16, aux_logits_8, aux_logits_4]:
        assert aux.grad is not None
        assert torch.isfinite(aux.grad).all()
        assert aux.grad.abs().sum() > 0


def test_legacy_spatial_moe_loss_returns_plain_dict():
    loss_fn = SpatialMoELoss()
    B, C, H, W = 2, 1, 8, 8
    saliency_logits = torch.randn(B, C, H, W)
    edge_logits = torch.randn(B, C, H, W)
    target = torch.randint(0, 2, (B, C, H, W)).float()

    losses = loss_fn(saliency_logits, edge_logits, [], target)

    assert isinstance(losses, dict)
    assert set(losses.keys()) == set(SHARED_CONTRACT_KEYS)


def test_legacy_wrapper_matches_combined():
    B, C, H, W = 2, 1, 8, 8
    saliency_logits = torch.randn(B, C, H, W)
    edge_logits = torch.randn(B, C, H, W)
    target = torch.randint(0, 2, (B, C, H, W)).float()

    legacy = SpatialMoELoss(lambda_ssim=1.0, lambda_boundary=0.5)
    combined = CombinedLoss(
        LossConfig(bce_weight=1.0, iou_weight=1.0, ssim_weight=1.0, boundary_weight=0.5)
    )

    legacy_res = legacy(saliency_logits, edge_logits, [], target)
    _, combined_res = combined(saliency_logits, edge_logits, [], target)

    for k in SHARED_CONTRACT_KEYS:
        assert torch.allclose(legacy_res[k], combined_res[k], atol=1e-6), k