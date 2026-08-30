import torch
import torch.nn as nn
import pytest
from src.loss import SpatialMoELoss
from src.moe_layer import MoEOutput

def get_dummy_moe_output(B, N, K, E, requires_grad=True):
    # Dummy router outputs
    clean_logits = torch.randn(B, E, 4, 4, requires_grad=requires_grad)
    
    # Simulating top-k output
    topk_indices = torch.randint(0, E, (B, N, K))
    topk_gates = torch.softmax(torch.randn(B, N, K), dim=-1)
    if requires_grad:
        topk_gates.requires_grad = True
        
    return MoEOutput(
        features=torch.randn(B, 1, 4, 4), # Dummy features
        routing_probs=torch.randn(B, E, 4, 4),
        topk_indices=topk_indices,
        topk_gates=topk_gates,
        entropy=torch.randn(B, 1, 4, 4),
        clean_logits=clean_logits
    )

def test_saliency_gradients():
    loss_fn = SpatialMoELoss()
    B, C, H, W = 2, 1, 16, 16
    
    saliency_logits = torch.randn(B, C, H, W, requires_grad=True)
    edge_logits = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randint(0, 2, (B, C, H, W)).float()
    gt_boundary = torch.randint(0, 2, (B, C, H, W)).float()
    
    moe_outputs = [] # Empty for saliency-only test
    
    losses = loss_fn(saliency_logits, edge_logits, moe_outputs, target, gt_boundary)
    
    # Test that primary loss components propagate to saliency_logits
    for key in ['L_bce', 'L_iou', 'L_ssim', 'L_boundary']:
        if saliency_logits.grad is not None:
            saliency_logits.grad.zero_()
        
        losses[key].backward(retain_graph=True)
        assert saliency_logits.grad is not None
        assert torch.isfinite(saliency_logits.grad).all()
        assert saliency_logits.grad.abs().sum() > 0

def test_router_gradients():
    loss_fn = SpatialMoELoss(lambda_z=1.0)
    B, N, K, E = 2, 16, 2, 8
    
    saliency_logits = torch.randn(B, 1, 4, 4)
    edge_logits = torch.randn(B, 1, 4, 4)
    target = torch.ones(B, 1, 4, 4)
    
    out = get_dummy_moe_output(B, N, K, E, requires_grad=True)
    moe_outputs = [out]
    
    losses = loss_fn(saliency_logits, edge_logits, moe_outputs, target)
    
    # Test LB and Importance loss flow to topk_gates
    for key in ['L_lb', 'L_importance']:
        if out.topk_gates.grad is not None:
            out.topk_gates.grad.zero_()
            
        losses[key].backward(retain_graph=True)
        assert out.topk_gates.grad is not None
        assert torch.isfinite(out.topk_gates.grad).all()
        assert out.topk_gates.grad.abs().sum() > 0
        
    # Test Z-loss flows to clean_logits
    if out.clean_logits.grad is not None:
        out.clean_logits.grad.zero_()
        
    losses['L_z'].backward(retain_graph=True)
    assert out.clean_logits.grad is not None
    assert torch.isfinite(out.clean_logits.grad).all()
    assert out.clean_logits.grad.abs().sum() > 0

def test_special_cases():
    loss_fn = SpatialMoELoss()
    B, C, H, W = 2, 1, 16, 16
    
    cases = {
        "zero_target": (torch.randn(B, C, H, W), torch.zeros(B, C, H, W)),
        "all_one_target": (torch.randn(B, C, H, W), torch.ones(B, C, H, W)),
        "perfect_pred": (torch.full((B, C, H, W), 10.0), torch.ones(B, C, H, W)), # Logits=10 ~ Prob=1
        "all_zero_pred": (torch.full((B, C, H, W), -10.0), torch.ones(B, C, H, W)),
        "random_pred": (torch.randn(B, C, H, W), torch.randint(0, 2, (B, C, H, W)).float()),
        "constant_pred": (torch.zeros(B, C, H, W), torch.randint(0, 2, (B, C, H, W)).float())
    }
    
    for name, (pred, target) in cases.items():
        losses = loss_fn(pred, pred, [], target, gt_boundary=target)
        for k, v in losses.items():
            assert torch.isfinite(v).all(), f"Loss {k} was not finite in case {name}: {v}"

def test_iou_independent_per_image():
    loss_fn = SpatialMoELoss()
    B, C, H, W = 2, 1, 8, 8
    
    # Image 0: Perfect prediction (large foreground)
    pred_0 = torch.full((1, C, H, W), 10.0)
    tgt_0 = torch.ones(1, C, H, W)
    
    # Image 1: Terrible prediction (all false positives on empty target)
    pred_1 = torch.full((1, C, H, W), 10.0)
    tgt_1 = torch.zeros(1, C, H, W)
    
    # Combine
    pred = torch.cat([pred_0, pred_1], dim=0)
    tgt = torch.cat([tgt_0, tgt_1], dim=0)
    
    # Compute soft IoU
    P = torch.sigmoid(pred)
    l_iou = loss_fn._soft_iou_loss(P, tgt)
    
    # Image 0 IoU should be ~1 (Loss = 0)
    # Image 1 IoU should be ~0 (Loss = 1)
    # Total mean loss should be ~0.5
    assert torch.isclose(l_iou, torch.tensor(0.5), atol=1e-3)

def test_load_balancing_correctness():
    loss_fn = SpatialMoELoss()
    
    B, N, K, E = 1, 4, 2, 4
    
    # Manually constructed topk_indices
    # Token 0 -> [0, 1]
    # Token 1 -> [0, 2]
    # Token 2 -> [2, 3]
    # Token 3 -> [1, 3]
    
    topk_indices = torch.tensor([
        [[0, 1],
         [0, 2],
         [2, 3],
         [1, 3]]
    ])
    
    # Manually constructed gates (let's say all 0.5)
    topk_gates = torch.full((1, 4, 2), 0.5)
    
    out = MoEOutput(
        features=torch.zeros(1),
        routing_probs=None,
        topk_indices=topk_indices,
        topk_gates=topk_gates,
        entropy=None,
        clean_logits=torch.zeros(B, E, 1, 1)
    )
    
    l_lb, l_imp, _ = loss_fn._moe_losses([out])
    
    # f_j counts: each expert appears exactly twice
    # Total assignments = 4 * 2 = 8
    # f_j = [2/8, 2/8, 2/8, 2/8] = [0.25, 0.25, 0.25, 0.25]
    
    # P_j (soft mass mean over tokens): 
    # Expert 0 receives 0.5 twice, sum = 1.0. Mean over 4 tokens = 0.25
    # All experts receive P_j = 0.25
    
    # L_lb = E * sum(f_j * P_j) = 4 * sum(0.25 * 0.25 for i in range(4))
    # L_lb = 4 * 4 * 0.0625 = 1.0
    
    assert torch.isclose(l_lb, torch.tensor(1.0), atol=1e-5)

def test_boundary_gradients_and_correctness():
    loss_fn = SpatialMoELoss()
    
    B, C, H, W = 1, 1, 8, 8
    
    # Prediction 1: perfect match (logits = 10 for 1s, -10 for 0s)
    pred_1 = torch.full((B, C, H, W), -10.0)
    pred_1[:, :, 2:6, 2:6] = 10.0
    P_1 = torch.sigmoid(pred_1)
    
    # Generate GT boundary perfectly matching the finite difference of P_1
    gx = torch.zeros_like(P_1)
    gy = torch.zeros_like(P_1)
    gx[:, :, :, :-1] = P_1[:, :, :, 1:] - P_1[:, :, :, :-1]
    gy[:, :, :-1, :] = P_1[:, :, 1:, :] - P_1[:, :, :-1, :]
    gt_boundary = torch.sqrt(gx**2 + gy**2 + 1e-6)
    
    # Prediction 2: displaced prediction
    pred_2 = torch.full((B, C, H, W), -10.0)
    pred_2[:, :, 3:7, 3:7] = 10.0
    P_2 = torch.sigmoid(pred_2)
    
    loss_1 = loss_fn._boundary_loss(P_1, gt_boundary)
    loss_2 = loss_fn._boundary_loss(P_2, gt_boundary)
    
    assert loss_1 < loss_2, "Displaced prediction should have higher boundary loss"

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])

def test_deep_supervision_gradients():
    loss_fn = SpatialMoELoss(lambda_deep_supervision=0.4)
    B, C, H, W = 2, 1, 96, 96
    
    saliency_logits = torch.randn(B, C, H, W, requires_grad=True)
    edge_logits = torch.randn(B, C, H, W, requires_grad=True)
    
    aux_logits_16 = torch.randn(B, C, 24, 24, requires_grad=True)
    aux_logits_8 = torch.randn(B, C, 48, 48, requires_grad=True)
    aux_logits_4 = torch.randn(B, C, 96, 96, requires_grad=True)
    
    target = torch.randint(0, 2, (B, C, H, W)).float()
    
    losses = loss_fn(
        saliency_logits=saliency_logits, 
        edge_logits=edge_logits, 
        moe_outputs=[], 
        target=target,
        gt_boundary=None,
        aux_logits_16=aux_logits_16,
        aux_logits_8=aux_logits_8,
        aux_logits_4=aux_logits_4
    )
    
    # Check that deep supervision loss is computed and > 0
    assert 'L_deep_supervision' in losses
    assert losses['L_deep_supervision'] > 0
    
    # Backward pass on total loss to check gradients
    losses['L_total'].backward()
    
    assert aux_logits_16.grad is not None
    assert aux_logits_8.grad is not None
    assert aux_logits_4.grad is not None
    
    assert torch.isfinite(aux_logits_16.grad).all()
    assert torch.isfinite(aux_logits_8.grad).all()
    assert torch.isfinite(aux_logits_4.grad).all()
    
    assert aux_logits_16.grad.abs().sum() > 0
    assert aux_logits_8.grad.abs().sum() > 0
    assert aux_logits_4.grad.abs().sum() > 0
