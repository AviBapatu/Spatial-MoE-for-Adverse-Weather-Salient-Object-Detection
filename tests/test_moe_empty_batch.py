import torch
import torch.nn as nn
import pytest
from src.moe_layer import SpatialMoELayer, TokenWiseMLPExpert

def test_expert_empty_batch():
    dim = 256
    expert = TokenWiseMLPExpert(dim=dim)
    
    # Empty tensor, 0 tokens
    x = torch.empty(0, dim, requires_grad=True)
    
    # Forward pass
    out = expert(x)
    
    assert out.shape == (0, dim), f"Expected shape (0, {dim}), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    
    # Try backward
    loss = out.sum()
    loss.backward()
    
    for name, p in expert.named_parameters():
        assert p.grad is not None, f"Parameter {name} has None grad after backward on empty batch"

def test_moe_layer_zero_tokens_force():
    dim = 256
    num_experts = 8
    layer = SpatialMoELayer(dim=dim, num_experts=num_experts, k=2, router_noise_enabled=False)
    layer.train()
    
    # Small spatial dims to force at least one expert to get 0 tokens
    B, C, H, W = 1, dim, 2, 2 # only 4 tokens total
    x = torch.randn(B, C, H, W, requires_grad=True)
    
    out = layer(x)
    
    loss = out.features.sum()
    loss.backward()
    
    # Verify no NaN/Inf
    assert not torch.isnan(out.features).any()
    assert not torch.isinf(out.features).any()
    
    # Ensure all experts got gradients even if they processed 0 tokens
    for i, expert in enumerate(layer.experts):
        for name, p in expert.named_parameters():
            assert p.grad is not None, f"Expert {i} parameter {name} has None grad"

if __name__ == "__main__":
    test_expert_empty_batch()
    test_moe_layer_zero_tokens_force()
    print("PASS: test_expert_empty_batch and test_moe_layer_zero_tokens_force")
