import torch
import torch.nn as nn
import time
import pytest
from src.moe_layer import SpatialMoELayer, TokenWiseMLPExpert

def test_routing_invariants():
    """Verify routing constraints like exactly K distinct experts per valid token."""
    B, C, H, W = 2, 64, 16, 16
    E = 8
    K = 2
    
    model = SpatialMoELayer(dim=C, num_experts=E, k=K)
    x = torch.randn(B, C, H, W)
    out = model(x)
    
    # 1. Shape checks
    assert out.topk_indices.shape == (B, H*W, K)
    assert out.topk_gates.shape == (B, H*W, K)
    
    # 2. Distinct K experts per token check
    # topk_indices must have exactly K distinct values per token
    unique_counts = torch.tensor([len(torch.unique(idx)) for idx in out.topk_indices.view(-1, K)])
    assert (unique_counts == K).all(), "Some tokens did not get exactly K distinct experts!"
    
    # 3. Gates sum to 1
    gate_sums = out.topk_gates.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)

def test_dense_sparse_equivalence():
    """Compute dense reference output for exactly the selected experts and compare with sparse output."""
    B, C, H, W = 1, 64, 8, 8
    E = 4
    K = 2
    
    model = SpatialMoELayer(dim=C, num_experts=E, k=K, router_noise_enabled=False)
    model.eval() # Disable dropout/noise
    
    x = torch.randn(B, C, H, W)
    
    # Run sparse implementation
    out_sparse = model(x)
    Y_sparse = out_sparse.features
    
    # Recreate the exact same result using a dense implementation that loops over the selected Top-K
    x_tokens = x.flatten(2).transpose(1, 2).reshape(B * H * W, C)
    Y_dense = torch.zeros_like(x_tokens)
    
    flat_indices = out_sparse.topk_indices.view(-1, K)
    flat_gates = out_sparse.topk_gates.view(-1, K)
    
    for i in range(B * H * W):
        token = x_tokens[i:i+1] # [1, C]
        for k in range(K):
            expert_idx = flat_indices[i, k].item()
            gate_val = flat_gates[i, k].item()
            
            expert = model.experts[expert_idx]
            Y_dense[i:i+1] += gate_val * expert(token)
            
    Y_dense = Y_dense.view(B, H * W, C).transpose(1, 2).view(B, C, H, W)
    
    max_err = torch.max(torch.abs(Y_dense - Y_sparse)).item()
    assert max_err < 1e-5, f"Sparse/Dense mismatch! Max absolute error: {max_err}"

def test_runtime_sparsity():
    """Instrument the forward pass to verify token-expert evaluation counts."""
    B, C, H, W = 2, 64, 16, 16
    E = 8
    K = 2
    
    model = SpatialMoELayer(dim=C, num_experts=E, k=K)
    x = torch.randn(B, C, H, W)
    
    # Monkey-patch experts to count tokens
    token_counts = [0] * E
    
    def get_wrapper(idx, orig_forward):
        def wrapper(tokens):
            token_counts[idx] += tokens.size(0)
            return orig_forward(tokens)
        return wrapper
        
    for i in range(E):
        model.experts[i].forward = get_wrapper(i, model.experts[i].forward)
        
    out = model(x)
    
    total_tokens = B * H * W
    expected_evaluations = K * total_tokens
    actual_evaluations = sum(token_counts)
    
    assert actual_evaluations == expected_evaluations, f"Expected {expected_evaluations} evaluations, got {actual_evaluations}"
    print(f"\nSparsity counts:")
    for i, count in enumerate(token_counts):
        print(f"  Expert {i}: {count} tokens")
    print(f"  Total evaluations: {actual_evaluations} (== {K} * {total_tokens})")

def test_gradient_correctness():
    """Verify that gradients correctly propagate to active experts and NOT inactive experts."""
    B, C, H, W = 1, 64, 8, 8
    E = 8
    K = 2
    
    model = SpatialMoELayer(dim=C, num_experts=E, k=K, router_noise_enabled=False)
    x = torch.randn(B, C, H, W, requires_grad=True)
    
    # Force a specific routing by manipulating the router weights 
    # to ensure some experts are completely inactive
    with torch.no_grad():
        model.router_mlp[-1].weight.zero_()
        model.router_mlp[-1].bias.zero_()
        model.router_mlp[-1].bias[0] = 10.0
        model.router_mlp[-1].bias[1] = 9.0
    
    out = model(x)
    
    # Determine active experts
    active_experts = set(out.topk_indices.view(-1).unique().tolist())
    inactive_experts = set(range(E)) - active_experts
    
    # Check that we actually have inactive experts in this test run
    assert len(inactive_experts) > 0, "Test needs at least one inactive expert"
    
    loss = out.features.sum()
    loss.backward()
    
    # Check Router Gradients
    assert model.router_mlp[-1].weight.grad is not None
    assert torch.isfinite(model.router_mlp[-1].weight.grad).all()
    
    # Check Expert Gradients
    for i in range(E):
        grad = model.experts[i].fc1.weight.grad
        if i in active_experts:
            assert grad is not None
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0
        else:
            # In PyTorch, unused parameters might have None grad or zero grad depending on DDP/setup.
            # Without DDP, simple unused parameters in standard backward will be None if never used in graph computation.
            assert grad is None or grad.abs().sum() == 0

def benchmark_dense_vs_sparse():
    B, C, H, W = 4, 128, 32, 32
    E = 8
    K = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_sparse = SpatialMoELayer(dim=C, num_experts=E, k=K).to(device)
    x = torch.randn(B, C, H, W).to(device)
    
    # Warmup
    for _ in range(5):
        out = model_sparse(x)
        out.features.sum().backward()
        
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(20):
        out = model_sparse(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    fwd_time = (time.time() - t0) / 20
    
    print(f"\nSparse Forward time: {fwd_time*1000:.2f} ms")

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
