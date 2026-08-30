import torch
import torch.nn as nn
import hashlib
import os

from src.diagnostics import MoEDiagnosticsEngine, RoutingTracker
from src.model import SpatialMoESODNet
from src.moe_layer import SpatialMoELayer, MoEOutput

def get_param_hash(model):
    names = sorted([name for name, _ in model.named_parameters()])
    values = [p.data.cpu().numpy().tobytes() for _, p in model.named_parameters()]
    h = hashlib.md5()
    for v in values:
        h.update(v)
    return h.hexdigest()

def test_routing_tracker_fractions():
    tracker = RoutingTracker(num_experts=6, top_k=2)
    B, N_tokens = 2, 64
    k = 2
    
    # Create dummy MoEOutput
    indices = torch.randint(0, 6, (B, N_tokens, k))
    gates = torch.softmax(torch.randn(B, N_tokens, k), dim=-1)
    
    moe_out = MoEOutput(
        features=torch.randn(B, 256, 8, 8),
        routing_probs=torch.randn(B, 6, 8, 8),
        topk_indices=indices,
        topk_gates=gates,
        entropy=torch.zeros(B, 1, 8, 8),
        clean_logits=torch.randn(B, 6, 8, 8)
    )
    
    tracker.update(moe_out)
    stats = tracker.get_stats()
    
    hard_fracs = stats['hard_fractions']
    soft_fracs = stats['soft_fractions']
    
    assert abs(sum(hard_fracs) - 1.0) < 1e-4, f"Hard fractions sum to {sum(hard_fracs)}"
    assert abs(sum(soft_fracs) - 1.0) < 1e-4, f"Soft fractions sum to {sum(soft_fracs)}"
    
def test_zero_usage_expert():
    tracker = RoutingTracker(num_experts=6, top_k=2)
    B, N_tokens = 2, 64
    k = 2
    
    # All tokens go to expert 0 and 1 only
    indices = torch.randint(0, 2, (B, N_tokens, k))
    gates = torch.softmax(torch.randn(B, N_tokens, k), dim=-1)
    
    moe_out = MoEOutput(
        features=torch.randn(B, 256, 8, 8),
        routing_probs=torch.randn(B, 6, 8, 8),
        topk_indices=indices,
        topk_gates=gates,
        entropy=torch.zeros(B, 1, 8, 8),
        clean_logits=torch.randn(B, 6, 8, 8)
    )
    
    tracker.update(moe_out)
    stats = tracker.get_stats()
    
    # Experts 2,3,4,5 should be exactly 0
    for i in range(2, 6):
        assert stats['hard_counts'][i] == 0
        assert stats['soft_mass'][i] == 0.0

def test_observational_regression():
    # 1. Put model in eval
    model = SpatialMoESODNet(dim=64, num_experts=6, k=2).eval()
    
    # 2. Save param checksum
    pre_hash = get_param_hash(model)
    
    # 3. Save a few model outputs
    dummy_input = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        pre_out, pre_moe = model(dummy_input)
        pre_sal = pre_out.saliency_logits.clone()
        
    # 4. Run diagnostics
    import random
    import numpy as np
    
    pre_rng_py = random.getstate()
    pre_rng_np = np.random.get_state()
    pre_rng_torch = torch.get_rng_state()
    
    with torch.no_grad():
        engine = MoEDiagnosticsEngine("/tmp/diag_test", num_experts=6, top_k=2)
        meta = {'name': ['test_clean', 'test_fog']}
        engine.update(dummy_input, pre_moe, meta, num_visual_samples=0)
        engine.finalize(0)
        
    random.setstate(pre_rng_py)
    np.random.set_state(pre_rng_np)
    torch.set_rng_state(pre_rng_torch)
    
    # 5. Recompute outputs
    post_hash = get_param_hash(model)
    with torch.no_grad():
        post_out, _ = model(dummy_input)
        post_sal = post_out.saliency_logits.clone()
        
    # 6. Compare
    assert pre_hash == post_hash, "Model parameters were modified by diagnostics!"
    assert torch.allclose(pre_sal, post_sal), "Model outputs changed after running diagnostics!"

def test_expert_ablation_bypass():
    model = SpatialMoESODNet(dim=64, num_experts=6, k=2).eval()
    dummy_input = torch.randn(1, 3, 128, 128)
    
    with torch.no_grad():
        ablation_cfg = {'scale': 4, 'expert_id': 3}
        out, moe_outputs = model(dummy_input, ablation_cfg=ablation_cfg)
        
        # Scale 4 is at index 0 (1/4 scale)
        ablation_moe = moe_outputs[0]
        
        assert (ablation_moe.topk_indices == 3).all(), "TopK indices not bypassed!"
        
        # Gate [:,:,0] should be 1.0, others 0.0
        assert (ablation_moe.topk_gates[:, :, 0] == 1.0).all(), "Gate 0 not 1.0!"
        assert (ablation_moe.topk_gates[:, :, 1] == 0.0).all(), "Gate 1 not 0.0!"
        
if __name__ == '__main__':
    print("Running Diagnostics tests...")
    test_routing_tracker_fractions()
    test_zero_usage_expert()
    test_observational_regression()
    test_expert_ablation_bypass()
    print("All tests passed!")
