import torch
import torch.nn as nn
import hashlib
import os
import pytest

from src.diagnostics import (
    MoEDiagnosticsEngine,
    RoutingTracker,
    WeatherAnalyzer,
    collapse_warnings,
    merge_states,
    weather_divergence,
)
from src.model import SpatialMoESODNet
from src.moe_layer import SpatialMoELayer, MoEOutput

pytestmark = pytest.mark.gpu

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
        
def test_merge_states_accumulates_across_ranks():
    num_experts, top_k = 6, 2

    # Rank-local shards with disjoint expert usage. Every token in rank r picks
    # exactly the pair {2*r, 2*r+1}, so the per-rank histogram is fully
    # deterministic: 16 tokens (2 x 16) x 1 pick each per expert.
    shard_trackers = []
    for rank in range(3):
        tracker = RoutingTracker(num_experts=num_experts, top_k=top_k)
        pair = torch.tensor([2 * rank, 2 * rank + 1])
        indices = pair.expand(2, 16, top_k)  # token t -> (2r, 2r+1)
        gates = torch.ones(2, 16, top_k)
        moe_out = MoEOutput(
            features=torch.randn(2, 256, 4, 4),
            routing_probs=torch.randn(2, num_experts, 4, 4),
            topk_indices=indices,
            topk_gates=gates,
            entropy=torch.zeros(2, 1, 4, 4),
            clean_logits=torch.randn(2, num_experts, 4, 4),
        )
        tracker.update(moe_out)
        shard_trackers.append(tracker)

    states = [t.state() for t in shard_trackers]
    merged = merge_states(states, num_experts=num_experts, top_k=top_k)

    # Combined histogram equals sum of the per-rank histograms.
    expected_counts = [0] * num_experts
    expected_tokens = 0
    for t in shard_trackers:
        for e in range(num_experts):
            expected_counts[e] += int(t.hard_counts[e])
        expected_tokens += t.total_tokens

    assert merged["hard_counts"] == expected_counts
    assert merged["total_tokens"] == expected_tokens
    # Each rank used its own expert pair: 32 flattened tokens, one pick per
    # expert per token -> 32 assignments per used expert.
    for e in range(num_experts):
        assert merged["hard_counts"][e] == 32
    # Gate mass equals number of assignments when all gates are 1.0
    # (2 picks per token): total_tokens * top_k.
    assert sum(merged["soft_mass"]) == pytest.approx(expected_tokens * top_k)
    assert sum(merged["hard_fractions"]) == pytest.approx(1.0)


def test_collapse_warnings():
    num_experts = 6

    # Healthy routing: no warnings.
    healthy = {
        "hard_fractions": [0.2, 0.15, 0.2, 0.15, 0.15, 0.15],
        "mean_normalized_entropy": 0.5,
    }
    assert collapse_warnings(healthy, num_experts) == []

    # A dead expert (<1% usage) is flagged.
    dead = {
        "hard_fractions": [0.0, 0.2, 0.2, 0.2, 0.2, 0.2],
        "mean_normalized_entropy": 0.5,
    }
    warnings = collapse_warnings(dead, num_experts)
    assert any("DEAD_EXPERT" in w for w in warnings)

    # Router collapse: a single expert takes >80% and route is deterministic.
    collapsed = {
        "hard_fractions": [0.9, 0.05, 0.01, 0.01, 0.02, 0.01],
        "mean_normalized_entropy": 0.1,
    }
    warnings = collapse_warnings(collapsed, num_experts)
    assert any("ROUTER COLLAPSE" in w for w in warnings)

    # Uniform usage + high entropy -> low specialization.
    uniform_high_entropy = {
        "hard_fractions": [1.0 / num_experts] * num_experts,
        "mean_normalized_entropy": 0.9,
    }
    warnings = collapse_warnings(uniform_high_entropy, num_experts)
    assert any("LOW SPECIALIZATION" in w for w in warnings)


def test_weather_divergence_images_and_rank_merge():
    analyzer = WeatherAnalyzer(num_experts=4)

    # 25 "fog" images all routed to expert 0/1; 25 "rain" all to expert 2/3.
    for i in range(25):
        fog_idx = torch.tensor([[0, 1]])
        analyzer.update(fog_idx, "fog")
    for i in range(25):
        rain_idx = torch.tensor([[2, 3]])
        analyzer.update(rain_idx, "rain")

    div = weather_divergence(analyzer.image_weather_stats, analyzer.num_experts, min_expert_samples=10)
    assert div["status"] == "success"
    assert set(div["expert_enrichment"][0]["p_weather_given_expert"].keys()) == {"fog", "rain"}

    # Experts 0/1 should strongly prefer fog, experts 2/3 strongly prefer rain.
    p_e0 = div["expert_enrichment"][0]["p_weather_given_expert"]
    p_e2 = div["expert_enrichment"][2]["p_weather_given_expert"]
    assert p_e0["fog"] > p_e0["rain"]
    assert p_e2["rain"] > p_e2["fog"]
    assert div["expert_enrichment"][0]["kl_div"] > 0.0

    # Below min_image_samples -> short-circuit status.
    small = WeatherAnalyzer(num_experts=4)
    small.update(torch.tensor([[0, 1]]), "fog")
    assert weather_divergence(small.image_weather_stats, small.num_experts, min_image_samples=20)["status"] == "insufficient_images"


if __name__ == '__main__':
    print("Running Diagnostics tests...")
    test_routing_tracker_fractions()
    test_zero_usage_expert()
    test_observational_regression()
    test_expert_ablation_bypass()
    test_merge_states_accumulates_across_ranks()
    test_collapse_warnings()
    test_weather_divergence_images_and_rank_merge()
    print("All tests passed!")
