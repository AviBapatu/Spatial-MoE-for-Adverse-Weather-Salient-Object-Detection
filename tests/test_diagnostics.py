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


# ---------------------------------------------------------------------------
# pad_mask tests
# ---------------------------------------------------------------------------

def _make_moe_output(B: int, N_tokens: int, K: int, num_experts: int) -> "MoEOutput":
    """Helper: build a minimal MoEOutput for testing."""
    indices = torch.randint(0, num_experts, (B, N_tokens, K))
    gates = torch.softmax(torch.randn(B, N_tokens, K), dim=-1)
    entropy = torch.rand(B, 1, int(N_tokens ** 0.5), int(N_tokens ** 0.5))
    return MoEOutput(
        features=torch.randn(B, 256, int(N_tokens ** 0.5), int(N_tokens ** 0.5)),
        routing_probs=torch.randn(B, num_experts, int(N_tokens ** 0.5), int(N_tokens ** 0.5)),
        topk_indices=indices,
        topk_gates=gates,
        entropy=entropy,
        clean_logits=torch.randn(B, num_experts, int(N_tokens ** 0.5), int(N_tokens ** 0.5)),
    )


def test_routing_tracker_pad_mask():
    """Masking half the tokens should halve hard_counts sum (±1) and set masked_tokens."""
    B, N_tokens, K, E = 2, 64, 2, 6
    moe_out = _make_moe_output(B, N_tokens, K, E)

    # Baseline: no mask
    tracker_base = RoutingTracker(num_experts=E, top_k=K)
    tracker_base.update(moe_out)

    # Masked: first half of tokens valid, second half masked
    total_flat = B * N_tokens
    pad_mask = torch.zeros(total_flat, dtype=torch.bool)
    pad_mask[: total_flat // 2] = True  # 50 % valid

    tracker_masked = RoutingTracker(num_experts=E, top_k=K)
    tracker_masked.update(moe_out, pad_mask=pad_mask)

    # hard_counts should be about half — allow ±5 for random distribution
    base_total = sum(tracker_base.hard_counts.tolist())
    masked_total = sum(tracker_masked.hard_counts.tolist())
    assert abs(masked_total - base_total // 2) <= 5, (
        f"Expected ~{base_total // 2} hard count total with half the tokens, got {masked_total}"
    )

    # masked_tokens should equal the number of tokens set to False
    expected_masked = total_flat - total_flat // 2
    assert tracker_masked.masked_tokens == expected_masked, (
        f"masked_tokens={tracker_masked.masked_tokens}, expected {expected_masked}"
    )

    # total_tokens should equal valid count
    assert tracker_masked.total_tokens == total_flat // 2, (
        f"total_tokens={tracker_masked.total_tokens}, expected {total_flat // 2}"
    )


def test_routing_tracker_all_masked():
    """All tokens masked → total_tokens == 0, masked_tokens == N, no division errors."""
    B, N_tokens, K, E = 2, 64, 2, 6
    moe_out = _make_moe_output(B, N_tokens, K, E)

    pad_mask = torch.zeros(B * N_tokens, dtype=torch.bool)  # all masked

    tracker = RoutingTracker(num_experts=E, top_k=K)
    tracker.update(moe_out, pad_mask=pad_mask)

    assert tracker.total_tokens == 0, f"total_tokens should be 0, got {tracker.total_tokens}"
    assert tracker.masked_tokens == B * N_tokens

    # get_stats must not raise (no division by zero)
    stats = tracker.get_stats()
    assert stats["total_tokens"] == 0
    assert all(f == 0.0 for f in stats["hard_fractions"])
    assert all(f == 0.0 for f in stats["soft_fractions"])
    assert stats["mean_entropy"] == 0.0

    # masked_token_fraction from finalize path
    observed = tracker.total_tokens + tracker.masked_tokens
    frac = tracker.masked_tokens / observed if observed > 0 else 0.0
    assert abs(frac - 1.0) < 1e-6


def test_routing_tracker_pad_mask_state_roundtrip():
    """masked_tokens survives state() / from_state() round-trip and merge()."""
    B, N_tokens, K, E = 2, 64, 2, 6
    moe_out = _make_moe_output(B, N_tokens, K, E)
    pad_mask = torch.zeros(B * N_tokens, dtype=torch.bool)
    pad_mask[:30] = True

    tracker = RoutingTracker(num_experts=E, top_k=K)
    tracker.update(moe_out, pad_mask=pad_mask)
    assert tracker.masked_tokens > 0

    # Round-trip through state / from_state
    restored = RoutingTracker.from_state(tracker.state())
    assert restored.masked_tokens == tracker.masked_tokens

    # Merge two identical trackers → masked_tokens doubles
    merged = RoutingTracker(num_experts=E, top_k=K)
    merged.merge(tracker).merge(tracker)
    assert merged.masked_tokens == tracker.masked_tokens * 2


def test_weather_analyzer_pad_mask():
    """WeatherAnalyzer with pad_mask should only count valid-token assignments."""
    E, K = 6, 2
    N_tokens = 36  # 6×6 spatial grid

    # All tokens assigned to expert 0
    indices = torch.zeros(N_tokens, K, dtype=torch.long)  # [N_tokens, K]

    # Mask the first half: only the second half is valid
    pad_mask = torch.zeros(N_tokens, dtype=torch.bool)
    pad_mask[N_tokens // 2 :] = True  # latter half valid

    analyzer = WeatherAnalyzer(num_experts=E)
    analyzer.update(indices, "fog", pad_mask=pad_mask)

    stat = analyzer.image_weather_stats[0]
    # Only the valid half contributed → expert 0 count == (N_tokens // 2) * K
    expected_count = (N_tokens // 2) * K
    assert stat["counts"][0] == expected_count, (
        f"Expected expert 0 count={expected_count}, got {stat['counts'][0]}"
    )
    assert stat["total"] == expected_count, (
        f"Expected total={expected_count}, got {stat['total']}"
    )


def test_expert_similarity_pad_mask(tmp_path):
    """Content-only activation similarity should differ from all-token similarity
    when half the input tokens are near-constant (padding-like)."""
    from src.diagnostics import ExpertSimilarityAnalyzer

    E, C, H, W = 4, 64, 8, 8
    N_tokens = H * W
    B = 1

    # Build a minimal MoE layer with 4 experts
    class _DummyExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(C, C * 4)
            self.fc2 = nn.Linear(C * 4, C)
            self.norm = nn.LayerNorm(C)

        def forward(self, x):
            z = self.norm(x)
            z = torch.relu(self.fc1(z))
            z = self.fc2(z)
            return x + z

    class _DummyMoELayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([_DummyExpert() for _ in range(E)])

    moe_layer = _DummyMoELayer().eval()
    analyzer = ExpertSimilarityAnalyzer(moe_layer, num_experts=E)

    # x: first half of spatial positions are near-constant (padding-like near-zero)
    x = torch.randn(B, C, H, W)
    x[:, :, : H // 2, :] = 1e-4  # simulate padding — almost zero features

    # pad_mask: only the second half is valid (flat over B*H*W)
    pad_mask = torch.zeros(B * N_tokens, dtype=torch.bool)
    pad_mask[B * N_tokens // 2 :] = True

    baseline_path = str(tmp_path / "sim_base.json")
    content_path = str(tmp_path / "sim_content.json")

    res_base = analyzer.analyze(x, baseline_path, pad_mask=None)
    res_cont = analyzer.analyze(x, content_path, pad_mask=pad_mask)

    base_vals = [
        res_base["activation_similarity"][i][j]
        for i in range(E) for j in range(i + 1, E)
    ]
    cont_vals = [
        res_cont["activation_similarity"][i][j]
        for i in range(E) for j in range(i + 1, E)
    ]

    mean_base = sum(base_vals) / len(base_vals)
    mean_cont = sum(cont_vals) / len(cont_vals)

    # With random uninitialised weights the direction of change is not
    # predictable in general: near-zero (padding-like) inputs drive the output
    # through the z-dominated path (small x, large z), which can look *more*
    # diverse than rich content tokens where the residual term x dominates.
    # What we can assert deterministically is that masking the padding region
    # produces a *different* mean similarity from the unmasked baseline — i.e.
    # the filter is actually active and changes the token set.
    assert abs(mean_base - mean_cont) > 0.05, (
        f"Expected masked ({mean_cont:.4f}) and unmasked ({mean_base:.4f}) similarity "
        f"to differ by >0.05 when half the tokens are near-constant, "
        f"but they were within the tolerance (delta={abs(mean_base - mean_cont):.4f})."
    )


if __name__ == '__main__':
    print("Running Diagnostics tests...")
    test_routing_tracker_fractions()
    test_zero_usage_expert()
    test_observational_regression()
    test_expert_ablation_bypass()
    test_merge_states_accumulates_across_ranks()
    test_collapse_warnings()
    test_weather_divergence_images_and_rank_merge()
    test_routing_tracker_pad_mask()
    test_routing_tracker_all_masked()
    test_routing_tracker_pad_mask_state_roundtrip()
    test_weather_analyzer_pad_mask()
    print("All tests passed!")

