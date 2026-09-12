"""Shape / invariant tests for the Spatial-MoE model modules.

Covers the MoE layer output shapes, per-token routing-normalization, the
extracted router-noise helper, and the DDP-safety regression that ensures
every expert receives a gradient even when it routed zero tokens.
"""
import torch
import pytest

from src.moe_layer import RouterNoise, SpatialMoELayer
from src.model import SpatialMoESODNet

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def test_moe_layer_output_feature_shape() -> None:
    """SpatialMoELayer output feature map must match the input shape."""
    B, C, H, W = 2, 64, 16, 16
    layer = SpatialMoELayer(dim=C, num_experts=8, k=2)
    x = torch.randn(B, C, H, W)

    out = layer(x)

    assert out.features.shape == (B, C, H, W)


def test_moe_layer_routing_probs_sum_to_one() -> None:
    """routing_probs must sum to ~1 per token over the expert axis."""
    B, C, H, W = 2, 64, 16, 16
    layer = SpatialMoELayer(dim=C, num_experts=8, k=2)
    x = torch.randn(B, C, H, W)

    out = layer(x)

    probs = out.routing_probs  # [B, E, H, W]
    assert probs.shape == (B, 8, H, W)
    per_token_sum = probs.sum(dim=1)  # [B, H, W]
    assert torch.allclose(per_token_sum, torch.ones_like(per_token_sum), atol=1e-5)


def test_moe_layer_topk_shapes() -> None:
    """topk_indices / topk_gates must be shaped [B, N_tokens, K]."""
    B, C, H, W, K = 2, 64, 16, 16, 2
    layer = SpatialMoELayer(dim=C, num_experts=8, k=K)
    x = torch.randn(B, C, H, W)

    out = layer(x)

    assert out.topk_indices.shape == (B, H * W, K)
    assert out.topk_gates.shape == (B, H * W, K)

    # Flattened per-token view matches the [N_tokens, K] contract.
    flat_indices = out.topk_indices.view(B * H * W, K)
    flat_gates = out.topk_gates.view(B * H * W, K)
    assert flat_indices.shape == (B * H * W, K)
    assert flat_gates.shape == (B * H * W, K)


def test_moe_layer_entropy_range_is_topk_bounded() -> None:
    """Routing entropy must cover the selected top-K gates, range [0, ln K].

    The decoder's EntropyFusionBlock divides the entropy map by ``log(2)``
    under the K=2 assumption (``src/decoder/blocks.py``); if the entropy
    instead spanned the full ``ln(E)`` of the 8-expert distribution, that
    normalization would be miscalibrated.
    """
    B, C, H, W, K = 2, 64, 16, 16, 2
    layer = SpatialMoELayer(dim=C, num_experts=8, k=K)
    x = torch.randn(B, C, H, W)

    out = layer(x)

    assert out.entropy.shape == (B, 1, H, W)
    assert out.entropy.min() >= 0.0
    assert out.entropy.max() <= torch.log(torch.tensor(K)) + 1e-6
    assert out.entropy.max() < torch.log(torch.tensor(layer.num_experts))


def test_router_noise_clamps_and_disables() -> None:
    """RouterNoise: clamps std, scales by noise_scale, no-ops in eval."""
    dim, num_experts = 32, 8
    noise = RouterNoise(dim=dim, num_experts=num_experts)
    clean_logits = torch.zeros(2, 4, num_experts)
    x_tokens = torch.randn(2, 4, dim)

    # Training mode: std clamped to the floor, logits perturbed.
    noisy, std = noise.add_noise(
        clean_logits, x_tokens, training=True, noise_scale=1.0, noise_min_std=0.05
    )
    assert noisy.shape == clean_logits.shape
    assert std is not None
    assert (std >= 0.05 - 1e-6).all()
    assert not torch.equal(noisy, clean_logits)

    # zero scale + zero floor -> deterministic no-op.
    noisy2, std2 = noise.add_noise(
        clean_logits, x_tokens, training=True, noise_scale=0.0, noise_min_std=0.0
    )
    assert (std2 == 0).all()
    assert torch.equal(noisy2, clean_logits)

    # Eval mode: logits untouched, std None.
    noisy3, std3 = noise.add_noise(clean_logits, x_tokens, training=False)
    assert std3 is None
    assert torch.equal(noisy3, clean_logits)


def test_ddp_safety_zero_token_experts_grads() -> None:
    """Every expert must get a non-None grad even with zero routed tokens.

    This is the invariant that keeps DDP gradient sync consistent across
    ranks with different routing patterns. Skew the router logits so at least
    one expert is never selected, then verify that expert's parameters still
    receive a (possibly zero) gradient after backward.
    """
    C, E = 32, 8
    model = SpatialMoELayer(dim=C, num_experts=E, k=2, router_noise_enabled=False)
    model.train()

    with torch.no_grad():
        model.router_mlp[-1].weight.zero_()
        model.router_mlp[-1].bias.zero_()
        model.router_mlp[-1].bias[0] = 5.0
        model.router_mlp[-1].bias[1] = 4.0
        model.router_mlp[-1].bias[E - 1] = -1000.0

    x = torch.randn(2, C, 8, 8)

    out = model(x)

    routed_experts = set(out.topk_indices.view(-1).unique().tolist())
    zero_token_experts = set(range(E)) - routed_experts
    assert len(zero_token_experts) > 0, "Test setup must leave an expert unrouted"

    loss = out.features.sum()
    loss.backward()

    for i, expert in enumerate(model.experts):
        for name, p in expert.named_parameters():
            assert p.grad is not None, (
                f"Expert {i} param {name} has None grad — "
                "this would desync DDP gradient sync across ranks"
            )


def test_full_model_smoke_shapes() -> None:
    """End-to-end SpatialMoESODNet output shapes on a small config."""
    model = SpatialMoESODNet(
        dim=64, num_experts=4, k=2, window_size=8, use_deep_supervision=True
    )
    model.eval()

    x = torch.randn(1, 3, 384, 384)
    decoder_out, moe_outputs = model(x)

    assert decoder_out.saliency_logits.shape == (1, 1, 384, 384)
    assert decoder_out.boundary_logits.shape == (1, 1, 384, 384)
    assert decoder_out.aux_logits_16.shape == (1, 1, 24, 24)
    assert decoder_out.aux_logits_8.shape == (1, 1, 48, 48)
    assert decoder_out.aux_logits_4.shape == (1, 1, 96, 96)

    # Per-scale MoE outputs at their native resolutions.
    assert moe_outputs[0].features.shape == (1, 64, 96, 96)
    assert moe_outputs[1].features.shape == (1, 64, 48, 48)
    assert moe_outputs[2].features.shape == (1, 64, 24, 24)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])