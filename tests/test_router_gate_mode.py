"""The gate formulation decides whether expert selection is learnable at all.

``gate_mode="renormalized"`` (the historical default) forms each gate as a
softmax over the ``k`` chosen logits.  That softmax is shift-invariant within the
selected pair, so the task loss has *exactly* zero gradient for every expert a
token did not pick: the router gets no signal about which experts it should have
preferred, and selection can only drift under router noise and the auxiliary
uniformity terms.  ``gate_mode="dense"`` gates each chosen expert by its
probability under the full softmax over all experts, which keeps a gradient on
the unpicked experts and makes selection learnable.

These tests pin that property — if a refactor ever leaves renormalisation as the
only path, they fail.
"""
from __future__ import annotations

import torch

from src.moe_layer import SpatialMoELayer

E, K, D, H = 4, 2, 16, 8


def _unpicked_expert_grad(gate_mode: str) -> tuple[int, int]:
    """Return ``(zero entries, total entries)`` in the task gradient on the
    logits of the experts each token did not select.

    Only the routed features feed the loss, so every gradient measured here came
    from the task path — no auxiliary loss term is involved.
    """
    torch.manual_seed(0)
    layer = SpatialMoELayer(dim=D, num_experts=E, k=K, gate_mode=gate_mode).eval()
    captured = {}

    def grab(_module, _inputs, output):
        output.retain_grad()
        captured["logits"] = output

    # Final router Linear: its output is [B, N, E] pre-top-k.
    handle = layer.router_mlp[2].register_forward_hook(grab)
    out = layer(torch.randn(1, D, H, H))
    handle.remove()

    logits = captured["logits"]
    B, N, _ = logits.shape
    with torch.no_grad():
        topk_idx = torch.topk(logits.detach(), k=K, dim=-1).indices

    (out.features * torch.randn_like(out.features)).sum().backward()

    picked = torch.zeros(B, N, E, dtype=torch.bool)
    for e in range(E):
        picked[:, :, e] = (topk_idx == e).any(dim=-1)
    g_unpicked = logits.grad.detach()[~picked].abs()
    return int((g_unpicked == 0).sum()), g_unpicked.numel()


def test_renormalized_gates_leave_unpicked_experts_without_gradient() -> None:
    zeros, total = _unpicked_expert_grad("renormalized")
    assert total > 0
    assert zeros == total, (
        "the renormalised gate is expected to give exactly zero gradient to every "
        "expert a token did not pick"
    )


def test_dense_gates_give_unpicked_experts_a_gradient() -> None:
    zeros, total = _unpicked_expert_grad("dense")
    assert total > 0
    assert zeros == 0, "dense gates must propagate gradient to unpicked experts"


def test_dense_gate_is_a_dense_probability() -> None:
    """Each dense gate is a probability, so the k gates sum to at most 1 per token."""
    torch.manual_seed(0)
    layer = SpatialMoELayer(dim=D, num_experts=E, k=K, gate_mode="dense").eval()
    with torch.no_grad():
        out = layer(torch.randn(1, D, H, H))

    gate_sums = out.topk_gates.sum(dim=-1)  # [B, N]
    assert torch.all(gate_sums > 0)
    assert torch.all(gate_sums <= 1.0 + 1e-6)


def test_renormalized_gate_sums_to_one() -> None:
    """The historical gate is a softmax over the chosen pair, so it sums to 1."""
    torch.manual_seed(0)
    layer = SpatialMoELayer(dim=D, num_experts=E, k=K, gate_mode="renormalized").eval()
    with torch.no_grad():
        out = layer(torch.randn(1, D, H, H))

    gate_sums = out.topk_gates.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)
