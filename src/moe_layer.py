"""Spatially-dynamic Mixture-of-Experts layer.

Contains the token-wise expert MLP, the noisy top-k router, and the true
sparse-dispatch computation that routes each spatial token to its ``k``
selected experts.
"""
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.log import get_logger

log = get_logger(__name__)


class MoEOutput(NamedTuple):
    """Container bundling the MoE forward-pass outputs.

    Attributes:
        features: Dispatched, gate-weighted expert output, shape ``[B, C, H, W]``.
        routing_probs: Sparse per-expert gate probabilities, shape ``[B, E, H, W]``.
        topk_indices: Selected expert ids per token, shape ``[B, N_tokens, K]``.
        topk_gates: Gate weights for the selected experts, shape ``[B, N_tokens, K]``.
        entropy: Per-token routing entropy map, shape ``[B, 1, H, W]``.
        clean_logits: Router logits *without* noise, shape ``[B, E, H, W]``.
        noise_std: Per-token noise std ``[B, N_tokens, E]`` during training, else ``None``.
    """

    features: torch.Tensor
    routing_probs: torch.Tensor
    topk_indices: torch.Tensor
    topk_gates: torch.Tensor
    entropy: torch.Tensor
    clean_logits: torch.Tensor
    noise_std: Optional[torch.Tensor] = None
    # False for stages that contain no router (dense / passthrough ablations), so
    # consumers can skip their sentinel routing fields instead of treating them as
    # real routing statistics.
    has_router: bool = True


class TokenWiseMLPExpert(nn.Module):
    """Token-wise residual MLP expert.

    ``LayerNorm -> Linear(C -> 4C) -> GELU -> Linear(4C -> C) -> residual``.

    Input/Output shape: ``[N_tokens, C]``.  The forward is safe to call with
    an empty token tensor (``[0, C]``), producing zero gradients instead of
    ``None`` for its parameters — this is the mechanism that keeps DDP
    gradient sync consistent when an expert receives no routed tokens.
    """

    def __init__(self, dim: int, expansion: int = 4) -> None:
        """Initialize the expert MLP.

        Args:
            dim: Token feature dimensionality ``C``.
            expansion: Hidden-to-input width ratio (default ``4``).
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual MLP to ``[N_tokens, C]`` tokens."""
        z = self.norm(x)
        z = self.fc1(z)
        z = self.act(z)
        z = self.fc2(z)
        return x + z


class RouterNoise(nn.Module):
    """Learned, input-dependent Gaussian noise for noisy top-k routing.

    Implements the Shazeer-style noise perturbation applied to router logits
    during training:

    .. code-block:: python

        noise_std    = noise_scale * softplus(W_noise @ x)
        noise_std    = clamp(noise_std, min=noise_min_std)
        noise        = randn_like(clean_logits) * noise_std
        noisy_logits = clean_logits + noise

    ``W_noise`` is a learned linear projection; the clipping floor prevents the
    noise from collapsing toward zero as ``softplus`` saturates.
    """

    def __init__(self, dim: int, num_experts: int) -> None:
        """Initialize the learned noise projection.

        Args:
            dim: Token feature dimensionality ``C``.
            num_experts: Number of router output logits / experts.
        """
        super().__init__()
        self.noise_linear = nn.Linear(dim, num_experts)

    def add_noise(
        self,
        clean_logits: torch.Tensor,
        x_tokens: torch.Tensor,
        training: bool,
        noise_scale: float = 1.0,
        noise_min_std: float = 0.05,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Add learned noise to *clean_logits* when ``training`` is ``True``.

        Args:
            clean_logits: Router logits ``[B, N_tokens, E]`` before noise.
            x_tokens: Token features ``[B, N_tokens, C]`` used to compute the
                input-dependent noise scale.
            training: Whether the model is in training mode.
            noise_scale: Global multiplier on the softplus noise std.
            noise_min_std: Lower clamp on the noise std.

        Returns:
            ``(noisy_logits, noise_std)`` — when *training* is ``False`` the
            logits are returned untouched and ``noise_std`` is ``None``.
        """
        if not training:
            return clean_logits, None

        noise_std = noise_scale * F.softplus(self.noise_linear(x_tokens))
        noise_std = torch.clamp(noise_std, min=noise_min_std)
        noise = torch.randn_like(clean_logits) * noise_std
        return clean_logits + noise, noise_std


class SpatialMoELayer(nn.Module):
    """Spatially-dynamic MoE operating on a 2D feature map.

    Each spatial token is routed through a depthwise-conv + MLP router to the
    top-``k`` experts of a shared expert pool.  Dispatch is *true-sparse*: only
    the tokens that selected an expert are passed to it, and the outputs are
    gate-weighted and scattered back.

    DDP safety: every expert is invoked on every forward pass (on its routed
    tokens or an empty slice when it received none), so each expert's
    parameters always receive at least a zero gradient.  This keeps DDP's
    ``all_reduce`` consistent across ranks with different routing patterns and
    must NOT be "optimized" into conditional expert calls.
    """

    def __init__(
        self,
        dim: int = 256,
        num_experts: int = 8,
        k: int = 2,
        router_hidden: int = 128,
        router_noise_enabled: bool = True,
        router_noise_scale: float = 1.0,
        router_noise_min_std: float = 0.05,
        gate_mode: str = "renormalized",
    ) -> None:
        """Initialize the MoE layer.

        Args:
            dim: Feature dimensionality ``C``.
            num_experts: Size of the shared expert pool.
            k: Number of experts selected per token (top-``k``).
            router_hidden: Hidden width of the router MLP.
            router_noise_enabled: Whether noisy routing is active in training.
            router_noise_scale: Multiplier for the learned noise std.
            router_noise_min_std: Lower clamp on the noise std.
            gate_mode: ``"renormalized"`` (softmax over the k chosen logits) or
                ``"dense"`` (each chosen expert's probability from the full
                softmax over all experts).  See :meth:`forward`.
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k
        self.router_noise_enabled = router_noise_enabled
        self.router_noise_scale = router_noise_scale
        self.router_noise_min_std = router_noise_min_std
        self.gate_mode = gate_mode

        # 1. Expert pool (sparse dispatch preserves DDP gradient sync).
        self.experts = nn.ModuleList(
            [TokenWiseMLPExpert(dim=dim) for _ in range(num_experts)]
        )

        # 2. Spatial context router: depthwise conv captures local context.
        self.router_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # 3. MLP producing clean routing logits from (token, local context).
        self.router_mlp = nn.Sequential(
            nn.Linear(2 * dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts),
        )

        # 4. Shazeer-style noisy routing projection.
        self.router_noise = RouterNoise(dim=dim, num_experts=num_experts)

    def _forward_forced_expert(self, x: torch.Tensor, expert_id: int) -> MoEOutput:
        """Counterfactual-ablation path that routes every token to one expert.

        Used only for ablation/diagnostics experiments (evaluation, not DDP
        training), so the sparse-dispatch touch-every-expert invariant does not
        apply here.

        Args:
            x: Input feature map ``[B, C, H, W]``.
            expert_id: Index of the forced expert.

        Returns:
            An ``MoEOutput`` with one-hot routing to *expert_id*.
        """
        B, C, H, W = x.shape
        N_tokens = H * W
        x_tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        expert = self.experts[expert_id]
        flat_x = x_tokens.reshape(B * N_tokens, C)
        expert_out = expert(flat_x)
        fused_feature_map = (
            expert_out.view(B, N_tokens, C).transpose(1, 2).view(B, C, H, W).contiguous()
        )

        topk_indices = torch.full(
            (B, N_tokens, self.k), expert_id, device=x.device, dtype=torch.long
        )
        topk_gates = torch.zeros(B, N_tokens, self.k, device=x.device, dtype=x.dtype)
        topk_gates[:, :, 0] = 1.0

        routing_probs = torch.zeros(B, self.num_experts, H, W, device=x.device, dtype=x.dtype)
        routing_probs[:, expert_id, :, :] = 1.0

        entropy_map = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        clean_logits_spatial = torch.zeros(
            B, self.num_experts, H, W, device=x.device, dtype=x.dtype
        )
        clean_logits_spatial[:, expert_id, :, :] = 1.0

        return MoEOutput(
            features=fused_feature_map,
            routing_probs=routing_probs,
            topk_indices=topk_indices,
            topk_gates=topk_gates,
            entropy=entropy_map,
            clean_logits=clean_logits_spatial,
            noise_std=None,
        )

    def forward(
        self,
        x: torch.Tensor,
        force_expert_id: Optional[int] = None,
        random_routing: bool = False,
    ) -> MoEOutput:
        """Route and dispatch *x* through the expert pool.

        Args:
            x: Input feature map ``[B, C, H, W]``.
            force_expert_id: If set, bypass routing and force every token to
                this expert (counterfactual ablation).
            random_routing: If ``True``, replace clean logits with random ones
                (ablation for removing the learned routing).

        Returns:
            An ``MoEOutput`` with the fused feature map and routing diagnostics.
        """
        B, C, H, W = x.shape
        N_tokens = H * W

        x_tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        if force_expert_id is not None:
            return self._forward_forced_expert(x, force_expert_id)

        # --- Routing input ---
        local_feat = self.router_dwconv(x)
        local_tokens = local_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        router_input = torch.cat([x_tokens, local_tokens], dim=-1)  # [B, H*W, 2C]

        # --- Routing ---
        clean_logits = self.router_mlp(router_input)  # [B, H*W, E]

        if random_routing:  # Ablation: replace learned routing with random logits
            clean_logits = torch.randn_like(clean_logits)

        if self.training and self.router_noise_enabled:
            noisy_logits, noise_std = self.router_noise.add_noise(
                clean_logits,
                x_tokens,
                training=self.training,
                noise_scale=self.router_noise_scale,
                noise_min_std=self.router_noise_min_std,
            )
        else:
            noisy_logits, noise_std = clean_logits, None

        topk_values, topk_indices = torch.topk(noisy_logits, k=self.k, dim=-1)
        # In "dense" mode the gate is the chosen expert's probability under the
        # FULL softmax over all experts, so the loss keeps a gradient for the
        # experts this token did not pick — that is what lets the router learn
        # which experts to prefer.  Renormalising over the k chosen logits (the
        # historical default) makes the gate shift-invariant within that pair,
        # leaving the task gradient exactly zero for every unpicked expert
        # (measured: 128/128 zero entries), so selection could only drift under
        # noise and the auxiliary uniformity terms.
        full_gates = F.softmax(noisy_logits, dim=-1)  # [B, H*W, E]
        if self.gate_mode == "dense":
            topk_gates = torch.gather(full_gates, -1, topk_indices)  # [B, H*W, K]
        else:
            topk_gates = F.softmax(topk_values, dim=-1)  # [B, H*W, K]

        # --- True sparse dispatch -------------------------------------------
        flat_x = x_tokens.reshape(B * N_tokens, C)
        flat_topk_indices = topk_indices.reshape(B * N_tokens, self.k)
        flat_topk_gates = topk_gates.reshape(B * N_tokens, self.k)

        output_tokens = torch.zeros_like(flat_x)

        # Every expert is called on every forward pass — including on an empty
        # token slice when it received no routes — so its parameters always
        # accumulate a (possibly zero) gradient.  Do NOT make these calls
        # conditional: DDP's gradient sync depends on this exact pattern.
        for i, expert in enumerate(self.experts):
            active_mask = flat_topk_indices == i  # [B*N_tokens, K]
            token_active = active_mask.any(dim=-1)  # [B*N_tokens] boolean

            selected_tokens = flat_x[token_active]  # [N_active, C], possibly [0, C]
            expert_out = expert(selected_tokens)  # [N_active, C]

            # Gate weight of the routing slot that selected this expert.
            expert_gates = flat_topk_gates[token_active][active_mask[token_active]]

            weighted_out = expert_out * expert_gates.unsqueeze(-1)

            output_tokens[token_active] += weighted_out

        fused_feature_map = (
            output_tokens.view(B, N_tokens, C).transpose(1, 2).view(B, C, H, W).contiguous()
        )

        # --- Diagnostics / reconstruction -----------------------------------
        # Sparse per-expert gate probabilities [B, E, H, W] for auxiliary losses.
        zeros = torch.zeros_like(noisy_logits, dtype=topk_gates.dtype)
        routing_gates = zeros.scatter(-1, topk_indices, topk_gates)
        routing_probs = (
            routing_gates.transpose(1, 2).view(B, self.num_experts, H, W).contiguous()
        )

        # Routing entropy over the FULL E-expert gate distribution (range [0, ln E]).
        # Note: Previous version calculated this over top-K gates only, which was
        # a measurement bug that artificially capped entropy at ln(K).
        entropy = -torch.sum(full_gates * torch.log(full_gates + 1e-9), dim=-1)
        entropy_map = entropy.view(B, 1, H, W).contiguous()

        # Clean (noise-free) logits reshaped to [B, E, H, W].
        clean_logits_spatial = (
            clean_logits.transpose(1, 2).view(B, self.num_experts, H, W).contiguous()
        )

        return MoEOutput(
            features=fused_feature_map,
            routing_probs=routing_probs,
            topk_indices=topk_indices,
            topk_gates=topk_gates,
            entropy=entropy_map,
            clean_logits=clean_logits_spatial,
            noise_std=noise_std,
        )


class DenseMoE16Adapter(nn.Module):
    """Single dense expert replacing sparse moe_16 for the ablation.

    Applies one TokenWiseMLPExpert to every token unconditionally and
    returns a MoEOutput with sentinel routing fields (zero logits, uniform
    gate, K=2 to match the sparse stage's NamedTuple shape) so the decoder
    receives features of the correct shape without any routing computation.
    The routing sentinel fields are intentionally excluded from routing losses
    by slicing moe_outputs[:2] in CombinedLoss when moe_16_dense=True.
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.expert = TokenWiseMLPExpert(dim=dim)
        self.is_dense = True  # no sparse router here; diagnostics skip this stage

    def forward(
        self,
        x: torch.Tensor,
        force_expert_id: Optional[int] = None,   # absorbed, unused
        random_routing: bool = False,             # absorbed, unused
    ) -> MoEOutput:
        B, C, H, W = x.shape
        N = H * W
        flat_x = x.flatten(2).transpose(1, 2).reshape(B * N, C)
        out_tokens = self.expert(flat_x)                              # [B*N, C]
        features = out_tokens.view(B, N, C).transpose(1, 2).view(B, C, H, W).contiguous()

        # Sentinel routing tensors — K=2 to match real SpatialMoELayer output shape.
        # Any code that consumes moe_outputs[2] routing fields (diagnostics, similarity
        # analyzer, RoutingTracker) must skip this stage; see diagnostic skip logic.
        sentinel_indices = torch.zeros(B, N, 2, device=x.device, dtype=torch.long)
        sentinel_gates   = torch.full((B, N, 2), 0.5, device=x.device, dtype=x.dtype)

        return MoEOutput(
            features=features,
            routing_probs=torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype),
            topk_indices=sentinel_indices,
            topk_gates=sentinel_gates,
            entropy=torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype),
            clean_logits=torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype),
            has_router=False,
        )


class PassthroughMoELayer(nn.Module):
    """No MoE at all: the features pass through untouched (``moe_type="none"``).

    Returns a :class:`MoEOutput` carrying the input features and sentinel routing
    fields, so the decoder keeps receiving the same structure as the sparse path.
    Note this removes both the routing *and* the expert parameters — so it is a
    different-capacity control; use ``DenseMoE16Adapter`` for the capacity-matched
    "same experts, no routing" comparison.
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.is_dense = True  # no sparse router here; diagnostics skip this stage

    def forward(
        self,
        x: torch.Tensor,
        force_expert_id: Optional[int] = None,   # absorbed, unused
        random_routing: bool = False,             # absorbed, unused
    ) -> MoEOutput:
        B, C, H, W = x.shape
        N = H * W
        return MoEOutput(
            features=x,
            routing_probs=torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype),
            topk_indices=torch.zeros(B, N, 2, device=x.device, dtype=torch.long),
            topk_gates=torch.full((B, N, 2), 0.5, device=x.device, dtype=x.dtype),
            entropy=torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype),
            clean_logits=torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype),
            has_router=False,
        )



if __name__ == "__main__":
    log.info("Testing SpatialMoELayer...")

    batch_size = 2
    channels = 256
    H, W = 48, 48

    dummy_input = torch.randn(batch_size, channels, H, W)
    log.info(f"Input shape: {dummy_input.shape}")

    moe_layer = SpatialMoELayer(dim=channels, num_experts=8, k=2)

    out = moe_layer(dummy_input)

    log.info("Output shapes:")
    log.info(f"  Fused features: {out.features.shape}")
    log.info(f"  Routing probabilities: {out.routing_probs.shape}")
    log.info(f"  TopK indices: {out.topk_indices.shape}")
    log.info(f"  TopK gates: {out.topk_gates.shape}")
    log.info(f"  Entropy map: {out.entropy.shape}")
    log.info(f"  Clean logits: {out.clean_logits.shape}")

    log.info("Basic execution successful!")
