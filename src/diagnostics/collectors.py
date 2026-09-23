"""Raw routing-stat collectors.

Each collector accumulates per-batch routing statistics in a rank-local,
fully additive manner: a ``RoutingTracker`` or ``WeatherAnalyzer`` is fed one
batch (or one image) at a time and keeps only running primitive sums.  This
makes collectors safe to use in DDP, where every rank already receives a
distinct, rank-aware shard of the validation set (``DistributedSampler`` on
``val_loader``).  Cross-rank reduction is just a matter of merging the raw
``state()`` dicts (see :func:`src.diagnostics.aggregation.merge_states`);
summary math lives in ``aggregation.py``, never here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class RoutingTracker:
    """Tracks hard assignments and soft gate mass per expert, by raw sums.

    ``update()`` accumulates, ``state()`` exposes a JSON-serializable raw
    snapshot for all-gather, and ``get_stats()`` projects the accumulated
    state into the summary dict consumed by ``MoEDiagnosticsEngine.finalize``.
    """

    def __init__(self, num_experts: int = 6, top_k: int = 2) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.reset()

    def reset(self) -> None:
        """Zero all accumulated counters (called from ``__init__``)."""
        self.hard_counts = torch.zeros(self.num_experts, dtype=torch.long)
        self.soft_mass = torch.zeros(self.num_experts, dtype=torch.float64)
        self.entropy_sum = 0.0
        self.entropy_max = 0.0
        self.total_tokens = 0
        self.masked_tokens = 0  # padding tokens filtered out when pad_mask is given
        self.logit_std_sum = 0.0
        self.top1_top2_margin_sum = 0.0

    def update(
        self,
        moe_output: Any,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate one MoE output tensor group into the counters.

        Args:
            moe_output: ``MoEOutput`` object from a ``SpatialMoELayer`` layer.
            pad_mask: Optional flat boolean tensor ``[B * N_tokens]`` where
                ``True`` marks a valid (non-padding) token.  When provided,
                only valid tokens contribute to all statistics.  When
                ``None``, all tokens are used (original behaviour, fully
                backward-compatible).
        """
        topk_indices = moe_output.topk_indices.view(-1, self.top_k).cpu()  # [N_tokens, K]
        topk_gates = moe_output.topk_gates.view(-1, self.top_k).cpu().double()  # [N_tokens, K]
        entropy = moe_output.entropy.view(-1).cpu().double()  # [N_tokens]

        # Router logit geometry, flattened to [N_tokens, E].  Normalized entropy
        # saturates near 1.0 for any weakly-trained router, so these two
        # distinguish "the logits are flat" from "the logits are large but tied".
        logits = moe_output.clean_logits.float().cpu()
        logits = (
            logits.view(logits.size(0), logits.size(1), -1)
            .permute(0, 2, 1)
            .reshape(-1, logits.size(1))
        )

        if pad_mask is not None:
            valid = pad_mask.view(-1).bool().cpu()  # [N_tokens]
            self.masked_tokens += int((~valid).sum().item())
            topk_indices = topk_indices[valid]
            topk_gates = topk_gates[valid]
            entropy = entropy[valid]
            logits = logits[valid]

        N = topk_indices.size(0)

        # Hard Counts
        for k_idx in range(self.top_k):
            counts = torch.bincount(topk_indices[:, k_idx], minlength=self.num_experts)
            self.hard_counts += counts

        # Soft Mass
        for k_idx in range(self.top_k):
            indices = topk_indices[:, k_idx]
            gates = topk_gates[:, k_idx]
            mass = torch.zeros(self.num_experts, dtype=torch.float64)
            mass.scatter_add_(0, indices, gates)
            self.soft_mass += mass

        # Entropy
        self.entropy_sum += entropy.sum().item()
        if N > 0:
            max_e = entropy.max().item()
            if max_e > self.entropy_max:
                self.entropy_max = max_e

            # Logit spread across experts, and the top-1/top-2 decision margin.
            self.logit_std_sum += logits.std(dim=-1).sum().item()
            if logits.size(1) >= 2:
                top2 = logits.topk(2, dim=-1).values
                self.top1_top2_margin_sum += (top2[:, 0] - top2[:, 1]).sum().item()

        self.total_tokens += N

    def get_stats(self) -> Dict[str, Any]:
        """Return the summary dict (projected from this tracker's raw state)."""
        from src.diagnostics.aggregation import routing_summary

        return routing_summary(self)

    def state(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot for cross-rank all-gather."""
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "hard_counts": self.hard_counts.tolist(),
            "soft_mass": self.soft_mass.tolist(),
            "entropy_sum": self.entropy_sum,
            "entropy_max": self.entropy_max,
            "total_tokens": self.total_tokens,
            "masked_tokens": self.masked_tokens,
            "logit_std_sum": self.logit_std_sum,
            "top1_top2_margin_sum": self.top1_top2_margin_sum,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "RoutingTracker":
        """Reconstruct a tracker from a ``state()`` snapshot."""
        tracker = cls(num_experts=state["num_experts"], top_k=state["top_k"])
        tracker.hard_counts = torch.tensor(state["hard_counts"], dtype=torch.long)
        tracker.soft_mass = torch.tensor(state["soft_mass"], dtype=torch.float64)
        tracker.entropy_sum = state["entropy_sum"]
        tracker.entropy_max = state["entropy_max"]
        tracker.total_tokens = state["total_tokens"]
        # Default to 0 for backward compat with snapshots saved before this field existed.
        tracker.masked_tokens = state.get("masked_tokens", 0)
        tracker.logit_std_sum = state.get("logit_std_sum", 0.0)
        tracker.top1_top2_margin_sum = state.get("top1_top2_margin_sum", 0.0)
        return tracker

    def merge(self, other: "RoutingTracker") -> "RoutingTracker":
        """Add ``other``'s accumulations into this tracker (in place).

        Used to combine rank-local shards: ``acc.merge(RoutingTracker.from_state(x))``.
        """
        assert self.num_experts == other.num_experts, "num_experts mismatch on merge"
        assert self.top_k == other.top_k, "top_k mismatch on merge"
        self.hard_counts += other.hard_counts
        self.soft_mass += other.soft_mass
        self.entropy_sum += other.entropy_sum
        self.entropy_max = max(self.entropy_max, other.entropy_max)
        self.total_tokens += other.total_tokens
        self.masked_tokens += other.masked_tokens
        self.logit_std_sum += other.logit_std_sum
        self.top1_top2_margin_sum += other.top1_top2_margin_sum
        return self


class WeatherAnalyzer:
    """Aggregates per-image expert token counts labelled by weather category.

    The per-image token counts (rather than raw tokens) let each image
    contribute equally to the weather-expert divergence estimate, preventing a
    single high-resolution image from dominating the statistics.
    """

    def __init__(self, num_experts: int = 6) -> None:
        self.num_experts = num_experts
        # image_weather_stats: list of dicts. Each dict is {'weather': str, 'counts': np.ndarray, 'total': int}
        self.image_weather_stats: List[Dict[str, Any]] = []

    def update(
        self,
        topk_indices: torch.Tensor,
        weather: str,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate one image's token counts.

        Args:
            topk_indices: ``(N_tokens, K)`` or ``(H, W, K)`` routed-expert
                indices for a single image.
            weather: weather label string for the image.
            pad_mask: Optional flat boolean tensor ``[N_tokens]`` where
                ``True`` marks a valid (non-padding) token.  When provided,
                only valid tokens contribute to the per-expert counts.  When
                ``None``, all tokens are used (original behaviour).
        """
        K = topk_indices.shape[-1]
        flat_indices = topk_indices.view(-1, K).cpu()  # [N_tokens, K]

        if pad_mask is not None:
            valid = pad_mask.view(-1).bool().cpu()  # [N_tokens]
            flat_indices = flat_indices[valid]

        N = flat_indices.size(0)
        counts = torch.bincount(flat_indices.view(-1), minlength=self.num_experts).numpy()
        self.image_weather_stats.append({
            "weather": weather,
            "counts": counts,
            "total": N * K,
        })

    def compute_divergence(
        self, min_expert_samples: int = 1000, min_image_samples: int = 20
    ) -> Dict[str, Any]:
        """Compute weather-enrichment divergence; delegate to aggregation.

        Kept as a method for API compatibility with
        ``MoEDiagnosticsEngine.finalize``; the math itself lives in
        :func:`src.diagnostics.aggregation.weather_divergence`.
        """
        from src.diagnostics.aggregation import weather_divergence

        return weather_divergence(
            self.image_weather_stats,
            self.num_experts,
            min_expert_samples=min_expert_samples,
            min_image_samples=min_image_samples,
        )
