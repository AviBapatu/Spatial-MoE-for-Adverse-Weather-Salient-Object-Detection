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

from typing import Any, Dict, List

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

    def update(self, moe_output: Any) -> None:
        """Accumulate one MoE output tensor group into the counters.

        Args:
            moe_output: ``MoEOutput`` object from a ``SpatialMoELayer`` layer.
        """
        topk_indices = moe_output.topk_indices.view(-1, self.top_k).cpu()  # [N_tokens, K]
        topk_gates = moe_output.topk_gates.view(-1, self.top_k).cpu().double()  # [N_tokens, K]

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
        entropy = moe_output.entropy.view(-1).cpu().double()
        self.entropy_sum += entropy.sum().item()

        # Normalize by log(K) to get 0-1 range roughly, though raw entropy can exceed log(K) if noise is weird
        # The prompt says: "Normalize by log(TOP_K) to obtain H_normalized in [0,1]"
        max_e = entropy.max().item()
        if max_e > self.entropy_max:
            self.entropy_max = max_e

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

    def update(self, topk_indices: torch.Tensor, weather: str) -> None:
        """Accumulate one image's token counts.

        Args:
            topk_indices: ``(H, W, K)`` routed-expert indices for a single image.
            weather: weather label string for the image.
        """
        N = topk_indices.numel() // topk_indices.shape[-1]
        flat_idx = topk_indices.view(-1)
        counts = torch.bincount(flat_idx, minlength=self.num_experts).cpu().numpy()
        self.image_weather_stats.append({
            "weather": weather,
            "counts": counts,
            "total": N * topk_indices.shape[-1],
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
