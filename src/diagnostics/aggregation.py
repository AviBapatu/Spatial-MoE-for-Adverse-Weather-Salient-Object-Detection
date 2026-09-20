"""Pure-functions that reduce collected routing stats into summary metrics.

Every function here is deterministic, side-effect free, and takes either a
collector or plain data structures — no model access, no I/O.  This is the
layer ``tests/test_diagnostics.py`` targets for correctness, and the layer a
DDP run calls after ``all_gather_object`` to build one global summary from the
per-rank ``RoutingTracker.state()`` snapshots.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np

from src.diagnostics.collectors import RoutingTracker


def routing_summary(tracker: RoutingTracker) -> Dict[str, Any]:
    """Project a tracker's raw accumulation into the summary metric dict.

    Returns a dict with:
    ``hard_counts``, ``hard_fractions``, ``soft_mass``, ``soft_fractions``,
    ``total_tokens``, ``mean_entropy``, ``mean_normalized_entropy``,
    ``max_entropy``.
    """
    total_assignments = tracker.total_tokens * tracker.top_k
    hard_fractions = (tracker.hard_counts.float() / max(1, total_assignments)).numpy()
    soft_fractions = (tracker.soft_mass / max(1, tracker.total_tokens)).numpy()
    mean_entropy = tracker.entropy_sum / max(1, tracker.total_tokens)
    mean_normalized_entropy = mean_entropy / max(1e-9, np.log(tracker.num_experts))

    return {
        "hard_counts": tracker.hard_counts.numpy().tolist(),
        "hard_fractions": hard_fractions.tolist(),
        "soft_mass": tracker.soft_mass.numpy().tolist(),
        "soft_fractions": soft_fractions.tolist(),
        "total_tokens": tracker.total_tokens,
        "mean_entropy": mean_entropy,
        "mean_normalized_entropy": mean_normalized_entropy,
        "max_entropy": tracker.entropy_max,
    }


def merge_states(
    states: Sequence[Dict[str, Any]], num_experts: int, top_k: int
) -> Dict[str, Any]:
    """Merge rank-local ``RoutingTracker.state()`` snapshots into one summary.

    Args:
        states: per-rank raw snapshots produced by ``RoutingTracker.state()``.
        num_experts: expert count (must match the trackers that made ``states``).
        top_k: expert count per token (must match the trackers).

    Returns:
        The combined ``routing_summary`` over all input shards.
    """
    merged = RoutingTracker(num_experts=num_experts, top_k=top_k)
    for state in states:
        merged.merge(RoutingTracker.from_state(state))
    return routing_summary(merged)


def collapse_warnings(summary: Dict[str, Any], num_experts: int) -> List[str]:
    """Detect expert collapse / dead-expert / low-specialization conditions.

    Args:
        summary: dict from :func:`routing_summary`.
        num_experts: total expert count.

    Returns:
        A list of human-readable warning strings (empty when healthy).
    """
    hard_fracs = summary["hard_fractions"]
    dead_experts = [i for i, f in enumerate(hard_fracs) if f < 0.01]
    uniformity = sum([abs(f - (1.0 / num_experts)) for f in hard_fracs]) < 0.1
    high_entropy = summary["mean_normalized_entropy"] > 0.8

    warnings = []
    if dead_experts:
        warnings.append(f"DEAD_EXPERT: {dead_experts}")
    if uniformity and high_entropy:
        warnings.append("LOW SPECIALIZATION / HIGH ROUTING UNCERTAINTY")
    elif max(hard_fracs) > 0.8:
        warnings.append("ROUTER COLLAPSE WARNING")
    return warnings


def weather_divergence(
    image_weather_stats: Sequence[Dict[str, Any]],
    num_experts: int,
    min_expert_samples: int = 1000,
    min_image_samples: int = 20,
) -> Dict[str, Any]:
    """Compute P(weather | expert) and KL/JS divergence metrics per expert.

    Args:
        image_weather_stats: per-image stats from ``WeatherAnalyzer``
            (each ``{'weather': str, 'counts': np.ndarray, 'total': int}``).
        num_experts: total expert count.
        min_expert_samples: minimum token count per expert before it is
            classified ``insufficient_tokens``.
        min_image_samples: minimum number of labeled images before the whole
            computation is skipped (``insufficient_images``).

    Returns:
        A dict with ``status``, ``global_weather_prior`` and a per-expert
        ``expert_enrichment`` list.  Mirrors the historical behavior exactly.
    """
    weather_counts = defaultdict(int)
    for stat in image_weather_stats:
        weather_counts[stat["weather"]] += 1

    if len(image_weather_stats) < min_image_samples:
        return {"status": "insufficient_images"}

    total_images = len(image_weather_stats)
    weather_categories = sorted(list(weather_counts.keys()))
    W = len(weather_categories)

    P_weather_global = np.array([weather_counts[w] / total_images for w in weather_categories])

    # Aggregate expert -> weather (image-level fraction mass, so a 10k-token
    # image does not dominate a 20-token one).
    expert_weather_mass = np.zeros((num_experts, W))
    expert_total_mass = np.zeros(num_experts)

    # Real token counts to check MIN_EXPERT_SAMPLES
    expert_token_counts = np.zeros(num_experts)

    for stat in image_weather_stats:
        w_idx = weather_categories.index(stat["weather"])
        fracs = stat["counts"] / stat["total"]  # [E]
        expert_weather_mass[:, w_idx] += fracs
        expert_total_mass += fracs
        expert_token_counts += stat["counts"]

    results = []
    alpha = 1.0  # Laplace smoothing

    for e in range(num_experts):
        if expert_token_counts[e] < min_expert_samples:
            results.append({"expert": e, "status": "insufficient_tokens", "token_count": expert_token_counts[e]})
            continue

        # P(weather | expert), Laplace-smoothed on the image-level mass.
        counts_e = expert_weather_mass[e, :]
        total_e = expert_total_mass[e]

        p_w_given_e = (counts_e + alpha) / (total_e + alpha * W)

        # Global prior smoothed similarly
        p_w_global = (np.array([weather_counts[w] for w in weather_categories]) + alpha) / (total_images + alpha * W)

        # KL Divergence: sum(P(w|e) * log(P(w|e) / P(w)))
        kl = np.sum(p_w_given_e * np.log(p_w_given_e / p_w_global))

        # JS Divergence
        m = 0.5 * (p_w_given_e + p_w_global)
        js = 0.5 * np.sum(p_w_given_e * np.log(p_w_given_e / m)) + 0.5 * np.sum(p_w_global * np.log(p_w_global / m))

        dist_dict = {weather_categories[i]: p_w_given_e[i] for i in range(W)}

        results.append({
            "expert": e,
            "status": "valid",
            "token_count": expert_token_counts[e],
            "kl_div": kl,
            "js_div": js,
            "p_weather_given_expert": dist_dict,
        })

    return {
        "status": "success",
        "global_weather_prior": {weather_categories[i]: P_weather_global[i] for i in range(W)},
        "expert_enrichment": results,
    }
