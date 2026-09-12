"""Reporting / orchestration: heatmaps, MoE diagnostics engine, expert similarity.

``MoEDiagnosticsEngine`` is the wrapper consumed by the training loop and the
standalone diagnostics runner.  It owns collectors (one ``RoutingTracker`` and
one ``WeatherAnalyzer`` per scale), drives aggregation, and writes the JSON +
CSV outputs.  ``ExpertSimilarityAnalyzer`` is a self-contained expert-pairwise
similarity report that could move to its own module if it ever grows.
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.diagnostics.aggregation import collapse_warnings, weather_divergence
from src.diagnostics.collectors import RoutingTracker, WeatherAnalyzer
from src.log import get_logger

log = get_logger(__name__)


class Visualizer:
    """Static helpers that render routing tensors to PNG heatmaps."""

    @staticmethod
    def save_heatmap(
        tensor_2d: torch.Tensor,
        path: str,
        cmap: str = "viridis",
        title: str | None = None,
    ) -> None:
        """Save a 2-D tensor as a colormapped heatmap PNG."""
        arr = tensor_2d.detach().cpu().numpy()
        arr = np.nan_to_num(arr)

        plt.figure(figsize=(6, 6))
        plt.imshow(arr, cmap=cmap)
        plt.colorbar()
        if title:
            plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()


class MoEDiagnosticsEngine:
    """Collects and reports routing statistics per scale across one epoch.

    Args:
        output_dir: where JSON/CSV routing stats and per-expert visualizations go.
        num_experts: expert count in the MoE layers being tracked.
        top_k: experts selected per token.
    """

    def __init__(self, output_dir: str, num_experts: int = 6, top_k: int = 2) -> None:
        self.output_dir = output_dir
        self.num_experts = num_experts
        self.top_k = top_k
        os.makedirs(output_dir, exist_ok=True)
        self.trackers = {
            "moe_4": RoutingTracker(num_experts, top_k),
            "moe_8": RoutingTracker(num_experts, top_k),
            "moe_16": RoutingTracker(num_experts, top_k),
        }
        self.weather_analyzers = {
            "moe_4": WeatherAnalyzer(num_experts),
            "moe_8": WeatherAnalyzer(num_experts),
            "moe_16": WeatherAnalyzer(num_experts),
        }
        self.visualized_stems = set()

    def update(
        self,
        images: torch.Tensor,
        moe_outputs_list: List[Any],
        meta_list: Dict[str, Any],
        num_visual_samples: int = 5,
    ) -> None:
        """Accumulate one batch of MoE outputs and optionally save visuals.

        Args:
            images: the input batch ``[B, C, H, W]`` (used for per-image loops).
            moe_outputs_list: ``[out_4, out_8, out_16]`` MoE outputs.
            meta_list: batch metadata dict that may contain ``"name"`` (list of
                stems); missing names fall back to random ``batch_*`` labels.
            num_visual_samples: max number of distinct images to render.
        """
        B = images.size(0)

        # Track statistics
        self.trackers["moe_4"].update(moe_outputs_list[0])
        self.trackers["moe_8"].update(moe_outputs_list[1])
        self.trackers["moe_16"].update(moe_outputs_list[2])

        names = meta_list.get("name", [f"batch_{random.randint(0, 1000)}" for _ in range(B)])

        for i in range(B):
            stem = names[i]
            parts = stem.split("_")
            weather = parts[1] if len(parts) >= 2 else "unknown"

            # Weather tracking
            self.weather_analyzers["moe_4"].update(moe_outputs_list[0].topk_indices[i], weather)
            self.weather_analyzers["moe_8"].update(moe_outputs_list[1].topk_indices[i], weather)
            self.weather_analyzers["moe_16"].update(moe_outputs_list[2].topk_indices[i], weather)

            # Visualization
            if len(self.visualized_stems) < num_visual_samples and stem not in self.visualized_stems:
                self.visualized_stems.add(stem)
                self._generate_visuals(stem, moe_outputs_list[0], "moe_4", i)
                self._generate_visuals(stem, moe_outputs_list[1], "moe_8", i)
                self._generate_visuals(stem, moe_outputs_list[2], "moe_16", i)

    def _generate_visuals(self, stem: str, moe_out: Any, scale_name: str, batch_idx: int) -> None:
        """Render entropy, hard-assignment and soft-gate heatmaps for one image."""
        out_path = os.path.join(self.output_dir, scale_name, stem)
        os.makedirs(out_path, exist_ok=True)

        # Entropy
        ent = moe_out.entropy[batch_idx, 0]  # [H, W]
        Visualizer.save_heatmap(
            ent,
            os.path.join(out_path, f"{stem}_entropy.png"),
            cmap="magma",
            title=f"Entropy - {scale_name}",
        )

        # Hard Assignment Map & Soft Gate Map (moe_layer returns [B, H*W, K])
        indices = moe_out.topk_indices[batch_idx]  # [H*W, K]
        gates = moe_out.topk_gates[batch_idx]  # [H*W, K]
        H = int(np.sqrt(indices.size(0)))
        W = H
        indices = indices.view(H, W, self.top_k)
        gates = gates.view(H, W, self.top_k)

        for exp_id in range(self.num_experts):
            # Hard
            hard_mask = (indices == exp_id).any(dim=-1).float()
            # Soft
            soft_mask = torch.zeros(H, W, device=gates.device)
            for k_idx in range(self.top_k):
                soft_mask += (indices[:, :, k_idx] == exp_id).float() * gates[:, :, k_idx]

            if hard_mask.sum() > 0:
                Visualizer.save_heatmap(
                    hard_mask,
                    os.path.join(out_path, f"expert_{exp_id}_hard.png"),
                    cmap="gray",
                    title=f"Exp {exp_id} Hard",
                )
                Visualizer.save_heatmap(
                    soft_mask,
                    os.path.join(out_path, f"expert_{exp_id}_soft.png"),
                    cmap="viridis",
                    title=f"Exp {exp_id} Soft",
                )

    def finalize(self, epoch: int) -> Dict[str, Any]:
        """Write per-epoch routing stats (JSON + CSV) and return them.

        Args:
            epoch: epoch index embedded in the output filenames.

        Returns:
            Dict keyed by scale, each with tracker summary, collapse warnings
            and weather-enrichment divergence.
        """
        stats = {}
        for scale in ["moe_4", "moe_8", "moe_16"]:
            s = self.trackers[scale].get_stats()
            s["warnings"] = collapse_warnings(s, self.num_experts)
            w_stats = weather_divergence(
                self.weather_analyzers[scale].image_weather_stats,
                self.num_experts,
            )
            s["weather_enrichment"] = w_stats
            stats[scale] = s

        self._save_routing_stats_json(epoch, stats)
        self._save_routing_stats_csv(epoch, stats)

        return stats

    def _save_routing_stats_json(self, epoch: int, stats: Dict[str, Any]) -> None:
        """Persist the full stats dict as ``routing_stats_ep{epoch}.json``."""
        with open(os.path.join(self.output_dir, f"routing_stats_ep{epoch}.json"), "w") as f:
            json.dump(stats, f, indent=4)

    def _save_routing_stats_csv(self, epoch: int, stats: Dict[str, Any]) -> None:
        """Export a per-scale, per-expert CSV table."""
        with open(os.path.join(self.output_dir, f"routing_stats_ep{epoch}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scale", "expert_id", "hard_count", "hard_fraction", "soft_mass", "soft_fraction"])
            for scale in ["moe_4", "moe_8", "moe_16"]:
                s = stats[scale]
                for i in range(self.num_experts):
                    writer.writerow([
                        scale,
                        i,
                        s["hard_counts"][i],
                        s["hard_fractions"][i],
                        s["soft_mass"][i],
                        s["soft_fractions"][i],
                    ])


class ExpertSimilarityAnalyzer:
    """Computes weight-space and activation-space similarity for MoE experts.

    Args:
        moe_layer: a ``SpatialMoELayer`` whose ``experts`` have ``fc1``/``fc2``.
        num_experts: expert count (must match ``moe_layer``).
    """

    def __init__(self, moe_layer: Any, num_experts: int = 8) -> None:
        self.moe_layer = moe_layer
        self.num_experts = num_experts

    def compute_weight_similarity(self) -> np.ndarray:
        """Pairwise cosine similarity of each expert's fc1/fc2 weights.

        Returns:
            ``(num_experts, num_experts)`` similarity matrix.
        """
        sim_matrix = np.zeros((self.num_experts, self.num_experts))
        for i in range(self.num_experts):
            for j in range(self.num_experts):
                if i == j:
                    sim_matrix[i, j] = 1.0
                    continue
                w_i = torch.cat([
                    self.moe_layer.experts[i].fc1.weight.flatten(),
                    self.moe_layer.experts[i].fc2.weight.flatten(),
                ])
                w_j = torch.cat([
                    self.moe_layer.experts[j].fc1.weight.flatten(),
                    self.moe_layer.experts[j].fc2.weight.flatten(),
                ])
                sim = F.cosine_similarity(w_i, w_j, dim=0).item()
                sim_matrix[i, j] = sim
        return sim_matrix

    def compute_activation_similarity(self, x: torch.Tensor) -> np.ndarray:
        """Run all experts on the same tokens; pairwise cosine similarity of outputs.

        CAVEAT — residual inflation of similarity scores:
        ``TokenWiseMLPExpert.forward`` returns ``(x + z)``, not ``z`` alone. If
        ``z`` (the expert's actual contribution) is small relative to ``x``
        (the skip-connection passthrough), any two experts' outputs will look
        similar simply because both are dominated by the same ``x`` — even if
        their ``z`` contributions differ substantially.

        A ``REDUNDANT_PAIR`` warning here therefore means one of two things:
          (a) Experts genuinely converged to the same function — true collapse.
          (b) Experts are all weak (small ``z``) but not collapsed — under-training/LR issue.
        Both are real problems, but with different fixes. Do NOT conclude expert
        collapse from this metric alone. Cross-check against the weather-analysis
        KL divergence for the same layer: low KL + high activation similarity =
        stronger evidence for (a); high KL + high activation similarity = (b).

        Args:
            x: input tensor ``[B, C, H, W]`` to route through all experts.

        Returns:
            ``(num_experts, num_experts)`` mean output cosine-similarity matrix.
        """
        B, C, H, W = x.shape
        N_tokens = H * W
        x_tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        flat_x = x_tokens.reshape(B * N_tokens, C)

        expert_outputs = []
        with torch.no_grad():
            for i in range(self.num_experts):
                expert_outputs.append(self.moe_layer.experts[i](flat_x))  # [B*N_tokens, C]

        sim_matrix = np.zeros((self.num_experts, self.num_experts))
        for i in range(self.num_experts):
            for j in range(self.num_experts):
                if i == j:
                    sim_matrix[i, j] = 1.0
                    continue
                sim = F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=-1).mean().item()
                sim_matrix[i, j] = sim
        return sim_matrix

    def analyze(self, x: torch.Tensor, output_path: str) -> Dict[str, Any]:
        """Run weight + activation similarity and persist results as JSON.

        Args:
            x: input batch used for activation similarity.
            output_path: where the JSON report is saved.

        Returns:
            Dict with ``warnings``, ``weight_similarity``, ``activation_similarity``.
        """
        weight_sim = self.compute_weight_similarity()
        act_sim = self.compute_activation_similarity(x)

        warnings = []
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                if act_sim[i, j] > 0.85:
                    warnings.append(f"REDUNDANT_PAIR: Expert {i} and Expert {j} (sim: {act_sim[i, j]:.3f})")
                    log.info(f"WARNING: REDUNDANT_PAIR found - Expert {i} and Expert {j} with activation similarity {act_sim[i, j]:.3f}")

        results = {
            "warnings": warnings,
            "weight_similarity": weight_sim.tolist(),
            "activation_similarity": act_sim.tolist(),
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        return results
