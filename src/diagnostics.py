import os
import csv
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

class RoutingTracker:
    """
    Tracks hard assignments and soft gate mass per expert, strictly mathematically.
    """
    def __init__(self, num_experts=6, top_k=2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.reset()
        
    def reset(self):
        self.hard_counts = torch.zeros(self.num_experts, dtype=torch.long)
        self.soft_mass = torch.zeros(self.num_experts, dtype=torch.float64)
        self.entropy_sum = 0.0
        self.entropy_max = 0.0
        self.total_tokens = 0
        
    def update(self, moe_output):
        """
        moe_output: MoEOutput object from SpatialMoELayer
        """
        topk_indices = moe_output.topk_indices.view(-1, self.top_k).cpu() # [N_tokens, K]
        topk_gates = moe_output.topk_gates.view(-1, self.top_k).cpu().double() # [N_tokens, K]
        
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
        
    def get_stats(self):
        total_assignments = self.total_tokens * self.top_k
        hard_fractions = (self.hard_counts.float() / max(1, total_assignments)).numpy()
        soft_fractions = (self.soft_mass / max(1, self.total_tokens)).numpy()
        mean_entropy = self.entropy_sum / max(1, self.total_tokens)
        mean_normalized_entropy = mean_entropy / max(1e-9, np.log(self.top_k))
        
        return {
            'hard_counts': self.hard_counts.numpy().tolist(),
            'hard_fractions': hard_fractions.tolist(),
            'soft_mass': self.soft_mass.numpy().tolist(),
            'soft_fractions': soft_fractions.tolist(),
            'total_tokens': self.total_tokens,
            'mean_entropy': mean_entropy,
            'mean_normalized_entropy': mean_normalized_entropy,
            'max_entropy': self.entropy_max
        }

class WeatherAnalyzer:
    """
    Computes P(weather | expert) and divergence metrics with Laplace smoothing.
    """
    def __init__(self, num_experts=6):
        self.num_experts = num_experts
        # We aggregate token counts per image, then average them
        # image_weather_stats: list of dicts. Each dict is {'weather': str, 'expert_counts': np.array}
        self.image_weather_stats = []
        
    def update(self, topk_indices, weather):
        """
        topk_indices: (H, W, K) for a single image
        weather: str
        """
        N = topk_indices.numel() // topk_indices.shape[-1]
        flat_idx = topk_indices.view(-1)
        counts = torch.bincount(flat_idx, minlength=self.num_experts).cpu().numpy()
        self.image_weather_stats.append({
            'weather': weather,
            'counts': counts,
            'total': N * topk_indices.shape[-1]
        })
        
    def compute_divergence(self, min_expert_samples=1000, min_image_samples=20):
        # Overall weather distribution (image-level)
        weather_counts = defaultdict(int)
        for stat in self.image_weather_stats:
            weather_counts[stat['weather']] += 1
            
        if len(self.image_weather_stats) < min_image_samples:
            return {"status": "insufficient_images"}
            
        total_images = len(self.image_weather_stats)
        weather_categories = sorted(list(weather_counts.keys()))
        W = len(weather_categories)
        
        P_weather_global = np.array([weather_counts[w] / total_images for w in weather_categories])
        
        # Aggregate expert -> weather (token-level weighted by image to prevent 10k token domination)
        # Actually, the user asked for image-level aggregation of token statistics:
        # "mean brightness of tokens routed to Expert 3 ... Then analyze across images"
        # For weather enrichment, we just sum the fractions per image.
        expert_weather_mass = np.zeros((self.num_experts, W))
        expert_total_mass = np.zeros(self.num_experts)
        
        # Real token counts to check MIN_EXPERT_SAMPLES
        expert_token_counts = np.zeros(self.num_experts)
        
        for stat in self.image_weather_stats:
            w_idx = weather_categories.index(stat['weather'])
            fracs = stat['counts'] / stat['total'] # [E]
            expert_weather_mass[:, w_idx] += fracs
            expert_total_mass += fracs
            expert_token_counts += stat['counts']
            
        results = []
        alpha = 1.0 # Laplace smoothing
        
        for e in range(self.num_experts):
            if expert_token_counts[e] < min_expert_samples:
                results.append({"expert": e, "status": "insufficient_tokens", "token_count": expert_token_counts[e]})
                continue
                
            # P(weather | expert)
            # p_smooth = (counts + alpha) / (total + alpha * K)
            # Here "counts" is the image-level fraction mass
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
                "p_weather_given_expert": dist_dict
            })
            
        return {
            "status": "success",
            "global_weather_prior": {weather_categories[i]: P_weather_global[i] for i in range(W)},
            "expert_enrichment": results
        }

class Visualizer:
    @staticmethod
    def save_heatmap(tensor_2d, path, cmap='viridis', title=None):
        arr = tensor_2d.detach().cpu().numpy()
        arr = np.nan_to_num(arr)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(arr, cmap=cmap)
        plt.colorbar()
        if title:
            plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        plt.close()

class MoEDiagnosticsEngine:
    def __init__(self, output_dir, num_experts=6, top_k=2):
        self.output_dir = output_dir
        self.num_experts = num_experts
        self.top_k = top_k
        os.makedirs(output_dir, exist_ok=True)
        self.trackers = {
            'moe_4': RoutingTracker(num_experts, top_k),
            'moe_8': RoutingTracker(num_experts, top_k),
            'moe_16': RoutingTracker(num_experts, top_k)
        }
        self.weather_analyzers = {
            'moe_4': WeatherAnalyzer(num_experts),
            'moe_8': WeatherAnalyzer(num_experts),
            'moe_16': WeatherAnalyzer(num_experts)
        }
        self.visualized_stems = set()
        
    def update(self, images, moe_outputs_list, meta_list, num_visual_samples=5):
        """
        moe_outputs_list: [out_4, out_8, out_16]
        """
        B = images.size(0)
        
        # Track statistics
        self.trackers['moe_4'].update(moe_outputs_list[0])
        self.trackers['moe_8'].update(moe_outputs_list[1])
        self.trackers['moe_16'].update(moe_outputs_list[2])
        
        names = meta_list.get('name', [f"batch_{random.randint(0,1000)}" for _ in range(B)])
        
        for i in range(B):
            stem = names[i]
            parts = stem.split('_')
            weather = parts[1] if len(parts) >= 2 else "unknown"
            
            # Weather tracking
            self.weather_analyzers['moe_4'].update(moe_outputs_list[0].topk_indices[i], weather)
            self.weather_analyzers['moe_8'].update(moe_outputs_list[1].topk_indices[i], weather)
            self.weather_analyzers['moe_16'].update(moe_outputs_list[2].topk_indices[i], weather)
            
            # Visualization
            if len(self.visualized_stems) < num_visual_samples and stem not in self.visualized_stems:
                self.visualized_stems.add(stem)
                self._generate_visuals(stem, moe_outputs_list[0], 'moe_4', i)
                self._generate_visuals(stem, moe_outputs_list[1], 'moe_8', i)
                self._generate_visuals(stem, moe_outputs_list[2], 'moe_16', i)

    def _generate_visuals(self, stem, moe_out, scale_name, batch_idx):
        out_path = os.path.join(self.output_dir, scale_name, stem)
        os.makedirs(out_path, exist_ok=True)
        
        # Entropy
        ent = moe_out.entropy[batch_idx, 0] # [H, W]
        Visualizer.save_heatmap(ent, os.path.join(out_path, f"{stem}_entropy.png"), cmap='magma', title=f"Entropy - {scale_name}")
        
        # Hard Assignment Map & Soft Gate Map
        # topk_indices: [K, H, W] - wait, moe_layer returns [B, H*W, K]
        indices = moe_out.topk_indices[batch_idx] # [H*W, K]
        gates = moe_out.topk_gates[batch_idx] # [H*W, K]
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
                Visualizer.save_heatmap(hard_mask, os.path.join(out_path, f"expert_{exp_id}_hard.png"), cmap='gray', title=f"Exp {exp_id} Hard")
                Visualizer.save_heatmap(soft_mask, os.path.join(out_path, f"expert_{exp_id}_soft.png"), cmap='viridis', title=f"Exp {exp_id} Soft")

    def finalize(self, epoch):
        stats = {}
        for scale in ['moe_4', 'moe_8', 'moe_16']:
            s = self.trackers[scale].get_stats()
            
            # Collapse Detectors
            hard_fracs = s['hard_fractions']
            dead_experts = [i for i, f in enumerate(hard_fracs) if f < 0.01]
            uniformity = sum([abs(f - (1.0 / self.num_experts)) for f in hard_fracs]) < 0.1
            high_entropy = s['mean_normalized_entropy'] > 0.8
            
            warnings = []
            if dead_experts:
                warnings.append(f"DEAD_EXPERT: {dead_experts}")
            if uniformity and high_entropy:
                warnings.append("LOW SPECIALIZATION / HIGH ROUTING UNCERTAINTY")
            elif max(hard_fracs) > 0.8:
                warnings.append("ROUTER COLLAPSE WARNING")
                
            s['warnings'] = warnings
            
            # Weather Enrichment
            w_stats = self.weather_analyzers[scale].compute_divergence()
            s['weather_enrichment'] = w_stats
            
            stats[scale] = s
            
        with open(os.path.join(self.output_dir, f"routing_stats_ep{epoch}.json"), "w") as f:
            json.dump(stats, f, indent=4)
            
        # CSV Export
        with open(os.path.join(self.output_dir, f"routing_stats_ep{epoch}.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['scale', 'expert_id', 'hard_count', 'hard_fraction', 'soft_mass', 'soft_fraction'])
            for scale in ['moe_4', 'moe_8', 'moe_16']:
                s = stats[scale]
                for i in range(self.num_experts):
                    writer.writerow([
                        scale, i, 
                        s['hard_counts'][i], s['hard_fractions'][i],
                        s['soft_mass'][i], s['soft_fractions'][i]
                    ])
                    
        return stats

class ExpertSimilarityAnalyzer:
    """
    Computes weight-space and activation-space similarity for experts in a MoE layer.
    """
    def __init__(self, moe_layer, num_experts=8):
        self.moe_layer = moe_layer
        self.num_experts = num_experts

    def compute_weight_similarity(self):
        """
        Computes pairwise cosine similarity between each expert's fc1 and fc2 weights.
        Returns a (num_experts, num_experts) matrix.
        """
        sim_matrix = np.zeros((self.num_experts, self.num_experts))
        for i in range(self.num_experts):
            for j in range(self.num_experts):
                if i == j:
                    sim_matrix[i, j] = 1.0
                    continue
                w_i = torch.cat([self.moe_layer.experts[i].fc1.weight.flatten(), self.moe_layer.experts[i].fc2.weight.flatten()])
                w_j = torch.cat([self.moe_layer.experts[j].fc1.weight.flatten(), self.moe_layer.experts[j].fc2.weight.flatten()])
                sim = F.cosine_similarity(w_i, w_j, dim=0).item()
                sim_matrix[i, j] = sim
        return sim_matrix

    def compute_activation_similarity(self, x):
        """
        x: input tensor [B, C, H, W]
        Run all experts on the same tokens and compute pairwise cosine similarity of outputs.

        CAVEAT — residual inflation of similarity scores:
        TokenWiseMLPExpert.forward returns (x + z), not z alone. If z (the expert's
        actual contribution) is small relative to x (the skip-connection passthrough),
        any two experts' outputs will look similar simply because both are dominated by
        the same x — even if their z contributions differ substantially.

        A REDUNDANT_PAIR warning here therefore means one of two things:
          (a) Experts genuinely converged to the same function  — true collapse.
          (b) Experts are all weak (small z) but not collapsed   — under-training / LR issue.
        Both are real problems, but with different fixes. Do NOT conclude expert collapse
        from this metric alone. Cross-check against WeatherAnalyzer KL-divergence for the
        same layer: low KL (experts see similar weather distributions) + high activation
        similarity = stronger evidence for (a); high KL + high activation similarity = (b).
        """
        B, C, H, W = x.shape
        N_tokens = H * W
        x_tokens = x.flatten(2).transpose(1, 2) # [B, H*W, C]
        flat_x = x_tokens.reshape(B * N_tokens, C)
        
        expert_outputs = []
        with torch.no_grad():
            for i in range(self.num_experts):
                expert_outputs.append(self.moe_layer.experts[i](flat_x)) # [B*N_tokens, C]
                
        sim_matrix = np.zeros((self.num_experts, self.num_experts))
        for i in range(self.num_experts):
            for j in range(self.num_experts):
                if i == j:
                    sim_matrix[i, j] = 1.0
                    continue
                # Cosine similarity along C dimension, then average over tokens
                sim = F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=-1).mean().item()
                sim_matrix[i, j] = sim
                
        return sim_matrix

    def analyze(self, x, output_path):
        """
        x: input batch to use for activation similarity.
        output_path: path to save the JSON file.
        """
        weight_sim = self.compute_weight_similarity()
        act_sim = self.compute_activation_similarity(x)
        
        warnings = []
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                if act_sim[i, j] > 0.85:
                    warnings.append(f"REDUNDANT_PAIR: Expert {i} and Expert {j} (sim: {act_sim[i, j]:.3f})")
                    print(f"WARNING: REDUNDANT_PAIR found - Expert {i} and Expert {j} with activation similarity {act_sim[i, j]:.3f}")
                    
        results = {
            "warnings": warnings,
            "weight_similarity": weight_sim.tolist(),
            "activation_similarity": act_sim.tolist()
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
            
        return results

