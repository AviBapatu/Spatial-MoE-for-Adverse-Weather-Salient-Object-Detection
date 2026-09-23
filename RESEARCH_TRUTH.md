# RESEARCH_TRUTH.md — Authoritative Scientific Claims

This file is the single source of truth for scientific claims in the paper.
Before drafting or revising any paper section, inspect this file.
Never silently add a claim not supported by implementation, experimental evidence, or verified literature.

---

## 1. Implementation-Verified Claims

These claims are verified by reading the source code. Cite file and line.

### 1.1 Architecture

| Claim | Evidence | Source |
|-------|----------|--------|
| Backbone is PVTv2-B4 | `timm.create_model('pvt_v2_b4', ...)` | `src/backbone.py:11` |
| Backbone channels are [64, 128, 320] | From timm feature_info.channels() | `src/backbone.py:19` |
| All scales projected to dim=256 | 1x1 conv per scale | `src/backbone.py:22-24` |
| 3 independent MoE layers (one per scale) | Separate module instances | `src/model.py:14-16` |
| 8 experts per scale (default) | `num_experts=8` in baseline config | `experiments/baseline_v1.json` |
| Top-k = 2 (default) | `top_k=2` in baseline config | `experiments/baseline_v1.json` |
| Router uses DWConv3x3 + MLP | Concatenates local+global context | `src/moe_layer.py:49-57` |
| Routing is noisy top-k (Shazeer-style) | Gaussian noise + softplus scale | `src/moe_layer.py:104-112` |
| Dispatch is true sparse (loop over experts) | Python for-loop, mask selection | `src/moe_layer.py:119-141` |
| Expert architecture: LN→4C→C + residual | `TokenWiseMLPExpert` | `src/moe_layer.py:14-33` |
| Entropy normalized by log(2) | Assumes K=2 | `src/decoder.py:23` |
| Decoder uses cross-attention fusion | Global (1/16→1/8), Windowed (1/8→1/4) | `src/decoder.py:232-233` |
| Boundary head exists | 1x1 conv on final features | `src/decoder.py:240` |
| Deep supervision: 3 aux heads | One per scale | `src/decoder.py:243-245` |

### 1.2 Tensor Dimensions

| Tensor | Shape | Source |
|--------|-------|--------|
| Input | `[B, 3, 384, 384]` | `src/model.py:54` |
| res_4 | `[B, 256, 96, 96]` | `src/backbone.py:62` (assertion) |
| res_8 | `[B, 256, 48, 48]` | `src/backbone.py:63` (assertion) |
| res_16 | `[B, 256, 24, 24]` | `src/backbone.py:64` (assertion) |
| Saliency logits | `[B, 1, 384, 384]` | `src/model.py:68` (assertion) |
| Boundary logits | `[B, 1, 384, 384]` | `src/model.py:69` (assertion) |
| Aux logits 16 | `[B, 1, 24, 24]` | `src/model.py:70` (assertion) |
| Aux logits 8 | `[B, 1, 48, 48]` | `src/model.py:71` (assertion) |
| Aux logits 4 | `[B, 1, 96, 96]` | `src/model.py:72` (assertion) |

### 1.3 Loss Function

| Component | Formula | Default Weight | Source |
|-----------|---------|----------------|--------|
| BCE | BCEWithLogitsLoss | 1.0 | `src/loss.py:82, 165` |
| Soft IoU | 1 - (intersection+eps)/(union+eps), per-image avg | 1.0 | `src/loss.py:86-93` |
| SSIM | Custom Gaussian window, window_size=11 | 0.0 (off) | `src/loss.py:9-57` |
| Boundary | SmoothL1(grad_mag_pred, gt_boundary) | 0.0 (off) | `src/loss.py:95-107` |
| Load balance | N * Σ(f_j * P_j) | 0.01 | `src/loss.py:125-138` |
| Importance | (std(P_j)/mean(P_j))² | 0.01 | `src/loss.py:140-144` |
| Z-loss | mean(logsumexp(logits)²) | 0.0 (off) | `src/loss.py:147-149` |
| Deep supervision | BCE on 3 aux heads, averaged | 0.4 | `src/loss.py:183-195` |

### 1.4 Training

| Claim | Evidence | Source |
|-------|----------|--------|
| DDP with NCCL, 2+ GPUs required | `dist.init_process_group("nccl")` | `src/train_ddp.py:136` |
| `find_unused_parameters=True` | Required for sparse routing | `src/train_ddp.py:296` |
| AMP FP16 with GradScaler | `torch.amp.autocast` + `GradScaler` | `src/train_ddp.py:423` |
| Backbone frozen epoch 0 only | `freeze_backbone` / `unfreeze_backbone` | `src/train_ddp.py:382-388` |
| WarmupCosine scheduler | warmup_ratio=0.01 | `src/optimization.py:84-102` |
| AdamW optimizer, weight_decay=1e-4 | Config default | `experiments/baseline_v1.json` |
| Gradient clipping max_norm=1.0 | `clip_grad_norm_` | `src/train_ddp.py:138` |
| Gradient accumulation with no_sync | `model.no_sync()` for non-final microsteps | `src/train_ddp.py:419-421` |

### 1.5 Dataset

| Claim | Evidence | Source |
|-------|----------|--------|
| WXSOD dataset, 3 splits | train_sys, test_sys, test_real | `src/dataset.py:31-41` |
| Scene-aware GroupShuffleSplit | `GroupShuffleSplit(groups=scene_ids)` | `src/dataset.py:260` |
| Strict train/val disjointness assertion | `assert isdisjoint()` | `src/dataset.py:269` |
| Filename pattern: `{scene_id}_{weather}.ext` | `get_weather_type(stem)` | `src/dataset.py:216-220` |
| Augmentations: HFlip + ColorJitter + Normalize | albumentations pipeline | `src/dataset.py:78-82` |
| Aspect-preserving resize to 384 + padding | `_aspect_preserving_resize_pad` | `src/dataset.py:107-143` |
| Deterministic per-sample augmentation | MD5-based seed per sample+epoch | `src/dataset.py:171-179` |

---

## 2. Experimentally-Verified Claims

These claims are verified from `evaluation/` and `metrics.json`.
Cite the specific evaluation directory.

### 2.1 Results: best_new_1.pth on test_sys (1,500 images)

| Metric | Value | Source |
|--------|-------|--------|
| MAE | 0.0192 | `results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/summary.txt` |
| S_measure | 0.9139 | same |
| E_adaptive | 0.9591 | same |
| E_mean | 0.9585 | same |
| E_max | 0.9633 | same |
| F_adaptive | 0.8822 | same |
| F_mean | 0.8886 | same |
| F_max | 0.9015 | same |
| boundary_MAE | 0.4817 | same |
| boundary_F1 | 0.5472 | same |

### 2.2 Results: best_new_1.pth on test_real (554 images)

| Metric | Value | Source |
|--------|-------|--------|
| MAE | 0.0168 | `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/summary.txt` |
| S_measure | 0.9151 | same |
| E_adaptive | 0.9530 | same |
| E_mean | 0.9551 | same |
| E_max | 0.9608 | same |
| F_adaptive | 0.8631 | same |
| F_mean | 0.8747 | same |
| F_max | 0.8936 | same |
| boundary_MAE | 0.6334 | same |
| boundary_F1 | 0.3601 | same |

### 2.3 Weather-Wise Breakdown (test_real)

| Weather | Count | MAE | S_measure | F_max |
|---------|-------|-----|-----------|-------|
| snow | 90 | 0.0094 | 0.9478 | 0.9421 |
| fog | 126 | 0.0148 | 0.9179 | 0.8914 |
| rain | 120 | 0.0165 | 0.9157 | 0.8935 |
| dark | 125 | 0.0191 | 0.9072 | 0.8885 |
| light | 93 | 0.0243 | 0.8897 | 0.8629 |

Source: `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/summary.txt`

### 2.4 Key Experimental Observations

| Observation | Evidence |
|-------------|----------|
| Best real-world weather: snow (MAE=0.0094) | test_real summary |
| Worst real-world weather: light (MAE=0.0243) | test_real summary |
| Compound weather degrades most on synthetic: rainafog (MAE=0.0237) | test_sys summary |
| Real-world MAE slightly better than synthetic | 0.0168 vs 0.0192 |

### 2.5 Forced-Expert Ablation

Source: `results/legacy/legacy_8expert/evaluation_results/force_expert_ablation.json`

| Metric | Normal | Forced Expert 0 (scale 4) | Delta |
|--------|--------|---------------------------|-------|
| MAE | 0.0168 | 0.0170 | +0.0002 (+1.2%) |
| S_measure | 0.9151 | 0.9139 | -0.0012 (-0.1%) |

### 2.6 Computational Cost

Source: `results/legacy/legacy_8expert/evaluation_results/compute_cost.json`

| Metric | Value |
|--------|-------|
| Parameters | 66.27M |
| MACs | 278.2G |
| FPS | 3.83 (hardware unspecified) |

---

## 3. Literature-Verified Claims

These claims are from the research blueprint or literature notes.
Cite the specific paper.

### 3.1 Novelty Positioning (from blueprint)

| Prior Work | Routing Granularity | This Work |
|------------|---------------------|-----------|
| WM-MoE (2023) | Feature-map / coarse-patch | Per-token, per-scale |
| MoWE (2023) | Image/patch level | Per-token, per-scale |
| Zhu et al. (CVPR 2023) | Channel (branch-level) | Per-token, per-scale |
| V-MoE (NeurIPS 2021) | Per-token (classification) | Per-token (dense SOD) |
| Soft MoE (ICLR 2024) | Differentiable weighting | Hard top-k with load-balance |

### 3.2 Key Differentiators

| Axis | Prior Art | This Work |
|------|-----------|-----------|
| Routing conditioning | Weather-type label/embedding | **Label-free, spatial** |
| Multi-scale routing | Single fused scale | **Independent routers at 1/4, 1/8, 1/16** |
| Task | Restoration (pixel regression) | **SOD (segmentation with topology)** |
| SOD boundary + MoE | No direct precedent | **Router-entropy-as-decoder-input** |

### 3.3 Dataset Context

| Dataset | Claim | Source |
|---------|-------|--------|
| WXSOD | 14,945 RGB images, per-image weather labels | blueprint:10-11 |
| WXSOD | Synthetic + real-world test splits | blueprint:10-11 |
| ACDC | Fog/rain/snow/night with pixel-level labels | blueprint:19 |

---

## 4. Unverified / Incomplete Claims

These are claims that need verification before inclusion in the paper.

| Claim | Status | What's Needed |
|-------|--------|---------------|
| Exact parameter count | VERIFIED: 66.27M | `results/legacy/legacy_8expert/evaluation_results/compute_cost.json` |
| Training epochs completed | Unknown | No training logs in repo |
| Comparison to SOTA methods | NOT DONE | Need to run baselines |
| Ablation study results | NOT RUN | Ablation code exists but no registry.csv |
| Router interpretability analysis | NOT RUN | Diagnostics code exists but disabled in training |
| Boundary F1 comparison to SOTA | NOT DONE | Need literature survey of boundary metrics |
| Whether `router_variant` or `moe_type` config fields are consumed | VERIFIED: NOT consumed | Forward path ignores both fields |

---

## 5. Forbidden Claims

Do not make these claims without explicit evidence:

- "Our method outperforms SOTA" — no comparison has been run
- "Expert X specializes in weather Y" — no interpretability analysis completed
- "Routing entropy correlates with difficulty" — no analysis done
- "Method works on [untested dataset]" — only WXSOD tested
- Any claim about computational efficiency — not measured against baselines
- Any claim about generalization to domains beyond WXSOD
- "Token-level routing causes improvement" — no non-routing baseline exists
- "MoE causes improvement" — no non-MoE baseline exists
- "Entropy fusion causes improvement" — no ON/OFF ablation exists
- "The model generalizes from synthetic to real" — no controlled domain-shift experiment
- "Experts specialize by weather" — no per-expert assignment analysis

---

*Last updated: 2026-09-02. Post-experiment audit.*
*This file must be updated when new experimental results are obtained or new implementation details are verified.*
