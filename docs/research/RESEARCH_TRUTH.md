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
| All scales projected to dim=256 | 1×1 conv per scale | `src/backbone.py:22-24` |
| 3 independent MoE layers (one per scale) | Separate module instances | `src/model.py:14-16` |
| 8 experts per scale (baseline config) | `num_experts=8` | `experiments/baseline_v1.json:14` |
| Top-k = 2 | `top_k=2` | `experiments/baseline_v1.json:15` |
| Router uses DWConv3x3 + MLP (2-layer) | Concatenates local+global, Linear→GELU→Linear | `src/moe_layer.py:49-57` |
| Routing is noisy top-k (Shazeer-style) | Gaussian noise + softplus scale | `src/moe_layer.py:104-112` |
| Dispatch is true sparse (loop over experts) | Python for-loop, mask selection, scatter-add | `src/moe_layer.py:119-141` |
| Expert architecture: LN→4C→C + residual MLP | `TokenWiseMLPExpert` | `src/moe_layer.py:14-33` |
| Entropy normalized by log(2) in decoder | Assumes K=2 | `src/decoder.py:23` |
| Decoder uses cross-attention fusion | Global (1/16→1/8), Windowed (1/8→1/4) | `src/decoder.py:232-233` |
| Boundary head exists | 1×1 conv on final features | `src/decoder.py:240` |
| Deep supervision: 3 aux heads | One per scale | `src/decoder.py:243-245` |
| RefinementBlocks: 2× (Conv3×3→GN→ReLU) + residual | 3 blocks total (1/4, 1/2, 1/1 scales) | `src/decoder.py:205-217` |

### 1.2 Tensor Dimensions

| Tensor | Shape | Source |
|--------|-------|--------|
| Input | `[B, 3, 384, 384]` | `src/model.py:54` |
| res_4 (after projection) | `[B, 256, 96, 96]` | `src/backbone.py:62` (assertion) |
| res_8 (after projection) | `[B, 256, 48, 48]` | `src/backbone.py:63` (assertion) |
| res_16 (after projection) | `[B, 256, 24, 24]` | `src/backbone.py:64` (assertion) |
| MoE routing_probs (per scale) | `[B, E, H_s, W_s]` | `src/moe_layer.py:148` |
| MoE topk_indices (per scale) | `[B, H_s*W_s, K]` | `src/moe_layer.py:111` |
| MoE topk_gates (per scale) | `[B, H_s*W_s, K]` | `src/moe_layer.py:112` |
| MoE entropy (per scale) | `[B, 1, H_s, W_s]` | `src/moe_layer.py:152` |
| Saliency logits | `[B, 1, 384, 384]` | `src/model.py:68` (assertion) |
| Boundary logits | `[B, 1, 384, 384]` | `src/model.py:69` (assertion) |
| Aux logits 16 | `[B, 1, 24, 24]` | `src/model.py:70` (assertion) |
| Aux logits 8 | `[B, 1, 48, 48]` | `src/model.py:71` (assertion) |
| Aux logits 4 | `[B, 1, 96, 96]` | `src/model.py:72` (assertion) |

### 1.3 Loss Functions

| Component | Formula | Default Weight | Source |
|-----------|---------|----------------|--------|
| BCE | `nn.BCEWithLogitsLoss()` on saliency_logits vs target | 1.0 | `src/loss.py:82,165` |
| Soft IoU | `1 - (Σ P·G + ε) / (Σ P + Σ G - Σ P·G + ε)`, per-image, averaged | 1.0 | `src/loss.py:86-93` |
| SSIM | Custom Gaussian window, window_size=11, `1 - SSIM(P, G)` | 0.0 (OFF) | `src/loss.py:41-57,167` |
| Boundary | `SmoothL1(‖∇P‖, gt_boundary)` | 0.0 (OFF) | `src/loss.py:95-107` |
| Load balance | `E × Σ_j (f_j × P_j)` averaged over scales | 0.01 | `src/loss.py:125-138` |
| Importance | `(std(P_j) / mean(P_j))²` averaged over scales | 0.01 | `src/loss.py:140-144` |
| Z-loss | `mean(logsumexp(clean_logits)²)` averaged over scales | 0.0 (OFF) | `src/loss.py:147-149` |
| Aux boundary | `BCEWithLogitsLoss(edge_logits, gt_boundary)` | 0.0 (OFF) | `src/loss.py:178-180` |
| Deep supervision | `mean(BCE(aux_16, target_16), BCE(aux_8, target_8), BCE(aux_4, target_4))` | 0.4 | `src/loss.py:183-195` |

### 1.4 Training Setup

| Claim | Evidence | Source |
|-------|----------|--------|
| DDP with NCCL, ≥2 GPUs required | `dist.init_process_group("nccl")` | `src/train_ddp.py:136` |
| `find_unused_parameters=True` | Required for sparse routing (some experts get 0 tokens) | `src/train_ddp.py:296` |
| AMP FP16 with GradScaler | `torch.amp.autocast(dtype=float16)` + `GradScaler` | `src/train_ddp.py:423` |
| Backbone frozen epoch 0 only | `freeze_backbone` / `unfreeze_backbone` | `src/train_ddp.py:382-388` |
| WarmupCosine scheduler | warmup_ratio=0.01, min_lr_ratio=0.01 | `src/optimization.py:84-102` |
| AdamW optimizer | weight_decay=1e-4 | `experiments/baseline_v1.json:36-39` |
| Gradient clipping max_norm=1.0 | `clip_grad_norm_` | `src/train_ddp.py:130` |
| Gradient accumulation with no_sync | `model.no_sync()` for non-final microsteps | `src/train_ddp.py:419-421` |
| Checkpoint selection by val MAE | `if val_mae < best_mae` | `src/train_ddp.py:589` |
| Deterministic seeding (rank-aware) | `seed = config.train.seed + rank` | `src/train_ddp.py:230` |

### 1.5 Dataset

| Claim | Evidence | Source |
|-------|----------|--------|
| WXSOD dataset, 3 splits | train_sys, test_sys, test_real | `src/dataset.py:31-41` |
| Scene-aware GroupShuffleSplit (80/20) | `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` | `src/dataset.py:260` |
| Strict train/val disjointness assertion | `assert train_scenes_set.isdisjoint(val_scenes_set)` | `src/dataset.py:269` |
| Filename pattern: `{scene_id}_{weather}.ext` | `get_weather_type(stem)` splits on `_` | `src/dataset.py:216-220` |
| Weather distribution verified | `verify_weather_distribution()` checks all weather categories present in val | `src/dataset.py:222-239` |
| Augmentations: HFlip(p=0.5) + ColorJitter(p=0.5) + Normalize | albumentations pipeline | `src/dataset.py:78-82` |
| Aspect-preserving resize to 384 + reflect/zero pad | `_aspect_preserving_resize_pad` | `src/dataset.py:107-143` |
| Deterministic per-sample augmentation seed | MD5-based seed = f(base_seed, epoch, index, name) | `src/dataset.py:171-179` |
| Boundary target: morphological dilate−erode with 5×5 ellipse | `BOUNDARY_KERNEL_SIZE=5, MORPH_ELLIPSE` | `src/boundary.py:4-5,13-29` |
| Image count: train_sys=12891, test_sys=1500, test_real=554 | Verified by file listing | File system |

### 1.6 Evaluation

| Claim | Evidence | Source |
|-------|----------|--------|
| Metrics: MAE, S_measure, E_adaptive, E_mean, E_max, F_adaptive, F_mean, F_max | py_sod_metrics library | `src/metrics.py:4-59` |
| Boundary metrics: boundary_MAE, boundary_F1 | Custom implementation | `src/metrics.py:61-114` |
| Geometry reversal: unpad + resize to original | `reverse_geometry()` | `src/evaluate.py:23-45` |
| Weather-wise breakdown by filename parsing | `get_weather_type(stem)` | `src/evaluate.py:80` |

---

## 2. Experimentally-Verified Claims

These claims are verified from `evaluation/` and `metrics.json`.
Cite the specific evaluation directory.

### 2.1 Results: best_new_1.pth on test_sys (1,500 images)

| Metric | Value | Source |
|--------|-------|--------|
| MAE | 0.0192 | `evaluation/best_new_1/test_sys/none/metrics.json` |
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
| MAE | 0.0168 | `evaluation/best_new_1/test_real/none/metrics.json` |
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

| Weather | N | MAE | S_measure | F_max |
|---------|---|-----|-----------|-------|
| snow | 90 | 0.0094 | 0.9478 | 0.9421 |
| fog | 126 | 0.0148 | 0.9179 | 0.8914 |
| rain | 120 | 0.0165 | 0.9157 | 0.8935 |
| dark | 125 | 0.0191 | 0.9072 | 0.8885 |
| light | 93 | 0.0243 | 0.8897 | 0.8629 |

Source: `evaluation/best_new_1/test_real/none/metrics.json`

### 2.4 Weather-Wise Breakdown (test_sys)

| Weather | N | MAE | S_measure | F_max |
|---------|---|-----|-----------|-------|
| clean | 167 | 0.0142 | 0.9329 | 0.9228 |
| fog | 166 | 0.0166 | 0.9308 | 0.9255 |
| snow | 172 | 0.0189 | 0.9192 | 0.9068 |
| rain | 179 | 0.0200 | 0.9173 | 0.9030 |
| dark | 156 | 0.0199 | 0.9049 | 0.8836 |
| light | 166 | 0.0215 | 0.9091 | 0.9046 |
| rainafog | 172 | 0.0237 | 0.8902 | 0.8691 |
| rainasnow | 166 | 0.0194 | 0.9152 | 0.9032 |
| snowafog | 156 | 0.0189 | 0.9051 | 0.8950 |

### 2.5 Key Experimental Observations

| Observation | Evidence |
|-------------|----------|
| Best real-world weather: snow (MAE=0.0094) | test_real metrics |
| Worst real-world weather: light (MAE=0.0243) | test_real metrics |
| Compound weather degrades most on synthetic: rainafog (MAE=0.0237) | test_sys metrics |
| Real-world MAE slightly better than synthetic: 0.0168 vs 0.0192 | global metrics comparison |

---

## 3. Forbidden Claims

Do not make these claims without explicit evidence:

- "Our method outperforms SOTA" — no comparison to other methods has been run
- "Expert X specializes in weather Y" — no interpretability analysis completed
- "Routing entropy correlates with difficulty" — no analysis done
- "Method works on [untested dataset]" — only WXSOD tested
- Any claim about computational efficiency (FLOPs, latency) — not measured
- Any claim about generalization to domains beyond WXSOD
- Any claim about the model's behavior on clean SOD benchmarks (DUTS, ECSSD, etc.)
- That ablation studies have been conducted — none have been run

---

## 4. Unverified / Incomplete Claims

| Claim | Status | What's Needed |
|-------|--------|---------------|
| Exact parameter count | Not verified | Run model parameter counting |
| Training epochs completed | Unknown | No training logs in repo |
| Comparison to SOTA methods | Not done | Need to run baselines |
| Ablation study results | Not run | Ablation code exists but no registry.csv |
| Router interpretability analysis | Not run | Diagnostics code exists but disabled in training |
| Boundary F1 comparison to SOTA | Not done | Need literature survey of boundary metrics |
| Whether `router_variant` or `moe_type` config fields are actually read by code | Unknown | Code inspection needed |
| Whether backbone_lr and new_module_lr differ in practice | Unknown | Both set to 1e-4 in config |

---

*Last updated: 2026-08-31. Based on full code audit of repository state.*
*This file must be updated when new experimental results are obtained or new implementation details are verified.*
