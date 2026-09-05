# PAPER_SOURCE_OF_TRUTH.md — Verified Scientific Claims

**Audit date:** 2026-09-02
**Evidence hierarchy:** Source code > configs > evaluation artifacts > research docs

---

## 1. Architecture (Source-Code Verified)

| # | Claim | Evidence | Source |
|---|-------|----------|--------|
| 1 | **Model:** `SpatialMoESODNet` — backbone → 3 independent SpatialMoE layers → cross-attention decoder → dual heads | Full forward pass | `src/model.py:6-48` |
| 2 | **Backbone:** PVTv2-B4 via `timm.create_model('pvt_v2_b4', pretrained=True, features_only=True, out_indices=(0,1,2))` | Model instantiation | `src/backbone.py:11-16` |
| 3 | **Three scales:** 1/4 (96×96), 1/8 (48×48), 1/16 (24×24) | `out_indices=(0,1,2)` + assertions | `src/backbone.py:15,62-64` |
| 4 | **C=256 projections:** Three 1×1 Conv2d from PVTv2 channels [64, 128, 320] → 256 | `nn.Conv2d(feature_channels[i], d, 1)` | `src/backbone.py:22-24` |
| 5 | **8 experts per scale:** 24 total (8 × 3 scales), no weight sharing | `num_experts=8`, separate ModuleLists | `experiments/baseline_v1.json:14`, `src/model.py:14-16` |
| 6 | **K=2 sparse routing:** Top-2 expert selection per token | `top_k=2` | `experiments/baseline_v1.json:15` |
| 7 | **Router:** DWConv3×3 (depthwise, groups=dim) + concat with token → MLP(2C→128→E) | Architecture definition | `src/moe_layer.py:50-57` |
| 8 | **Noisy gating:** Gaussian noise with input-dependent scale via softplus(noise_linear(x)), training only | `router_noise_enabled=True`, noise in training branch | `src/moe_layer.py:104-109` |
| 9 | **True sparse dispatch:** Python loop over experts, mask-based token selection, scatter-add | Explicit loop with `active_mask` | `src/moe_layer.py:119-141` |
| 10 | **Expert architecture:** LN→Linear(C,4C)→GELU→Linear(4C,C)→residual (4× expansion) | `TokenWiseMLPExpert` class | `src/moe_layer.py:14-33` |
| 11 | **Independent expert pool per scale:** Three separate `SpatialMoELayer` instances | `self.moe_4`, `self.moe_8`, `self.moe_16` | `src/model.py:14-16` |
| 12 | **Entropy:** $\mathcal{H}_i = -\sum_k g_{i,k} \log g_{i,k}$, normalized by $\log 2$ → [0,1] | Entropy computation + `math.log(2.0)` normalization | `src/moe_layer.py:151`, `src/decoder.py:23` |
| 13 | **Entropy→decoder fusion:** `Proj_Y(Y) + scale * Proj_H(norm_entropy)` with learnable scale (init=0.1) | `EntropyFusionBlock` class | `src/decoder.py:14-27` |
| 14 | **Decoder:** 3× EntropyFusionBlock → GlobalCrossAttn(1/16→1/8) → WindowedCrossAttn(1/8→1/4) → 3× RefinementBlock → dual heads | Full decoder forward pass | `src/decoder.py:219-280` |
| 15 | **Cross-attention direction:** Coarse→fine: Q=up(coarser), KV=finer | `q_x=F16_up, kv_x=F8_local` then `q_x=F8_up, kv_x=F4_ctx` | `src/decoder.py:253,259` |
| 16 | **RefinementBlocks:** 3× (Conv3×3→GN32→ReLU→Conv3×3→GN32→ReLU + residual) at 1/4, 1/2, 1 scales | `RefinementBlock` class, three instances | `src/decoder.py:205-217,235-237` |
| 17 | **Saliency/boundary heads:** 1×1 Conv2d(dim, 1) each | `self.saliency_head`, `self.edge_head` | `src/decoder.py:239-240` |
| 18 | **Active losses:** BCE (λ=1.0) + soft IoU (λ=1.0) + load-balance (λ=0.01) + importance (λ=0.01). SSIM, boundary, z-loss, aux-boundary all OFF (λ=0) | Loss weights in config | `experiments/baseline_v1.json:24-33`, `src/loss.py:198-206` |
| 19 | **Deep supervision:** OFF in config (`deep_supervision: false`). Code exists (3 aux heads at each scale) but unused. Deep supervision weight=0.4 in loss config but model doesn't produce aux outputs. | Config flag + code path | `experiments/baseline_v1.json:20`, `src/decoder.py:242-245` |

**CORRECTION to current paper:** The current paper states the loss has "four active terms" and does not mention deep supervision status. The paper correctly lists only BCE, IoU, LB, and IMP. However, the deep_supervision_weight=0.4 in the config is a red herring — the model flag is `false`, so no aux outputs are produced and the deep supervision loss receives None inputs (computed as 0.0 at `src/loss.py:184`). This is consistent.

---

## 2. Tensor Dimensions (Source-Code Verified)

| Tensor | Shape | Source |
|--------|-------|--------|
| Input | `[B, 3, 384, 384]` | `src/model.py:54` |
| res_4 (post-projection) | `[B, 256, 96, 96]` | `src/backbone.py:62` |
| res_8 (post-projection) | `[B, 256, 48, 48]` | `src/backbone.py:63` |
| res_16 (post-projection) | `[B, 256, 24, 24]` | `src/backbone.py:64` |
| MoE routing_probs (per scale) | `[B, E, H_s, W_s]` | `src/moe_layer.py:148` |
| MoE entropy (per scale) | `[B, 1, H_s, W_s]` | `src/moe_layer.py:152` |
| Saliency logits | `[B, 1, 384, 384]` | `src/model.py:68` |
| Boundary logits | `[B, 1, 384, 384]` | `src/model.py:69` |
| Aux logits (unused) | `[B, 1, 24/48/96, 24/48/96]` | `src/model.py:70-72` |

---

## 3. Training Configuration (Config + Code Verified)

| Parameter | Value | Source |
|-----------|-------|--------|
| DDP, NCCL | ≥2 GPUs required | `src/train_ddp.py:136,167` |
| AMP FP16 | `autocast(dtype=float16)` + GradScaler | `src/train_ddp.py:423,437` |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 | `experiments/baseline_v1.json:36-39` |
| Scheduler | WarmupCosine, warmup_ratio=0.01 | `experiments/baseline_v1.json:42` |
| Epochs | 50 | `experiments/baseline_v1.json:49` |
| Batch | 1/GPU × 2 GPUs × 16 accum = effective 32 | `experiments/baseline_v1.json:47-50` |
| Gradient clipping | max_norm=1.0 | `src/train_ddp.py:130` (via OptimizationEngine) |
| Backbone freeze | Epoch 0 only | `src/train_ddp.py:382-388` |
| Image size | 384×384 (aspect-preserving + pad) | `src/dataset.py:107-143` |
| Augmentation | HFlip(0.5) + ColorJitter(0.5) + ImageNet norm | `src/dataset.py:78-82` |
| find_unused_parameters | True (required for sparse routing) | `src/train_ddp.py:296` |
| Seed | 42 (rank-aware: seed+rank) | `experiments/baseline_v1.json:51`, `src/train_ddp.py:230` |
| Checkpoint selection | Best val MAE | `src/train_ddp.py:589` |

---

## 4. Dataset (Code + Filesystem Verified)

| Property | Value | Source |
|----------|-------|--------|
| Dataset | WXSOD | `src/dataset.py:19` |
| train_sys | 12,891 images | Filesystem count |
| test_sys | 1,500 images (9 weather categories) | Filesystem count |
| test_real | 554 images (5 weather categories) | Filesystem count |
| Split method | Scene-aware GroupShuffleSplit (80/20) | `src/dataset.py:260` |
| Boundary target | Morphological dilate−erode, 5×5 ellipse kernel | `src/boundary.py:4-5,13-29` |
| Filename pattern | `{scene_id}_{weather}.{ext}` | `src/dataset.py:216-220` |

**test_real weather categories:** dark (125), fog (126), light (93), rain (120), snow (90)
**test_sys weather categories:** clean (167), dark (156), fog (166), light (166), rain (179), rainafog (172), rainasnow (166), snow (172), snowafog (156)

---

## 5. Measured Results (Evaluation Artifact Verified)

### 5.1 Global Results

| Metric | test_sys (N=1500) | test_real (N=554) | Source |
|--------|-------------------|-------------------|--------|
| MAE ↓ | 0.0192 | 0.0168 | `evaluation/best_new_1/*/none/metrics.json` |
| S_measure ↑ | 0.9139 | 0.9151 | same |
| E_adaptive ↑ | 0.9591 | 0.9530 | same |
| E_mean ↑ | 0.9585 | 0.9551 | same |
| E_max ↑ | 0.9633 | 0.9608 | same |
| F_adaptive ↑ | 0.8822 | 0.8631 | same |
| F_mean ↑ | 0.8886 | 0.8747 | same |
| F_max ↑ | 0.9015 | 0.8936 | same |

### 5.2 Weather-Wise Results (test_real)

| Weather | N | MAE ↓ | S_measure ↑ | F_max ↑ | E_adaptive ↑ |
|---------|---|-------|-------------|---------|--------------|
| snow | 90 | 0.0094 | 0.9478 | 0.9421 | 0.9825 |
| fog | 126 | 0.0148 | 0.9179 | 0.8914 | 0.9560 |
| rain | 120 | 0.0165 | 0.9157 | 0.8935 | 0.9645 |
| dark | 125 | 0.0191 | 0.9072 | 0.8885 | 0.9477 |
| light | 93 | 0.0243 | 0.8897 | 0.8629 | 0.9129 |

### 5.3 Routing Entropy (Normalized by log 2)

| Scale | Synthetic | Real | Source |
|-------|-----------|------|--------|
| 1/4 | 0.6910 | 0.6912 | `evaluation_results/entropy_comparison.json` |
| 1/8 | 0.6931 | 0.6931 | same |
| 1/16 | 0.6800 | 0.6822 | same |

### 5.4 Forced-Expert Ablation (scale 1/4 → expert 0 only, test_real)

| Metric | Normal | Forced | Δ | Δ% |
|--------|--------|--------|---|-----|
| MAE ↓ | 0.0168 | 0.0170 | +0.0002 | +1.19% |
| S_measure ↑ | 0.9151 | 0.9139 | −0.0012 | −0.13% |
| F_mean ↑ | 0.8747 | 0.8713 | −0.0034 | −0.39% |
| F_max ↑ | 0.8936 | 0.8928 | −0.0008 | −0.09% |

Source: `evaluation_results/force_expert_ablation.json`

**Interpretation note:** Only 1 of 3 scales forced. Full disabling of all routers remains untested.

### 5.5 Computational Cost

| Metric | Value | Source |
|--------|-------|--------|
| Parameters | 66.27M | `evaluation_results/compute_cost.json` |
| MACs | 278.2G | same |
| FPS | 3.83 | same (hardware unspecified) |

---

## 6. Forbidden Claims (No Evidence)

The following claims MUST NOT appear in the paper:

| Claim | Why Forbidden |
|-------|---------------|
| "Outperforms SOTA" | No comparison to other methods run |
| "Experts specialize by weather" | No per-expert assignment analysis |
| "Routing entropy correlates with difficulty" | No correlation analysis done |
| "MoE improves over non-MoE" | No non-MoE baseline exists |
| "Entropy fusion improves decoder" | No ON/OFF ablation exists |
| "Independent per-scale routing is better" | No comparison to shared routing |
| "Model generalizes synthetic→real" | Observational only, not controlled |
| "Computational efficiency" | No baseline comparison |
| Any claim about expert specialization | Diagnostics disabled in training |
| Any clean-domain SOD results | Only WXSOD tested |

---

## 7. Scientific Corrections to Current Paper

| Issue | Current Paper | Correct Status |
|-------|---------------|----------------|
| Deep supervision implied active | Not explicitly stated, but config has `deep_supervision_weight: 0.4` | OFF — `model.deep_supervision: false`, no aux outputs produced |
| Window size | Paper says "7×7 windowed attention" (§3) | Config: `window_size: 7`. Correct. |
| Loss "four active terms" | Paper: "four active terms" | Correct: BCE, IoU, LB, IMP are active. Deep sup, SSIM, boundary, z-loss, aux-boundary are OFF. |
| Router MLP hidden dim | Paper: "hidden dimension 128" | Code: `router_hidden=128` (default param at `moe_layer.py:36`). Correct. |
| Table 1 boundary metrics | Not reported | Boundary MAE/F1 are measured but not in the main table. Consider adding or omitting consistently. |
| "log K" normalization | Paper: "normalized by log K" | Code: `math.log(2.0)` hard-coded for K=2. Correct for K=2, but not general K. |

---

## 8. Unverified / Incomplete Items

| Item | Status |
|------|--------|
| Training completed epochs | Unknown — no training logs in repo |
| Multiple seed runs | NOT DONE — single training run |
| Ablation studies | NOT RUN — system exists, zero registry entries |
| SOTA comparison | NOT RUN |
| Per-expert analysis | NOT RUN — diagnostics disabled |
| FPS hardware context | NOT SPECIFIED |
| Validation training curves | NOT AVAILABLE |

---

*This document is the authoritative reference for all scientific claims in the paper. Update only with new experimental evidence or source-code verification.*
