# BLUEPRINT_CODE_AUDIT.md — Comprehensive Audit

Full audit of the blueprint document against the actual repository implementation.

**Audit date:** 2026-08-31
**Blueprint file:** `spatial-moe-adverse-weather-sod-blueprint.md`
**Repository state:** HEAD (latest commit)

---

## 1. Executive Summary

The blueprint is a planning/hypothesis document. The implementation is a working codebase
with one trained model and evaluation results on two test sets. Key findings:

- The core architecture (PVTv2-B4, per-scale token-level MoE, cross-attention decoder) is
  **faithfully implemented**.
- Several blueprint-proposed components are **not implemented** (image-level routing ablation,
  Soft MoE comparison, domain-generalization cross-check).
- Several **implemented components are not in the blueprint** (SSIM loss, deep supervision,
  Z-loss, boundary loss, aspect-preserving padding, scene-aware splitting).
- One critical equation differs: the blueprint's edge loss (BCE on edge maps) vs. the
  implementation's boundary loss (SmoothL1 on gradient magnitudes).
- The entropy computation differs: implementation uses top-k gate entropy only; blueprint
  implies full entropy over all N experts.
- Only **one baseline experiment** has been run. No ablations exist.
- No literature verification has been performed.

---

## 2. Actual Architecture

**Entry point:** `src/model.py:6` → `SpatialMoESODNet`

```
Input [B, 3, 384, 384]
  → MultiScaleBackbone (PVTv2-B4 via timm, features_only, out_indices=(0,1,2))
      → 1×1 Conv2d projection per scale (channels: 64→256, 128→256, 320→256)
      → res_4: [B, 256, 96, 96]    (9,216 tokens)
      → res_8: [B, 256, 48, 48]    (2,304 tokens)
      → res_16: [B, 256, 24, 24]   (576 tokens)
  → Three independent SpatialMoELayer (one per scale, NOT shared)
      → Router: DWConv3x3 → cat(x, local) → MLP(512→128→E) + noisy top-k
      → 8 × TokenWiseMLPExpert (LN→Linear(256→1024)→GELU→Linear(1024→256) + residual)
      → True sparse dispatch (for-loop over experts, mask selection, weighted scatter-add)
  → SpatialMoEDecoder
      → 3 × EntropyFusionBlock (proj_y + scale * proj_entropy, normalized by log(8))
      → GlobalCrossAttentionBlock: 1/16→1/8 fusion
      → WindowedCrossAttentionBlock: 1/8→1/4 fusion (window_size from config)
      → 3 × RefinementBlock (Conv3×3→GN→ReLU→Conv3×3→GN→ReLU + residual)
      → Bilinear upsample: 1/4→1/2→1/1
      → Saliency head: Conv2d(256, 1, 1)
      → Boundary head: Conv2d(256, 1, 1)
      → 3 auxiliary heads (deep supervision, one per scale)
```

### Key source file references:

| Component | File | Lines |
|-----------|------|-------|
| Model entry point | `src/model.py` | 6-48 |
| Backbone | `src/backbone.py` | 5-39 |
| MoE layer (router + experts) | `src/moe_layer.py` | 35-165 |
| Expert architecture | `src/moe_layer.py` | 14-33 |
| Decoder | `src/decoder/` | 219-280 |
| Loss system | `src/loss.py` | 62-219 |
| Config system | `src/config.py` | 7-148 |
| Dataset | `src/dataset.py` | 19-293 |
| Optimization | `src/optimization.py` | 6-161 |
| Training loop | `src/train_ddp.py` | 117-704 |
| Evaluation | `src/evaluate.py` | 47-231 |
| Ablation generators | `src/ablations.py` | 1-157 |

---

## 3. Actual Training Setup

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | AdamW | `experiments/baseline_v1.json:36` |
| Learning rate (all params) | 1e-4 | `baseline_v1.json:37-38` |
| Weight decay | 1e-4 | `baseline_v1.json:39` |
| Warmup ratio | 0.01 | `baseline_v1.json:40` |
| Scheduler | WarmupCosine | `baseline_v1.json:41` |
| Min LR ratio | 0.01 (of base LR) | `src/optimization.py:88` |
| Gradient clipping | 1.0 | `baseline_v1.json:42` |
| AMP | FP16 | `baseline_v1.json:43` |
| Batch per GPU | 1 | `baseline_v1.json:44` |
| Gradient accumulation | 16 | `baseline_v1.json:45` |
| Effective global batch | 32 (2 GPUs × 1 × 16) | `baseline_v1.json:46` |
| Epochs | 50 | `baseline_v1.json:49` |
| Backbone freeze | epoch 0 only | `baseline_v1.json:50` |
| Seed | 42 | `baseline_v1.json:51` |
| DDP | NCCL, find_unused_parameters=True | `src/train_ddp.py:136,296` |
| Checkpoint selection | Lowest val MAE | `src/train_ddp.py:589` |

**Note:** The config defines separate `backbone_lr` and `new_module_lr` but both are set to
1e-4 in the baseline. The parameter group code (`src/optimization.py:6`) supports differential
LR but it is not used in the current config.

---

## 4. Actual Datasets

| Property | Value | Source |
|----------|-------|--------|
| Dataset | WXSOD | `src/dataset.py:31-41` |
| Train split | train_sys (12,891 images) | File count verification |
| Test synthetic | test_sys (1,500 images) | `results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/metrics.json` |
| Test real | test_real (554 images) | `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json` |
| Train/Val split | 80/20 scene-aware GroupShuffleSplit | `src/dataset.py:260` |
| Disjointness | Strict assertion: `train_scenes.isdisjoint(val_scenes)` | `src/dataset.py:269` |
| Weather categories (test_real) | dark, fog, light, rain, snow | `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json` |
| Weather categories (test_sys) | clean, dark, fog, light, rain, rainafog, rainasnow, snow, snowafog | `results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/metrics.json` |

**Augmentations (train):**
- HorizontalFlip(p=0.5)
- ColorJitter(p=0.5)
- Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
- Deterministic per-sample seed (MD5 of sample+epoch+index+name)
- Aspect-preserving resize to max 384, then reflect-pad (image) / zero-pad (mask)

**Source:** `src/dataset.py:78-86`

---

## 5. Actual Experiments

### 5.1 Existing Experiment: baseline_v1

| Property | Value |
|----------|-------|
| Config | `experiments/baseline_v1.json` |
| Checkpoints | `checkpoints/best.pth`, `checkpoints/best_new_1.pth` |
| Evaluation | `results/legacy/legacy_8expert/evaluation/best_new_1/` |
| Training epochs completed | UNKNOWN (no training logs in repo) |
| Training GPU count | UNKNOWN (requires ≥2 GPUs per code) |
| TensorBoard events | 14 files in `runs/spatial_moe_sod/` |

### 5.2 Existing Results (best_new_1.pth)

**test_sys (1,500 images):**

| Metric | Value |
|--------|-------|
| MAE | 0.0192 |
| S_measure | 0.9139 |
| E_adaptive | 0.9591 |
| E_mean | 0.9585 |
| E_max | 0.9633 |
| F_adaptive | 0.8822 |
| F_mean | 0.8886 |
| F_max | 0.9015 |
| boundary_MAE | 0.4817 |
| boundary_F1 | 0.5472 |

**test_real (554 images):**

| Metric | Value |
|--------|-------|
| MAE | 0.0168 |
| S_measure | 0.9151 |
| E_adaptive | 0.9530 |
| E_mean | 0.9551 |
| E_max | 0.9608 |
| F_adaptive | 0.8631 |
| F_mean | 0.8747 |
| F_max | 0.8936 |
| boundary_MAE | 0.6334 |
| boundary_F1 | 0.3601 |

**Weather-wise (test_real):**

| Weather | N | MAE | S_measure | F_max |
|---------|---|-----|-----------|-------|
| snow | 90 | 0.0094 | 0.9478 | 0.9421 |
| fog | 126 | 0.0148 | 0.9179 | 0.8914 |
| rain | 120 | 0.0165 | 0.9157 | 0.8935 |
| dark | 125 | 0.0191 | 0.9072 | 0.8885 |
| light | 93 | 0.0243 | 0.8897 | 0.8629 |

**Weather-wise (test_sys):**

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

### 5.3 No Ablations Exist

- `experiments/registry.csv` does NOT exist
- `src/ablations.py` defines ablation config generators but no runs have been executed
- No ablation results anywhere in the repository

---

## 6. Blueprint vs. Actual Implementation

| Blueprint Claim | Actual Implementation | Status | Evidence |
|----------------|----------------------|--------|----------|
| **PVTv2-B4 backbone** | PVTv2-B4 | MATCHES | `src/backbone.py:6,11` |
| **Swin-B alternative** | Not implemented | NOT IMPLEMENTED | — |
| **1/4, 1/8, 1/16 feature scales** | 1/4, 1/8, 1/16 | MATCHES | `src/backbone.py:15,62-64` |
| **Token-level router** | Token-level router | MATCHES | `src/moe_layer.py:62-112` |
| **DWConv3×3 router context** | DWConv3×3 router context | MATCHES | `src/moe_layer.py:50` |
| **2-layer MLP router** | 2-layer MLP (Linear→GELU→Linear) | MATCHES | `src/moe_layer.py:53-57` |
| **Router input = cat(x, local)** | cat(x_tokens, local_tokens) dim=-1 | MATCHES | `src/moe_layer.py:99` |
| **Noisy top-k (Shazeer)** | Gaussian noise + softplus scale | MATCHES | `src/moe_layer.py:104-107` |
| **k=2** | k=2 | MATCHES | `experiments/baseline_v1.json:15` |
| **N=6–8 experts** | N=8 | MATCHES | `experiments/baseline_v1.json:14` |
| **Independent experts per scale** | 3 separate SpatialMoELayer instances | MATCHES | `src/model.py:14-16` |
| **Expert = depthwise-separable + residual** | Expert = LN→4C→C residual MLP (NOT depthwise-separable) | DOES NOT MATCH | `src/moe_layer.py:14-33` |
| **Cross-attention fusion** | GlobalCrossAttention (1/16→1/8), WindowedCrossAttention (1/8→1/4) | MATCHES | `src/decoder/decoder.py` |
| **Fusion formula: Q=Y_s, K=V=Upsample(Y_{s+1})** | Q=F16_up, KV=F8_local (cross-attn, not residual add) | PARTIALLY MATCHES | `src/decoder/blocks.py` |
| **Router entropy as decoder input** | Yes, via EntropyFusionBlock | MATCHES | `src/decoder/blocks.py` (EntropyFusionBlock) |
| **Entropy: full over all N experts** | Entropy computed over top-k gates only | DOES NOT MATCH | `src/moe_layer.py:151` |
| **Decoder concatenates entropy** | Decoder adds entropy (learned projection + scale param) | DOES NOT MATCH | `src/decoder/blocks.py` |
| **BCE loss** | BCEWithLogitsLoss | MATCHES | `src/loss.py:82,165` |
| **IoU loss** | Soft IoU (per-image, averaged) | MATCHES | `src/loss.py:86-93` |
| **Edge/boundary loss** | Boundary loss = SmoothL1(grad_mag, gt_boundary), λ=0.0 (OFF) | PARTIALLY MATCHES | `src/loss.py:95-107` |
| **Edge loss formula (BCE on edge maps)** | SmoothL1 on gradient magnitudes (NOT BCE) | DOES NOT MATCH | `src/loss.py:95-107` |
| **Load-balance loss** | N × Σ(f_j × P_j) | MATCHES | `src/loss.py:125-138` |
| **Importance loss** | (std(P_j)/mean(P_j))² | MATCHES | `src/loss.py:140-144` |
| **SSIM loss** | Implemented, weight=0.0 (OFF) | NOT IN BLUEPRINT | `src/loss.py:41-57,167` |
| **Deep supervision** | 3 aux heads + BCE, weight=0.4 | NOT IN BLUEPRINT | `src/loss.py:183-195` |
| **Z-loss** | Implemented, weight=0.0 (OFF) | NOT IN BLUEPRINT | `src/loss.py:147-149` |
| **λ_iou=1, λ_edge=1** | λ_iou=1, λ_boundary=0.0 (OFF) | PARTIALLY MATCHES | `baseline_v1.json:27-28` |
| **λ_lb=0.01, λ_imp=0.01** | λ_lb=0.01, λ_imp=0.01 | MATCHES | `baseline_v1.json:29-30` |
| **Image-level routing ablation** | Code generates config, but NO run executed | NOT IMPLEMENTED | `src/ablations.py:134-138` |
| **Soft MoE comparison** | Not implemented | NOT IMPLEMENTED | — |
| **Multi-scale independent vs shared routing** | Not implemented | NOT IMPLEMENTED | — |
| **Domain-generalization cross-check** | Not implemented | NOT IMPLEMENTED | — |
| **t-SNE/UMAP analysis** | Not implemented | NOT IMPLEMENTED | — |
| **Per-expert heatmap visualization** | Code exists in diagnostics, disabled in training | NOT IMPLEMENTED | `src/diagnostics/:243-270` |
| **Routing consistency across severity** | Not implemented | NOT IMPLEMENTED | — |

---

## 7. Equation Audit

### Eq. 1: Backbone Feature Maps
Blueprint: $F_s = \text{Backbone}_s(I) \in \mathbb{R}^{H_s \times W_s \times C_s}$
Implementation: Features extracted at `out_indices=(0,1,2)` from PVTv2-B4, then projected via 1×1 Conv2d to dim=256.
**Status:** MATCHES IMPLEMENTATION
**Note:** Blueprint notation uses $H_s \times W_s \times C_s$ (channel-last); implementation uses channel-first `[B, C, H, W]`. This is a convention difference, not a mathematical discrepancy.

### Eq. 2: Token Flattening + Projection
Blueprint: $\tilde{X}_s = X_s W_s^{proj}$
Implementation: `x.flatten(2).transpose(1, 2)` after 1×1 Conv2d projection in backbone.
**Status:** MATCHES IMPLEMENTATION

### Eq. 3: Router
Blueprint: $g_i = W_2 \sigma(W_1 [x_i \| \text{DWConv}_{3\times3}(X_s)_i] + b_1) + b_2$
Implementation: `router_mlp(cat([x_tokens, local_tokens]))` where `local_tokens = router_dwconv(x)`.
**Status:** MATCHES IMPLEMENTATION

### Eq. 4: Noisy Routing
Blueprint: $g_i' = g_i + \epsilon_i, \epsilon_i \sim \mathcal{N}(0, \text{softplus}(W_{noise} x_i)^2)$
Implementation: `noise_std = router_noise_scale * softplus(noise_linear(x_tokens))`, `noise = randn_like(clean_logits) * noise_std`.
**Status:** MATCHES IMPLEMENTATION

### Eq. 5: Top-k Selection
Blueprint: Top-k of $g_i'$, softmax over selected experts.
Implementation: `torch.topk(noisy_logits, k=self.k, dim=-1)`, `F.softmax(topk_values, dim=-1)`.
**Status:** MATCHES IMPLEMENTATION

### Eq. 6: Gate-weighted Expert Output
Blueprint: $y_i = \sum_{j=1}^{N} p_i^{(j)} E_j(x_i)$
Implementation: Loop over experts, mask selection, scatter-add with gate weights.
**Status:** MATCHES IMPLEMENTATION (the sum is over active experts only, which is equivalent since $p_i^{(j)}=0$ for non-selected experts)

### Eq. 7: Expert Architecture
Blueprint: $E_j(x) = x + W_{out}^{(j)} \text{GELU}(\text{DWConv}^{(j)}(W_{in}^{(j)} x))$
Implementation: `x + Linear(GELU(Linear(LayerNorm(x))))` (MLP, NOT depthwise-separable conv).
**Status:** DOES NOT MATCH
**Note:** The blueprint proposes depthwise-separable conv experts; the implementation uses token-wise MLP experts. This is a deliberate architectural difference.

### Eq. 8: Load-Balancing Loss
Blueprint: $\mathcal{L}_{lb} = N \sum_{j=1}^{N} f_j \cdot P_j$
Implementation: `E * sum(f_j * P_j)` where `f_j = bincount(indices)/total` and `P_j = scatter_add(gates)/(B*N_tokens)`.
**Status:** MATCHES IMPLEMENTATION

### Eq. 9: Importance Loss
Blueprint: $\mathcal{L}_{imp} = (\text{std}(\text{Importance})/\text{mean}(\text{Importance}))^2$
Implementation: `mean_imp = P_j_sum.mean()`, `std_imp = P_j_sum.std(unbiased=False)`, `l_imp = (std_imp/(mean_imp+1e-6))**2`.
**Status:** MATCHES IMPLEMENTATION

### Eq. 10: Edge/Boundary Loss (Blueprint)
Blueprint: $\mathcal{L}_{edge} = -\frac{1}{HW}\sum_{u,v}[G^{edge}_{uv}\log S^{edge}_{uv} + (1-G^{edge}_{uv})\log(1-S^{edge}_{uv})]$
Implementation: `SmoothL1(sqrt(gx²+gy²+eps), gt_boundary)` (SmoothL1, NOT BCE; gradient magnitudes, NOT edge probability).
**Status:** DOES NOT MATCH
**Note:** The blueprint proposes BCE on binary edge maps. The implementation uses SmoothL1 on continuous gradient magnitudes. Both are disabled in the baseline config (λ=0.0).

### Eq. 11: Routing Entropy
Blueprint: $H_i = -\sum_j p_i^{(j)} \log p_i^{(j)}$ (sum over all N experts)
Implementation: `entropy = -sum(topk_gates * log(topk_gates + 1e-9), dim=-1)` (sum over top-k gates only).
**Status:** DOES NOT MATCH
**Note:** The blueprint implies entropy over all N experts' routing probabilities. The implementation computes entropy over the K top-k gate values only. Since non-selected experts have $p=0$, the full entropy would equal the top-k entropy (by convention $0 \log 0 = 0$). However, the implementation uses the *gate weights* (post-softmax over top-k) not the *full routing probabilities* (which would include zeros for non-selected experts). These are mathematically equivalent when $p=0$ for non-selected, but the implementation explicitly works only with the K-dimensional gate vector.

### Eq. 12: Decoder Fusion
Blueprint: $\hat{Y}_8 = Y_8 + \text{CrossAttn}(Q=Y_8, K=V=\text{Upsample}(\hat{Y}_{16}))$
Implementation: `F8 = cross_attn_16_to_8(q_x=F16_up, kv_x=F8_local)` where `F16_up = interpolate(F16, F8_local.shape)`.
**Status:** PARTIALLY MATCHES
**Note:** The blueprint specifies Q from the local feature, K/V from the upsampled coarse feature. The implementation reverses this: Q comes from the upsampled coarse feature (F16_up), K/V from the local feature (F8_local). This is a meaningful difference in cross-attention direction.

### Eq. 13: Entropy as Decoder Input
Blueprint: $S = \text{DecoderHead}([\hat{Y}_4 \| H_4, \hat{Y}_8 \| H_8, \hat{Y}_{16} \| H_{16}])$ (concatenation)
Implementation: `y_proj + scale * e_proj` where `y_proj = proj_y(Y)` and `e_proj = proj_entropy(entropy_norm)` (addition with learned scale).
**Status:** DOES NOT MATCH
**Note:** Blueprint says concatenate entropy as extra channel. Implementation uses learned additive fusion with a scalar scale parameter (initialized to 0.1).

---

## 8. Existing Ablation Audit

**No ablations have been executed.** The following ablation infrastructure exists but has produced zero results:

| Ablation | Code Location | Status |
|----------|--------------|--------|
| Architecture matrix (backbone, expert count, top-k) | `src/ablations.py:20-68` | Code exists, no results |
| MoE ladder (none/dense/sparse) | `src/ablations.py:70-91` | Code exists, no results |
| Loss matrix (BCE+IoU, +SSIM, +Boundary) | `src/ablations.py:93-117` | Code exists, no results |
| Router matrix (token_only, token_local, global) | `src/ablations.py:119-140` | Code exists, no results |
| Registry/export | `src/ablations.py:142-157` | No registry.csv exists |
| Counterfactual expert ablation | `src/train_ddp.py:624-654` | Code exists, not invoked |

---

## 9. Proposed Experiments (from blueprint)

| # | Experiment | Hypothesis | Status | Feasibility |
|---|-----------|------------|--------|-------------|
| 1 | Image-level vs token-level routing | Spatial granularity necessary for SOD | Code exists (`ablations.py:134`), not run | Feasible |
| 2 | Multi-scale independent vs shared routing | Independent routing needed per receptive field | Not implemented | Feasible (requires model changes) |
| 3 | MoE vs matched-FLOPS dense baseline | MoE earns its complexity | Not implemented | Feasible |
| 4 | Top-k and expert count sweep | k=2, N=8 justified | Code exists (`ablations.py:20-68`), not run | Feasible |
| 5 | Auxiliary loss ablation | Each loss prevents specific failure | Code exists (`ablations.py:93-117`), not run | Feasible |
| 6 | Domain-generalization cross-check | Generalization to unseen real weather | Not implemented | Requires new training pipeline |

---

## 10. Incorrect Blueprint Assumptions

1. **Expert architecture:** Blueprint proposes depthwise-separable conv experts. Implementation uses token-wise MLP experts (LN→4C→C). This is a significant structural difference.

2. **Edge loss formula:** Blueprint proposes BCE on binary edge maps. Implementation uses SmoothL1 on gradient magnitudes. Both are disabled in baseline (λ=0.0).

3. **Entropy computation:** Blueprint implies full entropy over all N routing probabilities. Implementation computes entropy over top-k gate values only. Mathematically equivalent but different code path.

4. **Entropy fusion method:** Blueprint says concatenate entropy as extra channel. Implementation uses learned additive fusion (proj + scale×proj_entropy).

5. **Cross-attention direction:** Blueprint: Q from local, K/V from upsampled coarse. Implementation: Q from upsampled coarse, K/V from local.

6. **Decoder topology:** Blueprint describes a standard progressive-upsampling head ("BASNet/U²-Net-style"). Implementation uses cross-attention blocks + RefinementBlocks + bilinear upsampling (structurally different from BASNet/U²-Net).

7. **SSIM loss not in blueprint but implemented:** Blueprint loss equation omits SSIM. Implementation has SSIM loss code (disabled in baseline).

8. **Deep supervision not in blueprint but implemented:** Blueprint has no deep supervision. Implementation has 3 auxiliary heads + BCE loss with λ=0.4.

9. **Number of loss components:** Blueprint equation has 5 terms. Implementation has up to 9 loss terms (BCE, IoU, SSIM, boundary, load-balance, importance, Z-loss, aux-boundary, deep-supervision).

---

## 11. Unknowns Requiring Manual Verification

| Item | What to verify | Why it matters |
|------|---------------|----------------|
| Training completion | Did training complete all 50 epochs? | Determines if results are final |
| Checkpoint identity | Which checkpoint is `best.pth` vs `best_new_1.pth`? | Results may differ |
| Number of GPUs used | 2 vs more? | Affects effective batch size |
| TensorBoard contents | What training curves exist? | Validates training stability |
| backbone_lr vs new_module_lr | Are they actually the same (1e-4) in the trained model? | Config says both are 1e-4 but parameter groups code supports differential LR |
| `router_variant` config field | Config has `router_variant: "token_only"` but code always uses DWConv3x3. Is this field actually read anywhere? | Determines if ablation configs change anything |
| `moe_type` config field | Config has `moe_type: "sparse"` but code always runs sparse routing. Is "none" or "dense" handled? | Determines if MoE ladder ablation is feasible |
| SSIM loss weight | Config says 0.0. Was it ever non-zero during training? | Affects loss landscape |
| best.pth vs best_new_1.pth | Are they different checkpoints? Different training runs? | Determines reproducibility |

---

*Generated by full code audit. All claims reference source code files and line numbers.*
