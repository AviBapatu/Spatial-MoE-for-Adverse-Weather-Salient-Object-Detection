# PAPER_SOURCE_OF_TRUTH.md

**Purpose:** Authoritative fact base for the paper manuscript. Every claim in the paper must trace to a source below. No exceptions.

**Audit date:** 2026-09-02

---

## 1. Architecture Facts

All claims verified by reading source code. Line numbers reference the current codebase.

### 1.1 End-to-End Pipeline

| Stage | Operation | Output Shape | Source |
|-------|-----------|-------------|--------|
| Input | RGB image | `[B, 3, 384, 384]` | `src/model.py:54` |
| Backbone | PVTv2-B4 (timm, `out_indices=(0,1,2)`) | 3 feature maps | `src/backbone.py:11-16` |
| Projection | 1×1 conv per scale → `d=256` | `[B, 256, 96, 96]`, `[B, 256, 48, 48]`, `[B, 256, 24, 24]` | `src/backbone.py:22-24`, assertions at `:62-64` |
| MoE layers | 3 independent `SpatialMoELayer` | same shapes + entropy `[B, 1, H, W]` | `src/model.py:14-16` |
| Decoder | Cross-attention fusion → refinement → heads | `[B, 1, 384, 384]` × 2 (saliency + boundary) | `src/decoder.py:219-280` |

### 1.2 Backbone

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | `pvt_v2_b4` (timm) | `src/backbone.py:11` |
| Pretrained | True | `src/backbone.py:12` |
| Raw channel dims | `[64, 128, 320]` | `src/backbone.py:19` (via `feature_info.channels()`) |
| Output indices | `(0, 1, 2)` → 1/4, 1/8, 1/16 scales | `src/backbone.py:15` |

### 1.3 Spatial MoE Layer

**Instantiation:** 3 independent instances at `src/model.py:14-16`.

| Component | Architecture | Source |
|-----------|-------------|--------|
| **Router: DWConv** | `Conv2d(256, 256, 3, padding=1, groups=256)` (depthwise) | `src/moe_layer.py:50` |
| **Router: MLP** | `Linear(512, 128)` → GELU → `Linear(128, 8)` | `src/moe_layer.py:53-57` |
| **Router input** | `cat([x_tokens, dwconv_tokens], dim=-1)` → `[B, H*W, 512]` | `src/moe_layer.py:99` |
| **Noise** | Gaussian noise scaled by `softplus(noise_linear(x))`; training only | `src/moe_layer.py:104-108` |
| **Selection** | `topk(noisy_logits, k=2)` | `src/moe_layer.py:111` |
| **Gate weights** | `softmax(topk_values)` (over selected experts only) | `src/moe_layer.py:112` |
| **Dispatch** | True sparse: Python for-loop over 8 experts, mask-based selection | `src/moe_layer.py:121-140` |
| **Output** | Gate-weighted sum of expert outputs per token | `src/moe_layer.py:137-140` |

### 1.4 Expert Architecture

**Class:** `TokenWiseMLPExpert` at `src/moe_layer.py:14-33`.

```
Input x [N_active, 256]
  → LayerNorm(256)
  → Linear(256, 1024)
  → GELU
  → Linear(1024, 256)
  → Add residual: x + z
```

- Expansion factor: 4× (1024/256)
- Residual connection: `return x + z`
- Each expert is independent (no weight sharing)

### 1.5 Number of Experts

| Config field | Value | Source |
|-------------|-------|--------|
| `model.num_experts` | 8 | `experiments/baseline_v1.json:14` |
| Per scale | 8 | `src/model.py:14-16` passes to `SpatialMoELayer` |
| Total (3 scales) | 24 | 3 × 8 |

### 1.6 Top-k Routing

| Parameter | Value | Source |
|-----------|-------|--------|
| `model.top_k` | 2 | `experiments/baseline_v1.json:15` |
| `SpatialMoELayer.k` | 2 (from config) | `src/model.py:14-16` |
| Softmax scope | Over top-k selected values only (not all experts) | `src/moe_layer.py:112` |

### 1.7 Routing Entropy

**Computation** at `src/moe_layer.py:151`:
```python
entropy = -torch.sum(topk_gates * torch.log(topk_gates + 1e-9), dim=-1)
```
- Computed over K=2 selected gates only.
- Output shape: `[B, H*W]`, reshaped to `[B, 1, H, W]`.
- Maximum possible entropy (K=2, uniform): `log(2)` ≈ 0.693.

**Normalization** at `src/decoder.py:23`:
```python
entropy_norm = entropy / math.log(2.0 + 1e-8)
```
- Scales to [0, 1] range assuming K=2.

### 1.8 Entropy Enters the Decoder

**Module:** `EntropyFusionBlock` at `src/decoder.py:14-27`.

| Parameter | Value | Source |
|-----------|-------|--------|
| `proj_y` | `Conv2d(256, 256, 1)` or Identity | `src/decoder.py:17` |
| `proj_entropy` | `Conv2d(1, 256, 1)` | `src/decoder.py:18` |
| `scale` | Learnable scalar, init=0.1 | `src/decoder.py:19` |
| Formula | `proj_y(Y) + scale * proj_entropy(entropy_norm)` | `src/decoder.py:27` |

Applied independently to each scale's MoE output before cross-attention fusion.

### 1.9 Decoder Architecture

**Class:** `SpatialMoEDecoder` at `src/decoder.py:219-280`.

| Step | Operation | Source |
|------|-----------|--------|
| 1 | EntropyFusion at each scale (16, 8, 4) | `src/decoder.py:248-250` |
| 2 | 1/16 → 1/8: bilinear upsample → GlobalCrossAttentionBlock (q=F16_up, kv=F8) | `src/decoder.py:252-253` |
| 3 | Global context: AdaptiveAvgPool2d(1) on F8 → 1×1 conv → add to F4 | `src/decoder.py:255-256` |
| 4 | 1/8 → 1/4: bilinear upsample → WindowedCrossAttentionBlock (q=F8_up, kv=F4_ctx) | `src/decoder.py:258-259` |
| 5 | RefineBlock at 1/4 | `src/decoder.py:261` |
| 6 | 2× upsample → RefineBlock at 1/2 | `src/decoder.py:262-263` |
| 7 | 2× upsample → RefineBlock at 1/1 | `src/decoder.py:264-265` |
| 8 | 1×1 conv → saliency logits `[B, 1, 384, 384]` | `src/decoder.py:267` |
| 9 | 1×1 conv → boundary logits `[B, 1, 384, 384]` | `src/decoder.py:268` |

**Cross-attention direction:**
- `GlobalCrossAttentionBlock`: Q from coarser scale, KV from finer scale. Coarsest (1/16) queries finer (1/8).
- `WindowedCrossAttentionBlock`: Same direction. Intermediate (1/8) queries finest (1/4).

**Windowed attention:** Default `window_size=7` (from config `experiments/baseline_v1.json:18`), 8 attention heads, learned relative position bias.

**RefinementBlock:** Conv3×3 → GroupNorm(32) → ReLU → Conv3×3 → GroupNorm(32) → ReLU → residual add.

### 1.10 Deep Supervision

| Config field | Value | Source |
|-------------|-------|--------|
| `model.deep_supervision` | `false` | `experiments/baseline_v1.json:21` |
| Loss weight `lambda_deep_supervision` | 0.4 | `experiments/baseline_v1.json:33` |

**Code path:**
- Model always creates aux heads (lines `decoder.py:242-245`) regardless of config.
- Decoder always produces `aux_logits_16/8/4` (lines `decoder.py:270-272`) when `use_deep_supervision=True` (default in `SpatialMoESODNet`).
- **However**, the config passes `use_deep_supervision=config.model.deep_supervision` which is `false`.
- In `SpatialMoESODNet.__init__` at `model.py:7`: `use_deep_supervision=True` is the Python default, but `train_ddp.py:33-34` passes `config.model.deep_supervision`.
- Loss code at `loss.py:184` checks: `if aux_logits_16 is not None and ... and self.lambda_deep_supervision > 0`.

**Resolution:** Since `config.model.deep_supervision=false`, the decoder constructor receives `False`, and aux heads are NOT created. The aux logits will be `None`, and the deep supervision loss term is zero. **Deep supervision is OFF in the baseline.**

---

## 2. Loss Function

**Class:** `SpatialMoELoss` at `src/loss.py:62-219`.

### 2.1 Active Loss Terms (Baseline)

| Term | Formula | Weight | Active? | Source |
|------|---------|--------|---------|--------|
| BCE | `BCEWithLogitsLoss(logits, target)` | 1.0 | **YES** | `loss.py:82,165` |
| Soft IoU | `1 − mean((intersection+ε)/(union+ε))` per-image | 1.0 | **YES** | `loss.py:86-93,166` |
| SSIM | `1 − SSIM(sigmoid(logits), target)`, Gaussian w=11 | 0.0 | NO | `loss.py:9-57,167` |
| Boundary | `SmoothL1(∇mag(pred), gt_boundary)` | 0.0 | NO | `loss.py:95-107,170-172` |
| Load balance | `E × Σ_j (f_j × P_j)` (Shazeer form) | 0.01 | **YES** | `loss.py:125-138` |
| Importance | `(std(P_j) / mean(P_j))²` | 0.01 | **YES** | `loss.py:140-144` |
| Z-loss | `mean(logsumexp(logits)²)` | 0.0 | NO | `loss.py:147-149` |
| Aux boundary | `BCEWithLogitsLoss(edge_logits, gt_boundary)` | 0.0 | NO | `loss.py:178-180` |
| Deep supervision | `BCE(aux_head_i, target_i)` averaged over 3 scales | 0.4 | NO (config=false) | `loss.py:183-195` |

**Total loss (active terms only):**
```
L = L_bce + 1.0 × L_iou + 0.01 × L_lb + 0.01 × L_imp
```

### 2.2 Load Balancing Loss Detail

At `src/loss.py:125-138`:
```python
f_j = bincount(flat_indices, minlength=E) / total_assignments   # detached hard fraction
P_j = scatter_add(flat_gates) / (B × N_tokens)                  # differentiable soft mass
l_lb = E × Σ(f_j × P_j)                                         # product form
```
- Averaged across 3 scales (line `loss.py:151-152`).

---

## 3. Training Protocol

| Parameter | Value | Source |
|-----------|-------|--------|
| Framework | PyTorch DDP, NCCL backend | `src/train_ddp.py:136` |
| GPUs | 2 (minimum) | `src/train_ddp.py:136` |
| `find_unused_parameters` | True (required for sparse routing) | `src/train_ddp.py:296` |
| Optimizer | AdamW | `experiments/baseline_v1.json:36` |
| Learning rate (backbone) | 1e-4 | `experiments/baseline_v1.json:37` |
| Learning rate (new modules) | 1e-4 | `experiments/baseline_v1.json:38` |
| Weight decay | 1e-4 | `experiments/baseline_v1.json:39` |
| Scheduler | WarmupCosine (warmup 1%, decay to 0.01×base) | `experiments/baseline_v1.json:42` |
| Epochs | 50 | `experiments/baseline_v1.json:44` |
| Batch per GPU | 1 | `experiments/baseline_v1.json:49` |
| Gradient accumulation | 16 steps | `experiments/baseline_v1.json:50` |
| Effective global batch | 32 (1 × 2 GPUs × 16 accum) | `experiments/baseline_v1.json:51` |
| AMP | FP16, GradScaler | `experiments/baseline_v1.json:43`, `train_ddp.py:423` |
| Gradient clipping | max_norm = 1.0 | `experiments/baseline_v1.json:41` |
| Backbone freeze | Epoch 0 only, unfreezed at epoch 1 | `train_ddp.py:382-388` |
| Seed | 42 | `experiments/baseline_v1.json:44` |
| Parameter groups | backbone/MoE/decoder/heads, each with decay/no_decay split | `src/optimization.py:6-72` |
| No-decay | biases, LayerNorm params, relative_position_bias_table | `src/optimization.py:56` |

---

## 4. Dataset Facts

### 4.1 WXSOD Dataset

| Property | Value | Source |
|----------|-------|--------|
| Total images | 14,945 | `spatial-moe-adverse-weather-sod-blueprint.md:10-11`, DB-34 |
| Training split | `train_sys`: 12,891 images | `docs/research/DATASETS.md:26` |
| Synthetic test | `test_sys`: 1,500 images | `results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/summary.txt:2` |
| Real-world test | `test_real`: 554 images | `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/summary.txt:2` |
| Weather labels | Per-image weather type in filename: `{scene_id}_{weather}.{ext}` | `src/dataset.py:216-220` |

### 4.2 Weather Categories

**test_sys (9 categories):** clean, fog, rain, snow, dark, light, rainafog, rainasnow, snowafog

**test_real (5 categories):** fog, rain, snow, dark, light

### 4.3 Split Protocol

- Scene-aware `GroupShuffleSplit` on `train_sys` → 80% train, 20% val, grouped by scene ID.
- Strict train/val disjointness assertion (`src/dataset.py:269`).
- No cross-contamination: test splits are pre-defined and held out.

### 4.4 Preprocessing

| Step | Operation | Source |
|------|-----------|--------|
| Resize | Aspect-preserving resize to max_dim=384 | `src/dataset.py:107-143` |
| Pad | To 384×384: image=`BORDER_REFLECT_101`, mask=`BORDER_CONSTANT(0)` | `src/dataset.py:135-143` |
| Normalize | ImageNet mean/std: `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` | `src/dataset.py:81,85` |

### 4.5 Augmentation (Train Only)

| Transform | p | Source |
|-----------|---|--------|
| HorizontalFlip | 0.5 | `src/dataset.py:79` |
| ColorJitter | 0.5 | `src/dataset.py:80` |
| Normalize | 1.0 | `src/dataset.py:81` |

Deterministic per-sample augmentation seeding via MD5 hash (`src/dataset.py:171-179`).

### 4.6 Boundary Computation

- 5×5 elliptical morphological kernel.
- `boundary = dilate(mask) − erode(mask)` (band, not single-pixel edge).
- Source: `src/boundary.py` via `src/dataset.py:77`.

---

## 5. Result Facts

### 5.1 Canonical Results: best_new_1.pth

**Source:** `results/legacy/legacy_8expert/evaluation/best_new_1/{test_sys,test_real}/none/`

#### test_sys (1,500 images)

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

#### test_real (554 images)

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

### 5.2 Weather-Wise Breakdown: test_real

| Weather | N | MAE | S_measure | F_max |
|---------|---|-----|-----------|-------|
| snow | 90 | 0.0094 | 0.9478 | 0.9421 |
| fog | 126 | 0.0148 | 0.9179 | 0.8914 |
| rain | 120 | 0.0165 | 0.9157 | 0.8935 |
| dark | 125 | 0.0191 | 0.9072 | 0.8885 |
| light | 93 | 0.0243 | 0.8897 | 0.8629 |

### 5.3 Weather-Wise Breakdown: test_sys (selected)

| Weather | N | MAE | S_measure |
|---------|---|-----|-----------|
| clean | 167 | 0.0142 | 0.9329 |
| fog | 166 | 0.0166 | 0.9308 |
| snow | 172 | 0.0189 | 0.9192 |
| rain | 179 | 0.0200 | 0.9173 |
| dark | 156 | 0.0199 | 0.9049 |
| light | 166 | 0.0215 | 0.9091 |
| rainafog | 172 | 0.0237 | 0.8902 |
| rainasnow | 166 | 0.0194 | 0.9152 |
| snowafog | 156 | 0.0189 | 0.9051 |

### 5.4 Forced-Expert Ablation

**Source:** `results/legacy/legacy_8expert/evaluation_results/force_expert_ablation.json`

All tokens at scale 1/4 forced to expert 0 (router bypassed). Scales 1/8 and 1/16 still routed normally.

| Metric | Normal | Forced (E0, scale 4) | Δ | Δ% |
|--------|--------|---------------------|---|-----|
| MAE | 0.0168 | 0.0170 | +0.0002 | +1.2% |
| S_measure | 0.9151 | 0.9139 | −0.0012 | −0.1% |
| F_mean | 0.8747 | 0.8713 | −0.0034 | −0.4% |

**Interpretation:** Marginal degradation. The forced-expert test affects only 1 of 3 scales. Cannot determine whether routing is critical without disabling all routing.

### 5.5 Routing Entropy

**Source:** `results/legacy/legacy_8expert/evaluation_results/entropy_comparison.json`

| Split | Scale 1/4 | Scale 1/8 | Scale 1/16 |
|-------|-----------|-----------|------------|
| Synthetic | 0.6910 | 0.9631 | 0.6800 |
| Real | 0.6912 | 0.6931 | 0.6822 |

All values normalized by log(2). Maximum possible = 1.0 (uniform over 2 experts).

### 5.6 Computational Cost

**Source:** `results/legacy/legacy_8expert/evaluation_results/compute_cost.json`

| Metric | Value |
|--------|-------|
| Parameters | 66.27M |
| MACs | 278.2G |
| FPS | 3.83 (hardware unspecified) |

---

## 6. Verified Literature Facts

### 6.1 Prior Art: MoE for Weather Restoration

| Paper | Routing | Weather labels at test | SOD | Source |
|-------|---------|----------------------|-----|--------|
| WM-MoE (2023, arXiv:2303.13739) | Token-level (per-patch) | No (learned weather branch via WGF-CL contrastive learning) | No | DB-19, LITERATURE_CLAIMS CLAIM-05,06 |
| MoFME (AAAI 2024) | Token-level (MC dropout uncertainty) | No | No | DB-20 |
| Complexity Experts (CVPR 2025) | Image-level (noisy top-1) | No | No | DB-29 |

**CORRECTION:** WM-MoE and MoWE are the same paper (arXiv:2303.13739). WM-MoE routes at TOKEN-LEVEL (not feature-map level). WM-MoE does NOT require weather labels at test time.

### 6.2 Prior Art: MoE for SOD

| Paper | Routing granularity | Task | Source |
|-------|-------------------|------|--------|
| MMSOD (IJCNN 2025) | Modality-level (RGB-D) | Multimodal SOD | DB-01 |
| CMFNet (IEEE TMM 2026) | Scale-level | RGB-D SOD | DB-03 |
| CMoE (AAAI 2026) | Modality-level | Modality-missing SOD | DB-04 |
| PSOD (IEEE TIP 2025) | Task-level | Pluralistic SOD | DB-06 |

**Verified:** No existing MoE-SOD paper performs spatial per-token routing for single-RGB-modality SOD. (LITERATURE_CLAIMS CLAIM-02)

### 6.3 Prior Art: Adverse-Weather SOD

| Paper | Approach | Weather labels | Source |
|-------|----------|---------------|--------|
| NIFM (2025, arXiv:2512.10592) | One-hot weather-type encoding | Yes (explicit) | DB-35 |
| WFANet (2025) | Two-branch (weather + saliency) | Yes (explicit) | DB-34 |
| WXSOD benchmark (2025/2026) | 14,945 images, 9 weather categories | Per-image labels in filenames | DB-34 |

### 6.4 Prior Art: Token-Level MoE

| Paper | Routing | Task | Source |
|-------|---------|------|--------|
| V-MoE (NeurIPS 2021) | Token-level top-k with noisy gating | Classification | DB-41 |
| Soft MoE (ICLR 2024) | Differentiable weighted combination (no top-k) | Classification | DB-42 |
| M³ViT (NeurIPS 2022) | Task-conditioned token routing | Multi-task (class+det+seg) | DB-43 |

### 6.5 Routing Entropy as Decoder Input

**Verified claim:** No prior work feeds routing entropy as an explicit spatial feature channel into a dense prediction decoder. Prior work uses routing entropy for OOD detection (DB-49), K-selection (DB-50, DB-51), and routing control (DB-52). (LITERATURE_CLAIMS CLAIM-21)

---

## 7. Defensible Novelty Claims

These claims are supported by the combination of implementation evidence and literature analysis.

### 7.1 PRIMARY: Token-Level Spatial MoE Routing for Dense SOD

- **Implementation:** True sparse dispatch of individual spatial tokens to experts at 3 pyramid scales (`moe_layer.py:119-141`).
- **Literature gap:** Existing MoE-SOD papers (MMSOD, CMFNet, CMoE, PSOD) route at modality, scale, or task level—not spatial token level for single-RGB SOD. WM-MoE does token-level routing but targets image restoration, not SOD.
- **Caveat:** No ablation comparing routed vs non-routed model exists.

### 7.2 PRIMARY: Independent Per-Scale Routers with Separate Expert Pools

- **Implementation:** 3 completely independent `SpatialMoELayer` instances, each with its own router MLP, expert pool, and noise parameters (`model.py:14-16`).
- **Literature gap:** WM-MoE uses a single router across scales. M³ViT uses task-conditioned routing, not independent per-scale routing.
- **Caveat:** No ablation comparing independent vs shared routing exists.

### 7.3 PRIMARY: Routing Entropy as Decoder Input Feature

- **Implementation:** `EntropyFusionBlock` fuses per-token routing entropy as an explicit channel in the decoder (`decoder.py:14-27`).
- **Literature gap:** No prior work feeds routing entropy as a spatial feature map into a dense prediction decoder.
- **Caveat:** No ablation with/without entropy fusion exists.

### 7.4 SECONDARY: Fully Label-Free Architecture

- **Implementation:** Router operates purely on spatial content features (DWConv3x3 + MLP). No weather branch, no weather labels, no CLIP features at any stage.
- **Literature gap:** WM-MoE has a dedicated weather feature branch (WGF-CL contrastive learning). NIFM and WFANet require explicit weather labels.
- **Caveat:** WM-MoE also routes without weather labels at test time via its learned weather branch. The distinction is narrower than "label-free"—we have no weather-specific architectural component at all.

### 7.5 SECONDARY: MoE Applied to Adverse-Weather SOD

- **Intersection:** No existing paper applies MoE specifically to SOD under adverse weather conditions.
- **Caveat:** Cannot claim MoE causes improvement without a non-MoE baseline.

---

## 8. Forbidden / Unsupported Claims

Do NOT make these claims in the paper.

| Claim | Reason | Source |
|-------|--------|--------|
| "Our method outperforms SOTA" | No comparison to other methods exists | FINAL_EXPERIMENTAL_RESULTS §7 |
| "Experts specialize by weather" | No per-expert assignment analysis exists; forced-expert test shows minimal degradation | FINAL_CLAIM_EVIDENCE_MATRIX §F |
| "Routing entropy correlates with difficulty" | No entropy-boundary analysis done | FINAL_CLAIM_EVIDENCE_MATRIX §D |
| "Token-level routing causes improvement" | No non-routing baseline exists | FINAL_CLAIM_EVIDENCE_MATRIX §A |
| "MoE causes improvement" | No non-MoE baseline exists | FINAL_CLAIM_EVIDENCE_MATRIX §E |
| "Entropy fusion causes improvement" | No ON/OFF ablation exists | FINAL_CLAIM_EVIDENCE_MATRIX §D |
| "Independent per-scale routing causes improvement" | No shared-vs-independent ablation exists | FINAL_CLAIM_EVIDENCE_MATRIX §C |
| "The model generalizes from synthetic to real" | No controlled domain-shift experiment; observational only | FINAL_CLAIM_EVIDENCE_MATRIX §H |
| "First MoE for SOD" | MMSOD, CMFNet, CMoE, PSOD, M4-SAM all apply MoE to SOD | LITERATURE_DATABASE DB-01 through DB-06 |
| "First label-free routing" | WM-MoE routes without weather labels at test time | LITERATURE_CLAIMS CLAIM-06 |
| "Computational efficiency" | No comparison to baselines; FPS hardware unspecified | FINAL_CLAIM_EVIDENCE_MATRIX §I |
| Any causal claim about a design choice | No ablation studies completed | FINAL_EXPERIMENTAL_RESULTS §4.2 |

---

## 9. Unresolved Ambiguities

| Ambiguity | Current State | Impact |
|-----------|--------------|--------|
| Exact PVTv2-B4 channel dims | timm returns `[64, 128, 320]` at `out_indices=(0,1,2)`. Verified by backbone assertion. | None — resolved. |
| Whether `deep_supervision=false` is actually enforced | Config says false. Model constructor Python default is True. `train_ddp.py` passes `config.model.deep_supervision`. Loss checks aux_logits not None. **If model is instantiated without train_ddp (e.g., direct import), deep_supervision defaults to True.** | Baseline training uses false. Paper should state this explicitly. |
| Whether `moe_type` or `router_variant` config fields are consumed | **NOT consumed.** Forward path always runs sparse dispatch regardless of these fields. | Ablation system defines variants that are not implemented. |
| Boundary metric discrepancy between evaluation runs | `results/legacy/legacy_8expert/evaluation_results/best/` has different boundary values than `results/legacy/legacy_8expert/evaluation/best_new_1/`. Root cause: earlier evaluation run with different boundary implementation. | Use `results/legacy/legacy_8expert/evaluation/best_new_1/` as canonical (consistent with current `src/metrics.py`). |
| FPS hardware context | Not specified in `compute_cost.json`. Likely Kaggle T4. | Report number without claiming efficiency. |
| Training logs / convergence | Not present in repository. | Cannot assess overfitting or convergence. |
| Multiple training runs / variance | Only one training run completed. No multiple seeds. | Cannot report confidence intervals. |
| Router hidden dimension | 128 (default in `SpatialMoELayer.__init__`). Not in config. | Report in architecture description. |

---

*This file is the authoritative source of truth for paper claims. Update only when new experiments are completed or implementation details are verified.*
