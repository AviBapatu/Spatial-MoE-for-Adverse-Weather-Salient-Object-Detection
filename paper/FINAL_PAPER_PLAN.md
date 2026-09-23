# FINAL_PAPER_PLAN.md — SCOVA 4-Page Conference Paper

**Title:** Spatially Dynamic Mixture-of-Experts for Adverse-Weather Salient Object Detection
**Date:** 2026-09-02
**Venue:** SCOVA (maximum 4 pages including references)
**Style:** Physics-like scientific exposition, mathematically rigorous, visually sophisticated
**Status:** PLAN ONLY — not yet written

---

## 0. Authoritative Source Documents

This plan is derived exclusively from the research corpus below. No outside assumptions are introduced.

| Document | Role in this plan |
|----------|-------------------|
| `RESEARCH_TRUTH.md` | Authoritative scientific claims; evidence hierarchy |
| `RESEARCH_TRUTH.md` | Which claims are supported/unsupported |
| `docs/research/RESULTS.md` | All experimental data with sources |
| `docs/research/FINAL_RESEARCH_POSITION.md` | Defensible novelty claims, corrections to prior work |
| `RESEARCH_TRUTH.md` | Verified architecture, tensor dims, results |
| `docs/research/ARCHITECTURE.md` | Component-level architecture details |
| `docs/research/EXPERIMENTS.md` | Config, weather-wise breakdown, key observations |
| `docs/research/RESULTS_NARRATIVE.md` | Honest scientific interpretation of results |
| `docs/research/ABLATIONS.md` | What ablations exist vs. what was run |
| `docs/research/NOVELTY_MATRIX.md` | Comparative analysis vs. prior work |
| `docs/research/LITERATURE_DATABASE.md` | Verified paper registry (63 papers) |
| `docs/research/LITERATURE_CLAIMS.md` | Externally verified literature claims |
| `docs/research/DATASETS.md` | WXSOD dataset structure and splits |
| `docs/research/TRAINING.md` | Training pipeline, loss, optimization |
| `paper/REDESIGN_PLAN.md` | Layout redesign guidance |
| `RESEARCH_TRUTH.md` | Paper-level verified claims |

---

## 1. Scientific Position (from FINAL_RESEARCH_POSITION.md + FINAL_CLAIM_EVIDENCE_MATRIX.md)

### 1.1 What the experiments support

1. **Performance reporting:** "Our model achieves X on test set Y under condition Z." — Fully supported.
2. **Architectural description:** "Our model uses token-level routing with K=2 experts at each of 3 scales..." — Fully supported by implementation.

### 1.2 What the experiments do NOT support

- Cannot claim any design choice causes improvement
- Cannot claim MoE is better than non-MoE
- Cannot claim routing is better than no routing
- Cannot claim entropy fusion helps
- Cannot claim experts specialize
- Cannot claim efficiency
- Cannot claim SOTA performance

### 1.3 Defensible novelty claims (from FINAL_RESEARCH_POSITION.md §4-5)

| Claim | Status | Recommended wording |
|-------|--------|-------------------|
| Token-level MoE for SOD | IMPLEMENTED, NOT TESTED against non-MoE | "We apply token-level spatial MoE routing to salient object detection, extending MoE from classification [V-MoE] and restoration [WM-MoE] to dense segmentation." |
| Independent per-scale routing | IMPLEMENTED, NOT TESTED against shared routing | "Each pyramid scale has its own independent router and expert pool, allowing different routing decisions at different receptive fields." |
| Routing entropy as decoder input | IMPLEMENTED, NOT TESTED against no-entropy | "We feed per-token routing entropy as an explicit feature channel into the decoder, providing the model with an explicit uncertainty signal about routing decisions at each spatial location." |
| No weather-specific component | IMPLEMENTED, NOT TESTED against weather-branch | "Unlike prior weather-aware methods that require a dedicated weather branch [WM-MoE] or explicit weather labels [NIFM, WFANet], our router operates purely on spatial content features." |
| MoE for adverse-weather SOD | IMPLEMENTED, NOT TESTED against non-MoE | "We apply mixture-of-experts specifically to salient object detection under adverse weather conditions, a setting not previously addressed with MoE routing." |

**Critical constraint:** These must be presented as architectural contributions with empirical performance characterization, NOT as empirically validated designs. No causal language.

---

## 2. Page Budget

| Page | Content | Approx. Lines |
|------|---------|---------------|
| **Page 1** | Title, Abstract, §1 Introduction, §2 Method (§2.1–§2.3) | ~55 |
| **Page 2** | §2 Method (§2.4–§2.6), **FIGURE 1** (hero architecture, full width) | ~55 |
| **Page 3** | §3 Experimental Setup, §4 Results (Table 1 + Table 2), **FIGURE 2** (routing mechanism zoom), **FIGURE 3** (qualitative predictions, full width) | ~55 |
| **Page 4** | §4 Results (§4.3 routing entropy, §4.4 forced-expert), §5 Limitations, §6 Conclusion, **FIGURE 4** (weather-wise bar chart), compact References | ~55 |

**Key constraint:** Total must be ≤ 4 pages including references. This plan assumes references fall on page 4 (approximately 20 lines for 6–8 entries).

---

## 3. Figure Plan

### FIGURE 1 — Hero Architecture Diagram (Page 2, full two-column width)

**Purpose:** Communicate the complete end-to-end pipeline visually.

**Content:**
```
RGB Input [3, 384, 384]
    ↓
PVTv2-B4 Backbone
    ↓ (three arrows, labeled)
[256, 96, 96]    [256, 48, 48]    [256, 24, 24]
   1/4 scale       1/8 scale       1/16 scale
    ↓                 ↓                ↓
SpatialMoELayer    SpatialMoELayer   SpatialMoELayer
(8 experts,        (8 experts,       (8 experts,
 top-2 routing)     top-2 routing)    top-2 routing)
    ↓   ↓             ↓   ↓             ↓   ↓
  Y_4  H_4         Y_8  H_8         Y_16 H_16
    ↓                 ↓                ↓
  EntropyFusion    EntropyFusion    EntropyFusion
    ↓                 ↓                ↓
    └──────── Cross-Attention Decoder ────────┘
                      ↓
              ┌───────┴───────┐
              ↓               ↓
        Saliency Head    Boundary Head
        [1, 384, 384]    [1, 384, 384]
```

**Style:**
- Color-coded blocks: blue=backbone, orange=router, green=expert, red=entropy, gray=decoder, purple=heads
- Tensor dimensions at key points (e.g., `[256, 96, 96]`)
- Data flow arrows with direction
- Scale labels ($1/4$, $1/8$, $1/16$) prominently displayed
- No decorative elements — pure scientific communication

**Source:** Derived from `src/model.py:6-48`, `src/backbone.py:5-39`, `src/moe_layer.py:35-165`, `src/decoder/:219-280`

**LaTeX:** Full-width `figure*` environment with TikZ. Keep arrow labels in `\scriptsize`.

### FIGURE 2 — Spatial-MoE Routing Mechanism (Page 3, half width)

**Purpose:** Show how a single spatial token gets routed to experts.

**Content:**
```
Token x_i ∈ ℝ^256
    ↓
DWConv3x3 → local context ∈ ℝ^256
    ↓
Concatenate [x_i ∥ local] ∈ ℝ^512
    ↓
MLP (512 → 128 → 8) → clean logits h_i ∈ ℝ^8
    ↓ (+ Gaussian noise during training)
Noisy logits → Top-2 selection
    ↓
Softmax over top-2 → gates g_{i,1}, g_{i,2}
    ↓
Expert_k1(x_i) × g_{i,1}  +  Expert_k2(x_i) × g_{i,2}
    ↓
Output y_i  +  Entropy H_i
```

**Style:**
- Compact flow diagram with math annotations
- Include the entropy formula as an inset equation: $\mathcal{H}_i = -\sum_k g_{i,k} \log g_{i,k}$
- Color-coding consistent with Fig. 1

**LaTeX:** Half-width `figure` environment. TikZ flow diagram.

### FIGURE 3 — Qualitative Predictions (Page 3, full two-column width)

**Purpose:** Visual evidence of model output across weather conditions.

**Content:** 3 rows, columns:
1. Input image (from `data/WXSDO_data/test_real/input/`)
2. Ground truth (from `data/WXSDO_data/test_real/gt/`)
3. Model prediction (from `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/`)

**Rows (selected from actual evaluation outputs):**
- **Row 1:** Snow (best MAE=0.0094) — select a representative snow image
- **Row 2:** Fog (mid MAE=0.0148) — select a representative fog image
- **Row 3:** Low-light (worst MAE=0.0243) — select a representative low-light image

**Source:** Use actual PNG files from `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/*.png` matched to input images by filename.

**Style:**
- Compact, high-resolution crops
- Each row labeled with weather type and sample count
- Caption states: "Qualitative examples on WXSOD real-world test. Rows: snow ($N{=}90$, MAE 0.0094), fog ($N{=}126$, MAE 0.0148), low-light ($N{=}93$, MAE 0.0243)."

**LaTeX:** Full-width `figure*` environment. Use `\includegraphics` with `width=\linewidth` for each subfigure column.

### FIGURE 4 — Weather-Wise Quantitative Visualization (Page 4, half width)

**Purpose:** Compact visual summary of performance across weather conditions.

**Content:** Horizontal bar chart showing MAE for 5 real-world weather categories (test_real):
- Snow: 0.0094 ($N{=}90$)
- Fog: 0.0148 ($N{=}126$)
- Rain: 0.0165 ($N{=}120$)
- Dark: 0.0191 ($N{=}125$)
- Low-light: 0.0243 ($N{=}93$)

**Style:**
- Clean bar chart with MAE values annotated on bars
- Color gradient from green (low MAE) to red (high MAE)
- Sample counts (N) on bars
- X-axis: MAE (lower is better)
- Y-axis: Weather category labels

**LaTeX:** Half-width `figure` environment. Generate with `pgfplots` or `matplotlib` (save as PDF).

### Figures NOT included (and why)

| Candidate | Reason for exclusion |
|-----------|---------------------|
| Forced-expert ablation chart | Marginal result (ΔMAE=+1.2%); better as text callout |
| Routing entropy visualization | Low information density; numbers stated in 1 sentence |
| Training curves | Not available in repository |
| t-SNE of expert embeddings | Not computed |
| Synthetic weather-wise chart | Redundant with real-world chart; save space |

---

## 4. Table Plan

### TABLE 1 — Main Results (Page 3, full width)

| Set | $N$ | MAE ↓ | $S_\phi$ ↑ | $F_\beta^{\max}$ ↑ | $E_\phi^{\text{adp}}$ ↑ |
|-----|-----|-------|------------|---------------------|-------------------------|
| Synthetic | 1,500 | 0.0192 | 0.9139 | 0.9015 | 0.9591 |
| Real-world | 554 | **0.0168** | **0.9151** | 0.8936 | 0.9530 |

**Caption:** Global performance on WXSOD test splits. Best values in **bold**.

**Source:** `results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/metrics.json`, `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json`

### TABLE 2 — Weather-Wise Results (Page 3, half width, beside Fig. 4)

| Weather | $N$ | MAE ↓ | $S_\phi$ ↑ | $F_\beta^{\max}$ ↑ |
|---------|-----|-------|------------|---------------------|
| Snow | 90 | **0.0094** | **0.9478** | **0.9421** |
| Fog | 126 | 0.0148 | 0.9179 | 0.8914 |
| Rain | 120 | 0.0165 | 0.9157 | 0.8935 |
| Dark | 125 | 0.0191 | 0.9072 | 0.8885 |
| Low-light | 93 | 0.0243 | 0.8897 | 0.8629 |

**Caption:** Weather-wise results on real-world test (554 images). Best values in **bold**.

**Source:** `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json`

### Tables REMOVED from current manuscript

| Removed Table | Replacement |
|---------------|-------------|
| Table 3 (entropy, 3 rows × 3 cols) | Fold key numbers into text: "Mean normalized entropy ranges from 0.68–0.69 across scales, with scale 1/16 slightly more confident (0.68)." One sentence. |
| Forced-expert table (Table 4 in earlier draft) | Replace with text callout: "Forcing all tokens at scale 1/4 to expert 0 increased MAE by +0.0002 (+1.2%)." No table needed. |

---

## 5. Equation Plan

All equations compact, variables defined before first use. No undefined symbols.

| Eq. | Content | Compact Form | Variables to define |
|-----|---------|--------------|-------------------|
| 1 | Router input | $\br_i = [\bx_i \;\|\; \text{DWConv}_{3\times3}(\bX_s)_i] \in \reals^{2C}$ | $\bX_s$ (features at scale $s$), $C{=}256$ (channels), $\|$ (concat) |
| 2 | Routing logits | $\bh_i = W_2 \cdot \text{GELU}(W_1 \br_i + b_1) + b_2 \in \reals^{E}$ | $W_1 \in \reals^{128 \times 2C}$, $W_2 \in \reals^{E \times 128}$, $E{=}8$ |
| 3 | Noisy top-$K$ | $\mathcal{T}_i = \topk_j(\tilde{h}_{i,j}, K{=}2)$, $g_{i,k} = \text{softmax}(\tilde{h}_{i,k})_{k \in \mathcal{T}_i}$ | $K{=}2$, $\tilde{h}_{i,j} = h_{i,j} + \epsilon_{i,j}$, $\epsilon_{i,j} \sim \mathcal{N}(0, \sigma_j^2(\bx_i))$ |
| 4 | Sparse dispatch | $\by_i = \sum_{k \in \mathcal{T}_i} g_{i,k} \cdot \E_k(\bx_i)$ | $\E_k$ (expert $k$) |
| 5 | Routing entropy | $\mathcal{H}_i = -\sum_{k \in \mathcal{T}_i} g_{i,k} \log g_{i,k}$, $\Hs_i = \mathcal{H}_i / \log 2$ | $\Hs_i \in [0,1]$ (normalized entropy) |
| 6 | Entropy fusion | $\bF_s = \text{Proj}_Y(\bY_s) + \lambda_s \cdot \text{Proj}_\mathcal{H}(\Hs_s)$ | $\lambda_s$ (learnable, init 0.1) |
| 7 | Total loss | $\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{IoU}}\mathcal{L}_{\text{IoU}} + \lambda_{\text{LB}}\mathcal{L}_{\text{LB}} + \lambda_{\text{IMP}}\mathcal{L}_{\text{IMP}}$ | $\lambda_{\text{IoU}}{=}1.0$, $\lambda_{\text{LB}}{=}\lambda_{\text{IMP}}{=}0.01$ |

**Loss term definitions (compact):**
- $\mathcal{L}_{\text{BCE}}$: Binary cross-entropy with logits.
- $\mathcal{L}_{\text{IoU}}$: Soft IoU: $1 - \text{mean}_b\!\left[\frac{\sum_{hw}\sigma(\hat{M}_b) \cdot M_b + \epsilon}{\sum_{hw}\sigma(\hat{M}_b) + \sum_{hw} M_b - \sum_{hw}\sigma(\hat{M}_b) \cdot M_b + \epsilon}\right]$.
- $\mathcal{L}_{\text{LB}}$: Load-balancing: $\frac{1}{|\mathcal{S}|}\sum_{s} E \sum_j f_j^{(s)} P_j^{(s)}$ (Shazeer form).
- $\mathcal{L}_{\text{IMP}}$: Importance: $\frac{1}{|\mathcal{S}|}\sum_{s} (\text{std}(\bP^{(s)}) / \text{mean}(\bP^{(s)}))^2$.

**Note:** Deep supervision, boundary, SSIM, and Z-loss losses are defined in the codebase but inactive ($\lambda{=}0$) in the reported configuration. State this in one sentence after Eq. 7.

---

## 6. Reference Shortlist (6–8 essential)

| # | Key | Citation | Why Essential |
|---|-----|----------|---------------|
| 1 | `wxsod2025` | Chen et al., Pattern Recognition 2026 | The benchmark. Must cite. |
| 2 | `nifm2025` | Chen et al., arXiv 2025 | Direct competitor (weather-label SOD). Most direct comparison point. |
| 3 | `wmmoe2023` | Luo et al., arXiv 2023 (WM-MoE) | Closest MoE weather restoration work. Must distinguish from. |
| 4 | `v-moe2021` | Riquelme et al., NeurIPS 2021 | Foundational token-level MoE routing. Architectural ancestor. |
| 5 | `pvtv2` | Wang et al., CVM 2022 | Backbone architecture. |
| 6 | `struct-measure` | Cheng et al., ICCV 2017 | $S_\phi$ metric definition. |
| 7 | `enhanced-measure` | Fan et al., IJCAI 2016 | $E_\phi$ metric definition. |
| 8 | `f-measure` | Margolin et al., arXiv 2014 | $F_\beta$ metric definition. |

**Optional (if space permits):**
- `wfynet2025` (WFANet) — second direct competitor
- `complexity2025` (Complexity Experts, CVPR 2025) — recent SOTA MoE restoration

**Removed from current 15 references:**
- `mmsod2025` (MMSOD) — multi-modal SOD, less relevant
- `cmfnet2026` (CMFNet) — RGB-D SOD, less relevant
- `cmoe2026` (CMoE) — modality-missing SOD, less relevant
- `psod2025` (PSOD) — pluralistic SOD, less relevant
- `soft-moe2024` (Soft MoE) — interesting but not cited in main claims
- `mofme2024` (MoFME) — less directly relevant

**Rationale:** For a 4-page paper, references must be highly targeted. Each reference must directly support a claim in the paper. Metric definitions are necessary for reproducibility. WM-MoE and NIFM are the two closest competitors and must be cited for differentiation.

---

## 7. Section Structure

### Title (Line 1)
**"Spatially Dynamic Mixture-of-Experts for Adverse-Weather Salient Object Detection"**

### Abstract (~100 words, 6 lines)

**Content structure:**
1. Problem: SOD degrades under adverse weather (1 sentence)
2. Limitation: Existing methods use explicit weather labels or apply MoE to restoration (1 sentence)
3. Approach: We present SpatialMoE-SOD — token-level spatial MoE routing at three independent pyramid scales without weather labels (1 sentence)
4. Key mechanism: Routing entropy injected as decoder feature via learnable additive fusion (1 sentence)
5. Results: MAE 0.0192 on 1,500 synthetic and 0.0168 on 554 real-world test images across five weather categories (1 sentence)

**Tone:** Factual. "We propose..." not "We show that...". No causal claims.

### §1 Introduction (~0.4 page, 12 lines)

**Paragraph 1 (6 lines):** Problem + prior work gap.
- SOD identifies visually prominent regions but degrades under adverse weather
- Two strategies exist: (a) weather-label-conditioned models [NIFM, WFANet] — require metadata at inference; (b) MoE for weather restoration [WM-MoE, Complexity Experts] — target pixel regression, not segmentation
- Existing MoE-SOD methods [MMSOD, CMFNet, CMoE, PSOD] route at modality/scale/task level — not spatial token level for single-RGB SOD
- No existing method combines spatial token-level routing, independent multi-scale routing, and routing-informed decoding without weather-specific supervision

**Paragraph 2 (6 lines):** Our approach + 3 contributions.
- We present SpatialMoE-SOD: token-level spatial mixture-of-experts that routes individual spatial tokens to specialized expert networks at three independent pyramid scales without weather labels
- Contribution 1: Token-level spatial router (DWConv3x3 + MLP, noisy top-K=2, true sparse dispatch)
- Contribution 2: Three independent routers at scales 1/4, 1/8, 1/16, each with separate expert pool
- Contribution 3: Per-token routing entropy normalized by $\log K$ and injected as explicit decoder feature via learnable additive fusion

### §2 Method (~1.2 pages, 36 lines)

#### §2.1 Overview (~0.1 page, 3 lines)
Brief pipeline description. Reference Fig. 1.

#### §2.2 Multi-Scale Features (~0.15 page, 5 lines)
- PVTv2-B4 backbone extracts features at 1/4, 1/8, 1/16 scales
- Each projected to $C{=}256$ via 1×1 conv
- Eq. 1 (token flattening + projection)

#### §2.3 Spatial Token-Level Router (~0.3 page, 9 lines)
- DWConv3x3 captures local context, concatenated with token → MLP → logits
- Noisy top-K selection during training, clean at inference
- Eq. 2 (router input), Eq. 3 (noisy routing)

#### §2.4 Sparse Expert Aggregation (~0.15 page, 5 lines)
- True sparse dispatch: Python loop over experts, mask selection, gate-weighted sum
- Expert architecture: LN → Linear($C$, $4C$) → GELU → Linear($4C$, $C$) → residual
- 8 experts per scale, 24 total, no weight sharing
- Eq. 4 (sparse dispatch)

#### §2.5 Decoder with Entropy Fusion (~0.3 page, 9 lines)
- Per-scale EntropyFusionBlock: $\bF_s = \text{Proj}_Y(\bY_s) + \lambda_s \cdot \text{Proj}_\mathcal{H}(\Hs_s)$
- Top-down cross-attention: global (1/16→1/8), windowed (1/8→1/4)
- Refinement blocks → dual heads (saliency + boundary)
- Eq. 5 (entropy), Eq. 6 (entropy fusion)

#### §2.6 Training Objective (~0.2 page, 6 lines)
- Eq. 7 (total loss): BCE + IoU + load-balance + importance
- Active weights: $\lambda_{\text{IoU}}{=}1.0$, $\lambda_{\text{LB}}{=}\lambda_{\text{IMP}}{=}0.01$
- One sentence: deep supervision, boundary, SSIM, Z-loss are inactive ($\lambda{=}0$)

### §3 Experimental Setup (~0.3 page, 9 lines)

- **Dataset:** WXSOD [Chen et al.]. train_sys (12,891 images, scene-aware 80/20 split), test_sys (1,500 synthetic, 9 weather categories), test_real (554 real-world, 5 categories: fog, rain, snow, dark, low-light).
- **Implementation:** PyTorch, DDP on 2 GPUs, AMP FP16, AdamW (lr $10^{-4}$, weight decay $10^{-4}$), WarmupCosine (1% warmup, 50 epochs), gradient accumulation (effective batch 32), gradient clipping (1.0). Backbone frozen epoch 0.
- **Metrics:** MAE, $S_\phi$ [Cheng et al.], $E_\phi^{\text{adp}}$ [Fan et al.], $F_\beta^{\max}$ [Margolin et al.].
- **Compute:** 69,213,120 parameters, 278.2G MACs.

### §4 Results and Analysis (~0.7 page, 21 lines)

#### §4.1 Main Results (~0.25 page, 7 lines)
- Table 1: Global results on test_sys and test_real
- Observation: Real-world MAE (0.0168) comparable to or slightly better than synthetic (0.0192)
- Qualification: observational comparison, not controlled domain-shift experiment; may reflect compound weather conditions in synthetic test set

#### §4.2 Weather-Wise Analysis (~0.25 page, 7 lines)
- Table 2: Weather-wise breakdown on test_real
- Snow easiest (MAE 0.0094), low-light hardest (MAE 0.0243), 2.6× range
- Fig. 3: Qualitative predictions across weather conditions
- Fig. 4: Weather-wise MAE bar chart

#### §4.3 Routing Behavior (~0.1 page, 3 lines)
- Mean normalized entropy 0.68–0.69 across scales; moderate uncertainty
- Scale 1/16 slightly more confident (0.68) than finer scales (0.69)
- Synthetic vs real entropy differs by <0.003 — routing generalizes across domains

#### §4.4 Forced-Expert Analysis (~0.1 page, 3 lines)
- Text callout (no table): Forcing all tokens at scale 1/4 to expert 0 → MAE +0.0002 (+1.2%)
- Two interpretations: (a) expert pool over-parameterized, or (b) experts learn similar functions
- Only 1 of 3 scales forced; full effect unknown
- This is a limited diagnostic, not evidence for or against routing utility

#### §4.5 Limitations (~0.1 page, 3 lines)
- Single training run, no variance estimates
- No controlled comparison against NIFM or WFANet
- No ablation studies completed (routing entropy, expert count, K value, scale independence)
- Forced-expert analysis suggests limited differentiation at scale 1/4
- Boundary F1 is low (0.36 real, 0.55 synthetic) — weakest component
- Low-light is a systematic weakness (MAE 0.0243, 44% worse than average)

### §5 Conclusion (~0.1 page, 3 lines)
- We presented SpatialMoE-SOD: token-level spatial MoE for adverse-weather SOD
- Routes tokens to 8 experts at 3 independent scales without weather labels
- Routing entropy fused as explicit decoder feature
- MAE 0.0192 (synthetic, 1,500 images) and 0.0168 (real-world, 554 images) on WXSOD
- Future work: ablation studies, SOTA comparison, expert interpretability analysis

### References (~0.3 page, 20 lines)
6–8 entries. See §6 above.

---

## 8. Content to Remove from Current Manuscript

| Current Content | Action | Rationale |
|----------------|--------|-----------|
| §2 Related Work (3 subsections, ~0.4 page) | Compress to 1 dense paragraph, ~8 lines | Save ~0.3 page. Integrate essential citations into Introduction. |
| §5 Discussion paragraph (~0.3 page) | Merge into §4.5 Limitations | Save ~0.2 page |
| §6 Conclusion (2 paragraphs, ~0.2 page) | Compress to 3–4 sentences, ~3 lines | Save ~0.15 page |
| Table 3 (entropy, 3 rows × 3 cols) | Remove, fold 2 numbers into text | Save ~0.1 page |
| Forced-expert table (if present) | Remove, replace with text callout | Save ~0.1 page |
| Verbose loss explanation | Compact to Eq. 7 + 1 sentence | Save ~0.05 page |
| Redundant MAE restatements | State once, reference table | Save ~0.05 page |
| 15 references → 8 | Trim to essential 6–8 | Save ~0.2 page |

**Total space recovered:** ~1.0 page. This provides the room for 3 additional figures.

---

## 9. Content to Convert from Prose into Diagrams

| Current Prose | Convert to | Figure |
|---------------|-----------|--------|
| Architecture description (§3.1–§3.5, ~0.6 page of text) | FIGURE 1: Full-width architecture diagram | Fig. 1 |
| Router mechanism description (§3.3, ~0.15 page) | FIGURE 2: Zoomed routing mechanism | Fig. 2 |
| Weather-wise analysis text (§4.2, ~0.25 page) | FIGURE 4: Weather-wise MAE bar chart | Fig. 4 |
| Qualitative analysis (if any) | FIGURE 3: Actual prediction examples | Fig. 3 |

**Net effect:** Prose decreases by ~1.0 page, figures increase by ~1.0 page. Net content is similar; visual communication improves dramatically.

---

## 10. Visual Design System

### 10.1 Color Palette (consistent across all figures)

| Component | Color | Hex |
|-----------|-------|-----|
| Backbone | Blue | `#4A90D9` |
| Router | Orange | `#E8913A` |
| Expert | Green | `#5CB85C` |
| Entropy | Red | `#D9534F` |
| Decoder | Gray | `#8E8E8E` |
| Heads | Purple | `#9B59B6` |
| Input/Output | Black | `#333333` |

### 10.2 Typography

- **Equations:** Standard LaTeX math mode, `\mathbf` for vectors/matrices
- **Table values:** `\small` or `\footnotesize` for readability at two-column scale
- **Figure labels:** `\scriptsize` for internal annotations, `\small` for axis labels
- **Bold best values** in tables. Arrows (↓/↑) in column headers.

### 10.3 Figure Placement

- Use `[t]` or `[b]` placement. Avoid `[h]`.
- Each figure occupies at most 50% of a page height.
- Full-width figures use `figure*` environment.
- Half-width figures use `figure` environment.

### 10.4 Table Formatting

- Use `\toprule`, `\midrule`, `\bottomrule` (booktabs).
- No vertical rules.
- Center-aligned numeric columns.
- Left-aligned category/weather columns.

---

## 11. Scientific Positioning (from FINAL_RESEARCH_POSITION.md + FINAL_CLAIM_EVIDENCE_MATRIX.md)

### 11.1 Claims to make (architectural contribution, empirical characterization)

1. We propose a token-level spatial router that assigns each token to top-2 of 8 experts per scale.
2. We deploy independent routers at each pyramid scale (1/4, 1/8, 1/16).
3. We feed per-token routing entropy as an explicit feature channel into the decoder.
4. The model has no weather-specific component — routing operates purely on spatial content features.
5. The model achieves MAE 0.0192 on synthetic and 0.0168 on real-world weather conditions.

### 11.2 Claims to AVOID (from FINAL_CLAIM_EVIDENCE_MATRIX.md)

| Forbidden Claim | Reason |
|----------------|--------|
| "Our method outperforms SOTA" | No comparison to other methods |
| "MoE improves over non-MoE" | No non-MoE baseline |
| "Routing is critical" | No routing ON/OFF ablation |
| "Entropy fusion helps" | No ON/OFF ablation |
| "Experts specialize by weather" | No per-expert analysis; forced-expert test shows minimal degradation |
| "Independent per-scale routing is better" | No shared-vs-independent comparison |
| "Model generalizes synthetic→real" | Observational only, not controlled |
| "Computational efficiency" | No baseline comparison |
| Any causal claim about design choice | No ablation studies completed |

### 11.3 How to handle the forced-expert result

The forced-expert ablation (MAE +1.2% when forcing scale 1/4 to expert 0) must be presented as:
- A limited diagnostic (only 1 of 3 scales forced)
- Not evidence for or against routing utility
- Two possible interpretations stated neutrally
- No causal conclusion drawn

### 11.4 How to handle the entropy fusion

Present as:
- A novel architectural design choice (routing entropy as decoder input)
- No prior work does this (verified claim from LITERATURE_CLAIMS CLAIM-21)
- Explicitly note that ablation to validate its contribution was not performed

---

## 12. LaTeX Template Notes

- Use `\documentclass[10pt,twocolumn]{article}` (already in `main.tex`)
- Use `\usepackage[margin=0.6in]{geometry}` (already in `main.tex`)
- Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`
- Verify page count ≤ 4 before submission
- Include `\usepackage{hyperref}` for clickable references
- Use `\usepackage{pgfplots}` for Figure 4 (bar chart) if generating programmatically
- Use `\usepackage{subcaption}` or manual spacing for Figure 3 subfigures

---

## 13. Verification Checklist

Before finalizing the manuscript, verify:

| # | Item | Source |
|---|------|--------|
| 1 | All MAE/S/F/E values match `results/legacy/legacy_8expert/evaluation/best_new_1/*/none/metrics.json` | PAPER_SOURCE_OF_TRUTH.md §5 |
| 2 | Weather-wise values match weather breakdown in metrics.json | PAPER_SOURCE_OF_TRUTH.md §5.2 |
| 3 | Routing entropy values match `results/legacy/legacy_8expert/evaluation_results/entropy_comparison.json` | PAPER_SOURCE_OF_TRUTH.md §5.3 |
| 4 | Forced-expert values match `results/legacy/legacy_8expert/evaluation_results/force_expert_ablation.json` | PAPER_SOURCE_OF_TRUTH.md §5.4 |
| 5 | Tensor dimensions match `RESEARCH_TRUTH.md` §1.2 | RESEARCH_TRUTH.md |
| 6 | No forbidden claims appear (§11.2 above) | FINAL_CLAIM_EVIDENCE_MATRIX.md |
| 7 | All references are from §6 shortlist | This plan §6 |
| 8 | Page count ≤ 4 via pdflatex build | Manual verification |
| 9 | All figures are self-contained (axis labels, legends, captions) | This plan §10 |
| 10 | Every variable defined before first use | This plan §5 |

---

*This plan is a structural blueprint for the FINAL manuscript. It defines WHAT to write, WHERE, and with what scientific positioning. The actual prose will be written separately following this plan.*
