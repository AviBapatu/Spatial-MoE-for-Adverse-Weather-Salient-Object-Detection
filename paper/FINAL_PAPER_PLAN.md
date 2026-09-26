# FINAL_PAPER_PLAN.md — SCOVA Conference Paper (6-page budget)

**Title:** Spatially Dynamic Mixture-of-Experts for Adverse-Weather Salient Object Detection
**Date:** 2026-09-02
**Venue:** SCOVA (maximum 6 pages including references)
**Style:** Physics-like scientific exposition, mathematically rigorous, visually sophisticated
**Status:** The manuscript `paper/main.tex` exists and follows the structure and float plan in §2–§7; this file is the brief it was written from and the reference for any rewrite.

---

## 0. Authoritative Source Documents

This plan is derived exclusively from the research corpus below. No outside assumptions are introduced.

| Document | Role in this plan |
|----------|-------------------|
| `RESEARCH_TRUTH.md` | Authoritative scientific claims; supported/unsupported claims; evidence hierarchy |
| `docs/research/RESULTS.md` | All experimental data with sources |
| `docs/research/RELATED_WORK.md` | Defensible novelty claims, positioning, corrections to prior work |
| `docs/research/ARCHITECTURE.md` | Component-level architecture details, tensor dims |
| `docs/research/EXPERIMENTS.md` | Config system, the current experiment set, protocols |
| `docs/research/RESULTS_NARRATIVE.md` | Honest scientific interpretation of results |
| `docs/research/ABLATIONS.md` | What ablations exist vs. what was run |
| `docs/research/LITERATURE_DATABASE.md` | Verified paper registry |
| `docs/research/DATASETS.md` | WXSOD dataset structure and splits |
| `docs/research/TRAINING.md` | Training pipeline, loss, optimization |
| `paper/REDESIGN_PLAN.md` | Layout redesign guidance |

---

## 1. Scientific Position (from RESEARCH_TRUTH.md + RELATED_WORK.md)

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

### 1.3 Defensible novelty claims (from RELATED_WORK.md)

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
| **Page 1** | Title, Abstract, §I Introduction, §II Related Work, §III Method (A–B) | ~55 |
| **Page 2** | §III Method (C–G), **FIGURE 1** (hero architecture, full width), **FIGURE 2** (routing mechanism, full width) | ~55 |
| **Page 3** | §III Method end, §IV-A Setup, §IV-B Main Results text | ~55 |
| **Page 4** | **TABLE 1**, §IV-C Weather-Wise Analysis with **TABLE 2** + **FIGURE 3** (weather-wise bar chart), §IV-D Routing Entropy with **TABLE 3**, §IV-E Forced-Expert Analysis, §IV-F Discussion | ~55 |
| **Page 5** | References (8 entries) | ~20 |
| **Page 6** | **FIGURE 4** (qualitative predictions, full width), placed after the reference list | ~12 |

**Key constraint:** Total must be ≤ 6 pages including references. The full-width qualitative figure sits on the last page, after the reference list, which is what takes the paper to six pages rather than five.

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

**Source:** Derived from `src/model.py`, `src/backbone.py`, `src/moe_layer.py`, `src/decoder/`

**LaTeX:** Full-width `figure*` environment with TikZ. Keep arrow labels in `\scriptsize`.

### FIGURE 2 — Spatial-MoE Routing Mechanism (Page 2, full two-column width)

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
- Include the entropy formula as an inset equation: $\mathcal{H}_i = -\sum_{j=1}^{E} p_{i,j} \log p_{i,j}$, normalized by $\log 8$
- Color-coding consistent with Fig. 1

**LaTeX:** Half-width `figure` environment. TikZ flow diagram.

### FIGURE 3 — Weather-Wise Quantitative Visualization (Page 4, half width)

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

### FIGURE 4 — Qualitative Predictions (Page 6, full two-column width)

**Purpose:** Visual evidence of model output across weather conditions.

**Content:** 3 rows, columns:
1. Input image (from `data/WXSDO_data/test_real/input/`)
2. Ground truth (from `data/WXSDO_data/test_real/gt/`)
3. Model prediction (from `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/`)
4. Absolute-error map (red-tinted)

**Rows (selected from actual evaluation outputs):**
- **Row 1:** Snow (best MAE=0.0094) — select a representative snow image
- **Row 2:** Fog (mid MAE=0.0148) — select a representative fog image
- **Row 3:** Low-light (worst MAE=0.0243) — select a representative low-light image

**Source:** Use actual PNG files from `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/*.png` matched to input images by filename.

**Style:**
- Compact, high-resolution crops
- Each row labeled with weather type and sample count
- Caption states: "Qualitative examples on WXSOD real-world test. Rows: snow ($N{=}90$, MAE 0.0094), fog ($N{=}126$, MAE 0.0148), low-light ($N{=}93$, MAE 0.0243)."

**LaTeX:** Full-width `figure*` environment, placed after the reference list so it lands on the last page. Use `\includegraphics` with `width=\linewidth` for each subfigure column.

### Figures NOT included (and why)

| Candidate | Reason for exclusion |
|-----------|---------------------|
| Forced-expert ablation chart | Marginal result (ΔMAE=+0.9%); better as text callout |
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

### TABLE 2 — Weather-Wise Results (Page 4, half width, beside Fig. 3)

| Weather | $N$ | MAE ↓ | $S_\phi$ ↑ | $F_\beta^{\max}$ ↑ |
|---------|-----|-------|------------|---------------------|
| Snow | 90 | **0.0094** | **0.9478** | **0.9421** |
| Fog | 126 | 0.0148 | 0.9179 | 0.8914 |
| Rain | 120 | 0.0165 | 0.9157 | 0.8935 |
| Dark | 125 | 0.0191 | 0.9072 | 0.8885 |
| Low-light | 93 | 0.0243 | 0.8897 | 0.8629 |

**Caption:** Weather-wise results on real-world test (554 images). Best values in **bold**.

**Source:** `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json`

### TABLE 3 — Mean Routing Entropy (Page 4, half width)

| Scale | Synthetic (nats) | % of max | Real (nats) | % of max |
|-------|------------------|----------|-------------|----------|
| $1/4$  | 2.0786 | 99.96 | 2.0788 | 99.97 |
| $1/8$  | 2.0788 | 99.97 | 2.0789 | 99.97 |
| $1/16$ | 2.0755 | 99.81 | 2.0756 | 99.82 |

**Caption:** Mean routing entropy per scale. Entropy is over the full distribution over all $E{=}8$ experts, so the maximum is $\log 8 = 2.0794$ nats.

**Source:** `results/legacy/legacy_8expert/routing_entropy/entropy_comparison.json`

### Tables NOT present

| Table | Why |
|-------|-----|
| Forced-expert table | Folded into text: forcing all tokens at scale 1/4 to expert 0 changes MAE by +0.0002 (+0.9%). |

---

## 5. Equation Plan

`paper/main.tex` numbers twelve equations in total; the table below lists the essential ones and their content.

| Eq. | Content | Compact Form | Variables to define |
|-----|---------|--------------|-------------------|
| 1 | Router input | $\br_i = [\bx_i \;\|\; \text{DWConv}_{3\times3}(\bX_s)_i] \in \reals^{2C}$ | $\bX_s$ (features at scale $s$), $C{=}256$ (channels), $\|$ (concat) |
| 2 | Routing logits | $\bh_i = W_2 \cdot \text{GELU}(W_1 \br_i + b_1) + b_2 \in \reals^{E}$ | $W_1 \in \reals^{128 \times 2C}$, $W_2 \in \reals^{E \times 128}$, $E{=}8$ |
| 3 | Noisy top-$K$ | $\mathcal{T}_i = \topk_j(\tilde{h}_{i,j}, K{=}2)$, $g_{i,k} = \text{softmax}(\tilde{h}_{i,k})_{k \in \mathcal{T}_i}$ | $K{=}2$, $\tilde{h}_{i,j} = h_{i,j} + \epsilon_{i,j}$, $\epsilon_{i,j} \sim \mathcal{N}(0, \sigma_j^2(\bx_i))$ |
| 4 | Sparse dispatch | $\by_i = \sum_{k \in \mathcal{T}_i} g_{i,k} \cdot \E_k(\bx_i)$ | $\E_k$ (expert $k$) |
| 5 | Routing entropy | $\mathcal{H}_i = -\sum_{j=1}^{E} p_{i,j} \log p_{i,j}$, $\Hs_i = \mathcal{H}_i / \log 8$ | $\Hs_i \in [0,1]$ (normalized entropy over all $E$ experts), $E{=}8$ |
| 6 | Entropy fusion | $\bF_s = \text{Proj}_Y(\bY_s) + \lambda_s \cdot \text{Proj}_\mathcal{H}(\Hs_s)$ | $\lambda_s$ (learnable, init 0.1) |
| 7 | Total loss | $\mathcal{L} = \lambda_{\text{BCE}}\mathcal{L}_{\text{BCE}} + \lambda_{\text{IoU}}\mathcal{L}_{\text{IoU}} + \lambda_{\text{SSIM}}\mathcal{L}_{\text{SSIM}} + \lambda_{\text{BND}}\mathcal{L}_{\text{BND}} + \lambda_{\text{LB}}\mathcal{L}_{\text{LB}} + \lambda_{\text{IMP}}\mathcal{L}_{\text{IMP}} + \lambda_{\text{auxBND}}\mathcal{L}_{\text{auxBND}} + \lambda_{\text{DS}}\mathcal{L}_{\text{DS}}$ | $\lambda_{\text{BCE}}{=}\lambda_{\text{IoU}}{=}\lambda_{\text{SSIM}}{=}\lambda_{\text{BND}}{=}1.0$, $\lambda_{\text{LB}}{=}\lambda_{\text{IMP}}{=}0.01$, $\lambda_{\text{auxBND}}{=}0.5$, $\lambda_{\text{DS}}{=}0.4$; Z-loss and router-confidence off |

**Loss term definitions (compact):**
- $\mathcal{L}_{\text{BCE}}$: Binary cross-entropy with logits.
- $\mathcal{L}_{\text{IoU}}$: Soft IoU: $1 - \text{mean}_b\!\left[\frac{\sum_{hw}\sigma(\hat{M}_b) \cdot M_b + \epsilon}{\sum_{hw}\sigma(\hat{M}_b) + \sum_{hw} M_b - \sum_{hw}\sigma(\hat{M}_b) \cdot M_b + \epsilon}\right]$.
- $\mathcal{L}_{\text{LB}}$: Load-balancing: $\frac{1}{|\mathcal{S}|}\sum_{s} E \sum_j f_j^{(s)} P_j^{(s)}$ (Shazeer form).
- $\mathcal{L}_{\text{IMP}}$: Importance: $\frac{1}{|\mathcal{S}|}\sum_{s} (\text{std}(\bP^{(s)}) / \text{mean}(\bP^{(s)}))^2$.

**Note:** the Z-loss and the router-confidence term are the only inactive terms ($\lambda{=}0$); deep supervision, boundary and SSIM are all active. State this in one sentence after the loss equation.

---

## 6. Reference List (8 entries)

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

These eight are exactly the keys in `paper/references.bib`, and all eight are cited in `paper/main.tex`.

**Absent from `paper/references.bib`** — do not cite without adding the entry:
- `wfynet2025` (WFANet) — second direct competitor
- `complexity2025` (Complexity Experts, CVPR 2025) — recent SOTA MoE restoration
- `mmsod2025`, `cmfnet2026`, `cmoe2026`, `psod2025`, `soft-moe2024`, `mofme2024` (considered and dropped)

**Rationale:** For a 6-page paper, references must still each directly support a claim in the paper, but the list can run to the full shortlist above. Metric definitions are necessary for reproducibility. WM-MoE and NIFM are the two closest competitors and must be cited for differentiation.

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

### §I Introduction (~0.4 page, 12 lines)

**Paragraph 1 (6 lines):** Problem + prior work gap.
- SOD identifies visually prominent regions but degrades under adverse weather
- Two strategies exist: (a) weather-label-conditioned models [NIFM] — require metadata at inference; (b) MoE for weather restoration [WM-MoE] — target pixel regression, not segmentation
- Existing MoE-SOD methods route at modality/scale/task level — not spatial token level for single-RGB SOD
- No existing method combines spatial token-level routing, independent multi-scale routing, and routing-informed decoding without weather-specific supervision
- Scope every novelty sentence with "to our knowledge" (see §11.2)

**Paragraph 2 (6 lines):** Our approach + 3 contributions.
- We present SpatialMoE-SOD: token-level spatial mixture-of-experts that routes individual spatial tokens to **separate** expert networks at three independent pyramid scales without weather labels
- Contribution 1: Token-level spatial router (DWConv3x3 + MLP, noisy top-K=2, true sparse dispatch)
- Contribution 2: Three independent routers at scales 1/4, 1/8, 1/16, each with separate expert pool
- Contribution 3: Per-token routing entropy over the full expert distribution, normalized by $\log 8$ and injected as explicit decoder feature via learnable additive fusion

### §II Related Work (~0.2 page, 8 lines)

Three short subsections, matching `paper/main.tex` §II:
- **A. Adverse-Weather SOD** — NIFM fuses noise indicators from explicit weather-type one-hot vectors; WXSOD provides the benchmark (per-image weather labels, synthetic and real test splits).
- **B. MoE for Weather Restoration** — WM-MoE uses weather-aware routing for adverse-weather image restoration; our method performs spatial token-level routing for single-RGB SOD without weather-specific supervision; existing MoE weather methods target pixel regression, not segmentation.
- **C. MoE for SOD** — to our knowledge existing MoE-SOD methods route at modality, scale or task level rather than per token for single-RGB input; V-MoE established token-level top-$K$ routing for classification; routing entropy has been used for uncertainty estimation in MoE classifiers but not, to our knowledge, as a dense-prediction decoder feature (and no ablation isolates it — say so).

### §III Method (~1.4 pages, 42 lines)

**Intro (3 lines).** The full pipeline, referencing Fig. 1; a shared PVTv2-B4 backbone, three independent SpatialMoE layers, and a cross-attention decoder.

#### §III-A Multi-Scale Features (~0.15 page, 5 lines)
- PVTv2-B4 backbone extracts features at 1/4, 1/8, 1/16 scales ($\texttt{out\_indices}=(0,1,2)$)
- Each projected to $C{=}256$ via 1×1 conv; tokens $\bx_i \in \reals^C$
- Token flattening + projection equation

#### §III-B Router (~0.25 page, 8 lines)
- Router input is the token itself concatenated with its local depthwise-conv context: $\br_i = [\bx_i \mid \text{DWConv}_{3\times3}(\bX_s)_i] \in \reals^{2C}$ (there is **no** global-context term in the router)
- Two-layer MLP: $W_2 \cdot \text{GELU}(W_1 \br_i + b_1) + b_2 \in \reals^{E}$, $E{=}8$, hidden width 128
- Router input and logit equations

#### §III-C Noisy Top-$K$ Routing (~0.2 page, 6 lines)
- Input-dependent Gaussian noise on the logits, training only: $\sigma_j(\bx_i) = \text{softplus}(w_j^\top \bx_i + b_j)$
- Top-$K{=}2$ selection; gate weights are the softmax over the **selected** logits only
- Noisy-gating and gate equations

#### §III-D Sparse Expert Aggregation (~0.2 page, 6 lines)
- True sparse dispatch: loop over experts, mask selection, gate-weighted scatter-add
- Expert: $\E_k(\bx) = \bx + W_k^{(2)} \text{GELU}(W_k^{(1)} \text{LN}(\bx))$, $4C$ hidden
- 8 experts per scale, 24 total, no weight sharing
- Dispatch and expert equations

#### §III-E Routing Entropy (~0.2 page, 6 lines)
- Entropy over the **full** $E$-way softmax of the router logits: $\mathcal{H}_i = -\sum_{j=1}^{E} p_{i,j} \log p_{i,j}$, normalized by $\log 8$ (hard-coded, matching $E{=}8$)
- Entropy equation

#### §III-F Decoder (~0.3 page, 9 lines)
- Per-scale EntropyFusionBlock: $\bF_s = \text{Proj}_Y(\bY_s) + \lambda_s \cdot \text{Proj}_\mathcal{H}(\Hs_s)$, $\lambda_s$ learnable, init 0.1
- Top-down cross-attention: global (1/16→1/8, 8 heads), windowed (1/8→1/4, 7×7 window)
- Refinement blocks → dual heads (saliency + boundary)
- Entropy-fusion and cross-attention equations

#### §III-G Training Objective (~0.2 page, 6 lines)
- The eight active terms: BCE + IoU + SSIM + boundary + per-scale load-balance + importance + auxiliary boundary + deep supervision
- Active weights: $\lambda_{\text{BCE}}{=}\lambda_{\text{IoU}}{=}\lambda_{\text{SSIM}}{=}\lambda_{\text{BND}}{=}1.0$, $\lambda_{\text{LB}}{=}\lambda_{\text{IMP}}{=}0.01$, $\lambda_{\text{auxBND}}{=}0.5$, $\lambda_{\text{DS}}{=}0.4$
- One sentence: the Z-loss and the router-confidence term are the only inactive terms ($\lambda{=}0$); deep supervision is enabled, so the auxiliary heads exist
- Loss equation

### §IV Experiments (~0.7 page)

#### §IV-A Setup (~0.3 page, 9 lines)

- **Dataset:** WXSOD [Chen et al.]. train_sys (12,891 images, scene-aware 80/20 split), test_sys (1,500 synthetic, 9 weather categories), test_real (554 real-world, 5 categories: fog, rain, snow, dark, low-light).
- **Implementation:** PyTorch, DDP on 2 GPUs, AMP FP16, AdamW (lr $10^{-4}$, weight decay $10^{-4}$), WarmupCosine (1% warmup, 50-epoch budget), gradient accumulation (effective batch 32), gradient clipping (1.0). Backbone frozen epoch 0. Loss weights as in §III-G.
- **Metrics:** MAE, $S_\phi$ [Cheng et al.], $E_\phi^{\text{adp}}$ [Fan et al.], $F_\beta^{\max}$ [Margolin et al.].
- **Compute:** 69.21M parameters (69,213,120 for the reported recipe, which has deep supervision on), 277.9G MACs.

#### §IV-B Main Results (~0.25 page, 7 lines)
- Table 1: Global results on test_sys and test_real
- Observation: Real-world MAE (0.0168) comparable to or slightly better than synthetic (0.0192)
- Qualification: observational comparison, not a controlled domain-shift experiment; may reflect compound weather conditions in the synthetic test set

#### §IV-C Weather-Wise Analysis (~0.25 page, 7 lines)
- Table 2: Weather-wise breakdown on test_real
- Snow easiest (MAE 0.0094), low-light hardest (MAE 0.0243), 2.6× range
- **Fig. 3**: Weather-wise MAE bar chart (half width)
- **Fig. 4**: Qualitative predictions across weather conditions (full width, last page)

#### §IV-D Routing Entropy (~0.1 page, 3 lines)
- Table 3: Mean routing entropy per scale
- 2.076–2.079 nats, i.e. 99.8–100% of the $\log 8 = 2.0794$ ceiling; routing over the eight experts is close to uniform
- Scale 1/16 is marginally less uniform (2.0755 nats) than the finer scales (2.0786–2.0789); synthetic vs real differ by <0.003

#### §IV-E Forced-Expert Analysis (~0.1 page, 3 lines)
- Text callout (no table): Forcing all tokens at scale 1/4 to expert 0 → MAE +0.0002 (+0.9%)
- Two interpretations: (a) expert pool over-parameterized, or (b) experts learn similar functions
- Only 1 of 3 scales forced; full effect unknown
- This is a limited diagnostic, not evidence for or against routing utility

#### §IV-F Discussion (~0.1 page, 5 lines)
- Single training run, no variance estimates
- No controlled comparison against NIFM
- No ablation studies completed (routing entropy, expert count, $K$ value, scale independence)
- Forced-expert analysis suggests limited differentiation at scale 1/4
- Boundary F1 is low (0.36 real, 0.55 synthetic) — weakest component
- Low-light is a systematic weakness (MAE 0.0243)

### §V Conclusion (~0.1 page, 3 lines)
- We presented SpatialMoE-SOD: token-level spatial MoE for adverse-weather SOD
- Routes tokens to 8 experts at 3 independent scales without weather labels
- Routing entropy fused as explicit decoder feature
- MAE 0.0192 (synthetic, 1,500 images) and 0.0168 (real-world, 554 images) on WXSOD
- Future work: ablation studies, comparison with existing methods, expert interpretability analysis

### References (~0.3 page, 20 lines)
8 entries, all cited in the text. See §6 above.

---

## 8. Content to Remove from Current Manuscript

**Status:** these compressions were considered for the earlier 4-page budget. Under the 6-page budget the manuscript keeps §II Related Work as three subsections, keeps Table 3, and keeps the Discussion subsection; treat this table as optional trimming, not as a description of `paper/main.tex`.

| Current Content | Action | Rationale |
|----------------|--------|-----------|
| §II Related Work (3 subsections, ~0.4 page) | Optional: compress to 1 dense paragraph, ~8 lines | Save ~0.3 page. Not applied — the manuscript keeps the three subsections. |
| §IV-F Discussion paragraph (~0.3 page) | Already folded into the Discussion subsection | Save ~0.2 page |
| §V Conclusion (2 paragraphs, ~0.2 page) | Compress to 3–4 sentences, ~3 lines | Save ~0.15 page |
| Table 3 (entropy) | Keep (not applied) | The manuscript reports it; Table 3 is in `paper/main.tex` |
| Forced-expert table | Already absent; text callout instead | Save ~0.1 page |
| Verbose loss explanation | Compact to the loss equation + weights sentence | Save ~0.05 page |
| Redundant MAE restatements | State once, reference table | Save ~0.05 page |
| 15 references → 8 | Trimmed to 8 essential references | Save ~0.2 page |

**Total space recovered:** ~0.5 page of the earlier estimate. Under the 6-page budget this is available, not required.

---

## 9. Content to Convert from Prose into Diagrams

| Current Prose | Convert to | Figure |
|---------------|-----------|--------|
| Architecture description (~0.6 page of text) | FIGURE 1: Full-width architecture diagram | Fig. 1 |
| Router mechanism description (~0.15 page) | FIGURE 2: Routing mechanism | Fig. 2 |
| Weather-wise analysis text (~0.25 page) | FIGURE 3: Weather-wise MAE bar chart | Fig. 3 |
| Qualitative analysis | FIGURE 4: Actual prediction examples | Fig. 4 |

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

## 11. Scientific Positioning (from RESEARCH_TRUTH.md + RELATED_WORK.md)

### 11.1 Claims to make (architectural contribution, empirical characterization)

1. We propose a token-level spatial router that assigns each token to top-2 of 8 experts per scale.
2. We deploy independent routers at each pyramid scale (1/4, 1/8, 1/16).
3. We feed per-token routing entropy as an explicit feature channel into the decoder.
4. The model has no weather-specific component — routing operates purely on spatial content features.
5. The model achieves MAE 0.0192 on synthetic and 0.0168 on real-world weather conditions.

### 11.2 Claims to AVOID (from RESEARCH_TRUTH.md §3)

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

The forced-expert ablation (MAE +0.9% when forcing scale 1/4 to expert 0) must be presented as:
- A limited diagnostic (only 1 of 3 scales forced)
- Not evidence for or against routing utility
- Two possible interpretations stated neutrally
- No causal conclusion drawn

### 11.4 How to handle the entropy fusion

Present as:
- A novel architectural design choice (routing entropy as decoder input)
- No prior work does this to our knowledge (see `docs/research/RELATED_WORK.md`)
- Explicitly note that ablation to validate its contribution was not performed

---

## 12. LaTeX Template Notes

- Use `\documentclass[10pt,twocolumn]{article}` (already in `main.tex`)
- Use `\usepackage[margin=0.6in]{geometry}` (already in `main.tex`)
- Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`
- Verify page count ≤ 6 before submission
- Include `\usepackage{hyperref}` for clickable references
- Use `\usepackage{pgfplots}` for Figure 4 (bar chart) if generating programmatically
- Use `\usepackage{subcaption}` or manual spacing for Figure 3 subfigures

---

## 13. Verification Checklist

Before finalizing the manuscript, verify:

| # | Item | Source |
|---|------|--------|
| 1 | All MAE/S/F/E values match `results/legacy/legacy_8expert/evaluation/best_new_1/*/none/metrics.json` | `docs/research/RESULTS.md` §1–2 |
| 2 | Weather-wise values match weather breakdown in metrics.json | `docs/research/RESULTS.md` §2 |
| 3 | Routing entropy values match `results/legacy/legacy_8expert/routing_entropy/entropy_comparison.json` (the pre-fix `eval_results/entropy_comparison.json` is top-2 only) | `docs/research/RESULTS.md` §4 |
| 4 | Forced-expert values match `results/legacy/legacy_8expert/eval_results/proxy_ablation_results.json` | `docs/research/RESULTS.md` §3 |
| 5 | Tensor dimensions match `RESEARCH_TRUTH.md` §1.1 | RESEARCH_TRUTH.md |
| 6 | No forbidden claims appear (§11.2 above) | RESEARCH_TRUTH.md §3 |
| 7 | All references are from §6 shortlist | This plan §6 |
| 8 | Page count ≤ 6 via pdflatex build | Manual verification |
| 9 | All figures are self-contained (axis labels, legends, captions) | This plan §10 |
| 10 | Every variable defined before first use | This plan §5 |

---

*This plan is a structural blueprint for the FINAL manuscript. It defines WHAT to write, WHERE, and with what scientific positioning. The actual prose will be written separately following this plan.*
