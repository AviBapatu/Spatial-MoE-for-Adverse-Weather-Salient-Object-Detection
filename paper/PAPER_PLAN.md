# PAPER_PLAN.md — Compact 4-Page Paper Plan

**Title:** Spatially Dynamic Mixture-of-Experts for Adverse-Weather Salient Object Detection

**Venue:** SCOVA (max 4 pages, including references)

**Style:** Physics-like scientific prose. Define variables, define operators, use compact equations. Distinguish measurements from interpretation.

---

## Page Budget

| Page | Content | Approx. Lines |
|------|---------|---------------|
| 1 | Abstract, Introduction, Method (§1–§3.3) | ~55 |
| 2 | Method (§3.4–§3.6), Experimental Setup, Figure 1 (architecture) | ~55 |
| 3 | Results (Table 1 + Table 2), Figure 2 (weather-wise), Figure 3 (entropy/qualitative) | ~55 |
| 4 | Analysis, Conclusion, References, Figure 4 (forced-expert) | ~55 |

---

## Figure Budget (4 figures)

| Fig | Content | Placement | Purpose |
|-----|---------|-----------|---------|
| **Fig. 1** | Architecture diagram: end-to-end pipeline. Backbone → 3 MoE layers → EntropyFusion → cross-attention decoder → dual heads. Include tensor dims. | Page 2, full width | Primary architecture figure |
| **Fig. 2** | Weather-wise bar chart: test_real MAE across 5 weather categories. Optionally overlaid with test_sys for same categories. | Page 3, half width | Shows weather robustness profile |
| **Fig. 3** | Qualitative grid: 2–3 rows (snow, fog, light), columns: input, prediction, entropy map (scale 1/8). | Page 3, half width | Visual evidence of predictions + entropy |
| **Fig. 4** | Forced-expert ablation: small table or bar chart showing MAE degradation when forcing expert 0 at scale 1/4. | Page 4, quarter width | Demonstrates routing sensitivity (or insensitivity) |

---

## Table Budget (3 tables)

| Table | Content | Placement |
|-------|---------|-----------|
| **Table 1** | Main results: test_sys and test_real global metrics (MAE, S, F_max, E_max). Row for each test set. | Page 3, full width |
| **Table 2** | Weather-wise results: test_real breakdown (weather, N, MAE, S_measure, F_max). | Page 3, half width (beside Fig. 2 or below) |
| **Table 3** | Forced-expert ablation: Normal vs Forced E0 (scale 4) on test_real. Δ column. | Page 4, half width |

---

## Mathematical Content

All equations must be compact, with variables defined before use.

### Eq. 1: Token-Level Routing

Given feature map $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$ at scale $s$:

$$\mathbf{h}_i = \text{MLP}\bigl([\mathbf{x}_i \| \text{DWConv}_{3\times3}(\mathbf{X})_i]\bigr) \in \mathbb{R}^{E}$$

Noisy logits during training: $\tilde{\mathbf{h}}_i = \mathbf{h}_i + \boldsymbol{\epsilon}_i$, $\epsilon_{i,j} \sim \mathcal{N}(0, \sigma_j^2)$.

Top-$K$ selection: $\mathcal{T}_i = \text{topk}(\tilde{\mathbf{h}}_i, K=2)$.

Gate weights: $g_{i,k} = \text{softmax}(\tilde{h}_{i,k})_{k \in \mathcal{T}_i}$.

Sparse dispatch: $\mathbf{y}_i = \sum_{k \in \mathcal{T}_i} g_{i,k} \cdot \mathcal{E}_k(\mathbf{x}_i)$.

**Variables to define:** $\mathbf{X}$ (input features), $C$ (channels), $E$ (number of experts), $K$ (top-k), $\mathcal{E}_k$ (expert $k$), $\sigma_j$ (noise scale per expert), $\|$ (concatenation).

### Eq. 2: Routing Entropy

$$\mathcal{H}_i = -\sum_{k \in \mathcal{T}_i} g_{i,k} \log g_{i,k}$$

Normalized: $\hat{\mathcal{H}}_i = \mathcal{H}_i / \log K$.

**Variables to define:** $\mathcal{H}_i$ (entropy at token $i$), $\hat{\mathcal{H}}_i$ (normalized entropy).

### Eq. 3: Entropy-Decoder Fusion

Per-scale fusion in decoder:

$$\mathbf{F}_s = \text{Proj}_Y(\mathbf{Y}_s) + \lambda \cdot \text{Proj}_\mathcal{H}(\hat{\mathbf{H}}_s)$$

where $\hat{\mathbf{H}}_s \in \mathbb{R}^{1 \times H_s \times W_s}$ is the normalized entropy map at scale $s$.

**Variables to define:** $\mathbf{Y}_s$ (MoE output at scale $s$), $\lambda$ (learnable scale, init 0.1).

### Eq. 4: Training Loss

$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{IoU}} \mathcal{L}_{\text{IoU}} + \lambda_{\text{LB}} \mathcal{L}_{\text{LB}} + \lambda_{\text{IMP}} \mathcal{L}_{\text{IMP}}$$

Load-balancing: $\mathcal{L}_{\text{LB}} = E \sum_{j=1}^{E} f_j \cdot P_j$, where $f_j$ is the hard assignment fraction (detached) and $P_j$ is the mean soft gate mass for expert $j$.

Importance: $\mathcal{L}_{\text{IMP}} = \left(\frac{\text{std}(\mathbf{P})}{\text{mean}(\mathbf{P}) + \epsilon}\right)^2$.

**Variables to define:** $f_j$, $P_j$, $\lambda_{\text{IoU}} = 1.0$, $\lambda_{\text{LB}} = \lambda_{\text{IMP}} = 0.01$.

---

## Section-by-Section Plan

### Abstract (~100 words)

**Content:** State the problem (SOD degrades under adverse weather), limitation of existing approaches (weather-label-dependent), our approach (token-level spatial MoE with label-free routing at three pyramid scales), key mechanism (routing entropy fused into decoder), and main results (MAE 0.0192 synthetic, 0.0168 real).

**Tone:** Factual, no causal claims. "We propose..." not "We show that..."

### 1. Introduction (~0.4 page)

**Paragraph 1:** SOD is critical for autonomous driving, surveillance, etc. Adverse weather (fog, rain, snow, low-light) degrades performance. Existing methods either (a) use explicit weather labels [NIFM, WFANet] or (b) apply MoE to restoration, not SOD [WM-MoE].

**Paragraph 2:** We propose SpatialMoE-SOD: a token-level spatial mixture-of-experts that routes individual spatial tokens to specialized experts without any weather label at train or test time. Three contributions: (1) token-level routing for dense SOD, (2) independent routers at three pyramid scales, (3) routing entropy as an explicit decoder input feature.

**Paragraph 3:** We evaluate on WXSOD (14,945 images) across 5 real-world and 9 synthetic weather categories.

### 2. Related Work (~0.3 page)

**Tight, citation-dense paragraph format (not subsections).**

- Adverse-weather SOD: NIFM (weather labels), WFANet (two-branch), WXSOD benchmark.
- MoE for weather restoration: WM-MoE (token-level, weather branch), MoFME (uncertainty), Complexity Experts (image-level).
- MoE for SOD: MMSOD (modality-level), CMFNet (scale-level), CMoE (modality-level). None perform spatial token-level routing.
- Token-level MoE: V-MoE (classification), Soft MoE (differentiable), M³ViT (task-conditioned). Applied to SOD here.

### 3. Method (~1.2 pages)

#### 3.1 Overview (~0.1 page)

Brief pipeline description. Reference Fig. 1.

#### 3.2 Backbone and Multi-Scale Features (~0.15 page)

PVTv2-B4 → features at 1/4, 1/8, 1/16 scales, each projected to $C=256$ via 1×1 conv.

#### 3.3 Spatial Token-Level Router (~0.3 page)

- Router input: concatenate token features with DWConv3×3 local context → MLP → logits.
- Noisy top-$K$ selection (Shazeer-style).
- True sparse dispatch: loop over experts, gate-weighted sum.
- **Eq. 1** and **Eq. 2** (entropy).

#### 3.4 Expert Pool (~0.15 page)

8 experts per scale (24 total). Each expert: LayerNorm → Linear($C$, $4C$) → GELU → Linear($4C$, $C$) → residual. Independent per scale.

#### 3.5 Multi-Scale Decoder with Entropy Fusion (~0.3 page)

- Per-scale EntropyFusionBlock (**Eq. 3**).
- Top-down cross-attention: global (1/16→1/8), windowed (1/8→1/4).
- Refinement blocks → 2× upsample × 2 → dual heads (saliency + boundary).

#### 3.6 Training Objective (~0.2 page)

**Eq. 4** (total loss). BCE + soft IoU + load-balance + importance. Explain each term briefly. Note: deep supervision and boundary loss are OFF in the reported configuration.

### 4. Experimental Setup (~0.3 page)

- **Dataset:** WXSOD. train_sys (12,891) → scene-aware 80/20 split. test_sys (1,500 synthetic), test_real (554 real-world). 5 real-world weather categories, 9 synthetic.
- **Implementation:** PyTorch, DDP (2 GPUs), AMP FP16, AdamW (lr=1e-4), WarmupCosine (50 epochs, warmup 1%), gradient accumulation (effective batch=32), gradient clipping (1.0). Backbone frozen for epoch 0.
- **Metrics:** MAE, S_measure, E_measure (adaptive/mean/max), F_measure (adaptive/mean/max), boundary MAE, boundary F1.
- **Compute:** 66.27M parameters, 278.2G MACs.

### 5. Results and Analysis (~0.8 pages)

#### 5.1 Main Results (~0.3 page)

**Table 1:** Global results on test_sys and test_real. Report MAE, S, F_max, E_max.

- test_sys: MAE=0.0192, S=0.9139, F_max=0.9015
- test_real: MAE=0.0168, S=0.9151, F_max=0.8936

Observation: Real-world MAE is comparable to or slightly better than synthetic. Possible explanation: synthetic test set contains harder compound weather conditions (rainafog, rainasnow, snowafog) not present in real-world test.

#### 5.2 Weather-Wise Analysis (~0.25 page)

**Table 2:** Weather-wise breakdown on test_real.

- Snow: MAE=0.0094 (easiest). High contrast between white snow and darker objects.
- Low-light: MAE=0.0243 (hardest). 2.6× range across conditions.
- Compound weather (rainafog): MAE=0.0237 worst on synthetic.

**Fig. 2:** Bar chart visualization.

**Fig. 3:** Qualitative examples.

#### 5.3 Routing Behavior (~0.15 page)

- Mean entropy 0.68–0.69 (normalized), indicating moderate routing uncertainty.
- Scale 1/16 slightly lower entropy (more confident) than finer scales.
- Synthetic vs real entropy nearly identical (<0.003 difference).

#### 5.4 Forced-Expert Analysis (~0.1 page)

**Table 3 + Fig. 4:** Forcing all tokens at scale 1/4 to expert 0 causes ΔMAE = +0.0002 (+1.2%). Only 1 of 3 scales affected. This marginal degradation has two interpretations: (a) expert pool is over-parameterized, or (b) experts learn similar functions. Per-expert analysis is needed to distinguish these.

### 6. Conclusion (~0.15 page)

- We present SpatialMoE-SOD, a token-level spatial mixture-of-experts for adverse-weather SOD.
- The architecture routes spatial tokens to specialized experts at three pyramid scales without weather labels.
- Routing entropy is fused as an explicit decoder feature.
- The model achieves MAE 0.0192 on synthetic and 0.0168 on real-world weather conditions across the WXSOD benchmark.
- Limitations: no ablation studies completed, no SOTA comparison, single training run, no per-expert specialization analysis.
- Future work: ablation of routing, entropy fusion, and multi-scale independence; SOTA comparison; expert interpretability analysis.

### References (~0.3 page)

~25–30 references. Key citations:
- WXSOD (DB-34), NIFM (DB-35), WFANet (DB-34 baseline)
- WM-MoE (DB-19), MoFME (DB-20), Complexity Experts (DB-29)
- V-MoE (DB-41), Soft MoE (DB-42), M³ViT (DB-43)
- MMSOD (DB-01), CMFNet (DB-03), PSOD (DB-06)
- PVTv2 (timm), BCE/IoU losses

---

## Writing Rules

1. **Define every variable before first use.** No undefined symbols.
2. **State equations in compact form.** Use operator notation, not expanded code.
3. **Distinguish measurements from interpretation.** "We observe X" ≠ "X demonstrates Y."
4. **No causal claims without ablation.** Present design choices as architectural decisions, not validated contributions.
5. **Cite source code line numbers** in the supplement or internal review, not in the paper body.
6. **Tables use consistent formatting.** Bold best values. Report 4 decimal places for MAE, 4 for S/F/E.
7. **Figures must be self-contained.** Axis labels, legends, and captions explain the figure without reading the text.

---

## LaTeX Template Notes

- No SCOVA template found locally. Use standard IEEE/CVPR single-column or two-column format as appropriate.
- Target: 4 pages maximum, including references.
- Use `\small` or `\footnotesize` for tables if needed.
- Figures: 1 column width for bar charts, 2 column width for architecture diagram.

---

*This plan is a structural blueprint. Actual prose will be written separately.*
