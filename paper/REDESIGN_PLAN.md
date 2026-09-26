# REDESIGN_PLAN.md — SCOVA Conference Paper Redesign (6-page budget)

**Date:** 2026-09-02
**Status:** Applied — `paper/main.tex` is the redesigned 6-page manuscript; this file is the record of the plan behind it.
**Target:** 6 pages MAXIMUM including references, two-column, English
**Style:** Physics-like scientific exposition, mathematically rigorous, visually sophisticated

---

## 1. Scientific Corrections Found

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | The loss description was checked against the `baseline_v1` template and is wrong: the reported recipe activates SSIM, boundary, auxiliary-boundary and deep supervision (weights 1.0 / 1.0 / 0.5 / 0.4), with only the Z-loss and router-confidence term off | HIGH | State the eight active terms and their weights. `paper/main.tex` §III-G now does this. |
| 2 | Current paper cites 15 references — more than a 6-page paper needs | HIGH | Trim to the 8 entries in `paper/references.bib`. See §5. |
| 3 | The pre-redesign paper had 3 tables | — | The manuscript keeps main, weather-wise and entropy tables (3 total); the forced-expert table is folded into text |
| 4 | Current paper has only 1 figure (architecture) — weak visual communication | HIGH | Add 2–3 more figures: routing mechanism zoom, qualitative predictions, weather-wise chart. |
| 5 | Architecture figure is TikZ but too simple — doesn't show data flow clearly | MEDIUM | Redesign with proper tensor dimensions, data flow arrows, and scale labels. |
| 6 | Pre-redesign paper discussed limitations in the main body — wasted space | MEDIUM | Fold limitations into §IV-F Discussion as a compact paragraph. |
| 7 | Pre-redesign paper repeats MAE values in abstract, results and conclusion | LOW | State once in the abstract, reference the table elsewhere. |
| 8 | No qualitative figure showing actual predictions | HIGH | Add Fig. 4 with actual evaluation PNG outputs. |

---

## 2. Layout Problems in the Pre-Redesign Manuscript

**Status:** these describe the 4-page v1 manuscript this plan replaced; `paper/main.tex` is the post-redesign version.

| Problem | Impact | Solution |
|---------|--------|----------|
| References dominate the last page (~0.5 page of bib) | Wastes 25% of the last page | Trim to 8 references |
| Only 1 figure in entire paper | Visually weak, doesn't communicate architecture effectively | Add 3 figures (routing mechanism, weather chart, qualitative) |
| Discussion section is verbose (~0.3 page) | Could be compressed | Compress to 1 paragraph (the manuscript keeps it as §IV-F) |
| Conclusion is ~0.2 page with repeated content | Redundant | Compress to 3–4 sentences |
| No actual prediction visualizations | No visual evidence of model output | Add qualitative figure using evaluation PNGs (Fig. 4) |

---

## 3. Final Figure Plan

### FIGURE 1 — Full-Width Hero Architecture Diagram (Page 2, full width)

**Purpose:** Communicate the complete pipeline visually.
**Content:**
```
RGB Input [3, 384, 384]
    ↓
PVTv2-B4 Backbone
    ↓ (three arrows)
[256, 96, 96]  [256, 48, 48]  [256, 24, 24]
    ↓               ↓               ↓
SpatialMoE       SpatialMoE       SpatialMoE
(8 experts,      (8 experts,      (8 experts,
 top-2 routing)   top-2 routing)   top-2 routing)
    ↓   ↓           ↓   ↓           ↓   ↓
  Y_4  H_4       Y_8  H_8       Y_16 H_16
    ↓               ↓               ↓
  EntropyFusion   EntropyFusion   EntropyFusion
    ↓               ↓               ↓
    └─────── Cross-Attention Decoder ───────┘
                    ↓
            ┌───────┴───────┐
            ↓               ↓
      Saliency Head    Boundary Head
      [1, 384, 384]    [1, 384, 384]
```

**Style:** Clean boxes, color-coded (blue=backbone, orange=router, green=expert, red=entropy, gray=decoder, purple=heads). Include tensor dims at key points. No decorative elements.

**Source:** Derived from `src/model.py`, `src/backbone.py`, `src/moe_layer.py`, `src/decoder/`.

### FIGURE 2 — Zoomed Spatial-MoE Routing Mechanism (Page 2, half width or page 3)

**Purpose:** Show how a single token gets routed.
**Content:**
```
Token x_i ∈ R^256
    ↓
DWConv3x3 → local context ∈ R^256
    ↓
Concat [x_i || local] ∈ R^512
    ↓
MLP (512→128→8) → clean logits h_i ∈ R^8
    ↓ (+ noise during training)
Noisy logits → Top-2 selection
    ↓
Softmax over top-2 → gates g_{i,1}, g_{i,2}
    ↓
Expert k1(x_i) × g_{i,1} + Expert k2(x_i) × g_{i,2}
    ↓
Output y_i + entropy H_i
```

**Style:** Compact flow diagram with math annotations. Include the entropy formula as an inset equation.

### FIGURE 3 — Weather-Wise Quantitative Visualization (Page 4, half width)

**Purpose:** Compact visual summary of performance across weather conditions.
**Content:** Horizontal bar chart showing MAE for 5 real-world weather categories (test_real):
- Snow: 0.0094
- Fog: 0.0148
- Rain: 0.0165
- Dark: 0.0191
- Low-light: 0.0243

**Style:** Clean bar chart with MAE values annotated. Color gradient from green (low MAE) to red (high MAE). Include sample counts (N) on bars.

**Alternative (if bar chart doesn't fit):** A compact table replacement — but the bar chart is preferred for visual impact.

### FIGURE 4 — Qualitative Predictions (Page 6, full width)

**Purpose:** Visual evidence of model output across weather conditions.
**Content:** 2–3 rows from test_real, columns:
1. Input image
2. Ground truth
3. Model prediction
4. Absolute-error map (red-tinted)

**Rows selected from actual evaluation outputs:**
- Row 1: Snow (best MAE=0.0094) — `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/` PNG
- Row 2: Fog (mid MAE=0.0148)
- Row 3: Low-light (worst MAE=0.0243)

**Source:** Use actual PNG files from `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/*.png` matched to input images.

**Note:** This full-width figure is placed after the reference list, so it lands on the last page.

### FIGURES NOT INCLUDED (and why)

| Candidate | Reason for exclusion |
|-----------|---------------------|
| Forced-expert ablation chart | Marginal result (ΔMAE=+0.9%); better as text callout than a full figure |
| Routing entropy visualization | Low information density; numbers can be stated in 1 sentence |
| Training curves | Not available in repository |
| t-SNE of expert embeddings | Not computed |

---

## 4. Final Table Plan

### TABLE 1 — Main Results (Page 3, full width)

| Set | N | MAE ↓ | S_ϕ ↑ | F_β^max ↑ | E_ϕ^adp ↑ |
|-----|---|-------|-------|-----------|-----------|
| Synthetic | 1,500 | 0.0192 | 0.9139 | 0.9015 | 0.9591 |
| Real-world | 554 | 0.0168 | 0.9151 | 0.8936 | 0.9530 |

**Caption:** Global performance on WXSOD test splits.

### TABLE 2 — Weather-Wise Results (Page 4, half width, beside Fig. 3)

| Weather | N | MAE ↓ | S_ϕ ↑ | F_β^max ↑ |
|---------|---|-------|-------|-----------|
| Snow | 90 | 0.0094 | 0.9478 | 0.9421 |
| Fog | 126 | 0.0148 | 0.9179 | 0.8914 |
| Rain | 120 | 0.0165 | 0.9157 | 0.8935 |
| Dark | 125 | 0.0191 | 0.9072 | 0.8885 |
| Low-light | 93 | 0.0243 | 0.8897 | 0.8629 |

**Caption:** Weather-wise results on real-world test (554 images).

### TABLES KEPT / NOT PRESENT

| Table | Decision |
|-------|----------|
| Entropy table (Table 3) | **Kept** in `paper/main.tex`: "Mean routing entropy per scale. Entropy is over the full distribution over all $E{=}8$ experts, so the maximum is $\log 8 = 2.0794$ nats." |
| Forced-expert table | Not present; a text callout instead: "Forcing all tokens at scale 1/4 to expert 0 increased MAE by +0.0002 (+0.9%)." |

---

## 5. Final Reference List (8 entries)

| # | Key | Citation | Why Essential |
|---|-----|----------|---------------|
| 1 | `wxsod2025` | Chen et al., Pattern Recognition 2026 | The benchmark. Must cite. |
| 2 | `nifm2025` | Chen et al., arXiv 2025 | Direct competitor (weather-label SOD) |
| 3 | `wmmoe2023` | Luo et al., arXiv 2023 (WM-MoE) | Closest MoE weather-restoration work. Must distinguish from. |
| 4 | `v-moe2021` | Riquelme et al., NeurIPS 2021 | Foundational token-level MoE routing |
| 5 | `pvtv2` | Wang et al., CVM 2022 | Backbone architecture |
| 6 | `struct-measure` | Cheng et al., ICCV 2017 | S_ϕ metric definition |
| 7 | `enhanced-measure` | Fan et al., IJCAI 2016 | E_ϕ metric definition |
| 8 | `f-measure` | Margolin et al., arXiv 2014 | F_β metric definition |

These eight are exactly the keys in `paper/references.bib`, and all eight are cited in `paper/main.tex`.

**Absent from `paper/references.bib`** — do not cite without adding the entry:
- `wfynet2025` (WFANet)
- `complexity2025` (Complexity Experts)
- `mmsod2025`, `cmfnet2026`, `cmoe2026`, `psod2025`, `soft-moe2024`, `mofme2024` (considered and dropped)

---

## 6. Proposed Page Allocation

### PAGE 1 (~55 lines)

| Section | Content | Approx. Lines |
|---------|---------|---------------|
| Abstract | 100 words. Problem, approach, key mechanism, main results. | 6 |
| §1 Introduction | 2 paragraphs: (1) problem + prior work gap, (2) our approach + 3 contributions. | 12 |
| §2 Related Work | 1 dense paragraph, citation-dense. Adverse-weather SOD, MoE for restoration, MoE for SOD, token-level MoE. | 8 |
| §3 Method §3.1–§3.3 | Overview, backbone, router (with Eq. 1–2). | 20 |
| §3 Method §3.4 | Expert pool + sparse dispatch (Eq. 3). | 9 |

**Page 1 total:** ~55 lines. Dense, no figures on this page.

### PAGE 2 (~55 lines)

| Section | Content | Approx. Lines |
|---------|---------|---------------|
| §3 Method §3.5 | Decoder with entropy fusion (Eq. 4–5). | 12 |
| §3 Method §3.6 | Training objective (Eq. 6). | 8 |
| **FIGURE 1** | Full-width architecture diagram. | ~12 (including caption) |
| §4 Experimental Setup | Dataset, implementation, metrics, compute. | 12 |
| §5 Results §5.1 | Main results text + Table 1. | 11 |

**Page 2 total:** ~55 lines. Figure 1 dominates the visual space.

### PAGE 3 (~55 lines)

| Section | Content | Approx. Lines |
|---------|---------|---------------|
| §IV-B cont. | Brief observation on synthetic vs real. | 3 |
| §IV-C | Weather-wise analysis text + Table 2. | 10 |
| §IV-D | Routing entropy (text + Table 3). | 4 |
| §IV-E | Forced-expert analysis (text callout, no table). | 6 |

**Page 3 total:** ~23 lines plus Table 2 and the surrounding floats.

### PAGE 4 (~55 lines)

| Section | Content | Approx. Lines |
|---------|---------|---------------|
| §IV-F | Discussion (compact paragraph, includes limitations). | 5 |
| §V | Conclusion, 3–4 sentences. | 5 |
| **FIGURE 3** | Weather-wise bar chart (half width). | ~8 |
| (Remaining space) | Float slack for the figures still drifting above. | ~35 |

**Page 4 total:** the discussion, the conclusion and the weather-wise figure.

### PAGE 5 (~55 lines)

| Section | Content | Approx. Lines |
|---------|---------|---------------|
| References | 8 entries. | ~20 |
| (Remaining space) | Blank or minimal. | ~30 |

**Page 5 total:** ~20 lines of references.

### PAGE 6 (~55 lines)

| Section | Content | Approx. Lines |
|---------|---------|---------------|
| **FIGURE 4** | Qualitative predictions grid (full width), placed after the reference list. | ~12 (including caption) |

**Page 6 total:** the trailing full-width figure.

**The key constraint is MAXIMUM 6 pages, not EXACTLY 6 pages of content.** References naturally fall on the penultimate page and the trailing full-width figure on the last.

---

## 7. Mathematical Equations

| Eq. | Content | Compact Form |
|-----|---------|--------------|
| 1 | Router input | r_i = [x_i ∥ DWConv₃ₓ₃(X)_i] ∈ ℝ^{2C} |
| 2 | Routing logits | h_i = W₂·GELU(W₁r_i + b₁) + b₂ ∈ ℝ^E |
| 3 | Noisy routing + top-K | T_i = topk(h_i + ε_i, K=2), g_{i,k} = softmax over T_i |
| 4 | Sparse dispatch | y_i = Σ_{k∈T_i} g_{i,k} · E_k(x_i) |
| 5 | Routing entropy | H_i = −Σ_{j=1}^{E} p_{i,j} log p_{i,j}, Ĥ_i = H_i / log 8 |
| 6 | Entropy fusion | F_s = Proj_Y(Y_s) + λ_s · Proj_H(Ĥ_s) |
| 7 | Total loss | L = L_BCE + λ_IoU·L_IoU + λ_LB·L_LB + λ_IMP·L_IMP |

All variables defined before first use. No undefined symbols.

---

## 8. Content to Remove/Compress

| Current Content | Action | Rationale |
|----------------|--------|-----------|
| §2 Related Work (3 subsections) | Compress to 1 dense paragraph | Save ~10 lines |
| §5 Discussion paragraph | Merge into §5.5 Limitations | Save ~5 lines |
| §6 Conclusion (2 paragraphs) | Compress to 3–4 sentences | Save ~8 lines |
| Table 3 (entropy) | Remove, fold 2 numbers into text | Save ~10 lines |
| Forced-expert table | Remove, replace with 1-sentence callout | Save ~8 lines |
| Verbose loss explanation | Compact to equation + 1 sentence each | Save ~5 lines |
| Redundant MAE restatements | State once, reference table | Save ~3 lines |

**Total space recovered:** ~39 lines ≈ 0.7 pages of two-column text.

---

## 9. Visual Design Rules

1. **No decorative figures.** Every figure communicates scientific information.
2. **Figures are self-contained.** Axis labels, legends, and captions explain without reading text.
3. **Color palette:** Consistent across figures. Blue=backbone, orange=router, green=expert, red=entropy, gray=decoder, purple=heads.
4. **Font sizes:** Use `\small` or `\footnotesize` for tables. Ensure readability at two-column scale.
5. **Figure placement:** Use `[t]` or `[b]` placement. Avoid `[h]` which causes layout instability.
6. **No full-page figures.** Each figure occupies at most 50% of a page height.
7. **Tables use `\toprule`, `\midrule`, `\bottomrule`** (booktabs). No vertical rules.
8. **Bold best values** in tables. Use arrows (↓/↑) in column headers.

---

## 10. LaTeX Template Notes

- Use `\documentclass[10pt,twocolumn]{article}` (already in place)
- Use `\usepackage[margin=0.6in]{geometry}` (already in place)
- Ensure total page count via `pdflatex` → `pdflatex` → `bibtex` → `pdflatex` × 2
- Verify page count is ≤ 6 before submission
- Include `\usepackage{hyperref}` for clickable references (already present)

---

*This plan is a structural blueprint. It defines WHAT to write and WHERE, not the actual prose. The manuscript will be written separately following this plan.*
