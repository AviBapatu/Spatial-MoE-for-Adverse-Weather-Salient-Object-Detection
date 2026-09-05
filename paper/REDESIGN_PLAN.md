# REDESIGN_PLAN.md — 4-Page SCOVA Conference Paper Redesign

**Date:** 2026-09-02
**Target:** 4 pages MAXIMUM including references, two-column, English
**Style:** Physics-like scientific exposition, mathematically rigorous, visually sophisticated

---

## 1. Scientific Corrections Found

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | Current paper says "four active terms" in loss — correct, but doesn't clarify deep supervision is OFF | LOW | Add one sentence: "Deep supervision, boundary, and SSIM losses are defined but inactive (λ=0) in the reported configuration." Already present in current paper §3. No change needed. |
| 2 | Current paper cites 15 references — too many for a 4-page paper | HIGH | Trim to ~6–8 essential references. See §8. |
| 3 | Current paper has 3 tables — too many, wastes space | HIGH | Reduce to 1 main table + 1 weather-wise table. Remove entropy table (fold into text). Remove forced-expert table (make text callout). |
| 4 | Current paper has only 1 figure (architecture) — weak visual communication | HIGH | Add 2–3 more figures: routing mechanism zoom, qualitative predictions, weather-wise chart. |
| 5 | Architecture figure is TikZ but too simple — doesn't show data flow clearly | MEDIUM | Redesign with proper tensor dimensions, data flow arrows, and scale labels. |
| 6 | Current paper discusses limitations in main body — wastes space | MEDIUM | Move limitations to a compact "Limitations" paragraph at end of §5 or §6. Don't dedicate separate paragraphs. |
| 7 | Current paper repeats MAE values in abstract, §4, §5.1, and §6 | LOW | State once in abstract, reference table elsewhere. |
| 8 | No qualitative figure showing actual predictions | HIGH | Add Fig. 3 with actual evaluation PNG outputs. |

---

## 2. Current Layout Problems

| Problem | Impact | Solution |
|---------|--------|----------|
| References dominate page 4 (~0.5 page of bib) | Wastes 25% of last page | Trim to 6–8 references |
| Only 1 figure in entire paper | Visually weak, doesn't communicate architecture effectively | Add 3 figures (routing zoom, qualitative, weather chart) |
| 3 tables (main, weather, entropy) | Redundant; entropy table adds little | Merge into 2 tables max |
| Discussion section is verbose (~0.3 page) | Could be compressed | Compress to 1 paragraph |
| Conclusion is ~0.2 page with repeated content | Redundant | Compress to 3–4 sentences |
| No actual prediction visualizations | No visual evidence of model output | Add qualitative figure using evaluation PNGs |
| Entropy table is standalone | Low information density | Fold key numbers into text or small inset |

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

**Source:** Derived from `src/model.py`, `src/backbone.py`, `src/moe_layer.py`, `src/decoder.py`.

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

### FIGURE 3 — Qualitative Predictions (Page 3, full width)

**Purpose:** Visual evidence of model output across weather conditions.
**Content:** 2–3 rows from test_real, columns:
1. Input image
2. Ground truth
3. Model prediction
4. Entropy map (scale 1/8, upsampled)

**Rows selected from actual evaluation outputs:**
- Row 1: Snow (best MAE=0.0094) — `evaluation/best_new_1/test_real/none/` PNG
- Row 2: Fog (mid MAE=0.0148)
- Row 3: Low-light (worst MAE=0.0243)

**Source:** Use actual PNG files from `evaluation/best_new_1/test_real/none/*.png` matched to input images.

**Note:** If entropy maps are not saved as separate files, this can be replaced with a simple 2-column (input vs prediction) qualitative grid. The key is showing actual model outputs, not synthetic illustrations.

### FIGURE 4 — Weather-Wise Quantitative Visualization (Page 3, half width)

**Purpose:** Compact visual summary of performance across weather conditions.
**Content:** Horizontal bar chart showing MAE for 5 real-world weather categories (test_real):
- Snow: 0.0094
- Fog: 0.0148
- Rain: 0.0165
- Dark: 0.0191
- Low-light: 0.0243

**Style:** Clean bar chart with MAE values annotated. Color gradient from green (low MAE) to red (high MAE). Include sample counts (N) on bars.

**Alternative (if bar chart doesn't fit):** A compact table replacement — but the bar chart is preferred for visual impact.

### FIGURES NOT INCLUDED (and why)

| Candidate | Reason for exclusion |
|-----------|---------------------|
| Forced-expert ablation chart | Marginal result (ΔMAE=+1.2%); better as text callout than a full figure |
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

### TABLE 2 — Weather-Wise Results (Page 3, half width, beside Fig. 4)

| Weather | N | MAE ↓ | S_ϕ ↑ | F_β^max ↑ |
|---------|---|-------|-------|-----------|
| Snow | 90 | 0.0094 | 0.9478 | 0.9421 |
| Fog | 126 | 0.0148 | 0.9179 | 0.8914 |
| Rain | 120 | 0.0165 | 0.9157 | 0.8935 |
| Dark | 125 | 0.0191 | 0.9072 | 0.8885 |
| Low-light | 93 | 0.0243 | 0.8897 | 0.8629 |

**Caption:** Weather-wise results on real-world test (554 images).

### TABLES REMOVED

| Removed Table | Reason |
|---------------|--------|
| Entropy table (Table 3 in current paper) | Fold into text: "Mean normalized entropy ranges from 0.68–0.69 across scales, with scale 1/16 slightly more confident (0.68)." One sentence. |
| Forced-expert table | Replace with text callout: "Forcing all tokens at scale 1/4 to expert 0 increased MAE by +0.0002 (+1.2%)." No table needed. |

---

## 5. Final Reference Shortlist (6–8 essential)

| # | Key | Citation | Why Essential |
|---|-----|----------|---------------|
| 1 | `wxsod2025` | Chen et al., Pattern Recognition 2026 | The benchmark. Must cite. |
| 2 | `nifm2025` | Chen et al., arXiv 2025 | Direct competitor (weather-label SOD) |
| 3 | `wfynet2025` | Chen et al., 2025 | Direct competitor (weather-label SOD) |
| 4 | `v-moe2021` | Riquelme et al., NeurIPS 2021 | Foundational token-level MoE routing |
| 5 | `pvtv2` | Wang et al., CVM 2022 | Backbone architecture |
| 6 | `struct-measure` | Cheng et al., ICCV 2017 | S_ϕ metric definition |
| 7 | `enhanced-measure` | Fan et al., IJCAI 2016 | E_ϕ metric definition |
| 8 | `f-measure` | Margolin et al., arXiv 2014 | F_β metric definition |

**Optional (if space permits):**
- `wmmoe2023` (WM-MoE) — closest MoE weather restoration work
- `complexity2025` (Complexity Experts) — recent SOTA MoE restoration

**Removed from current 15:**
- `mmsod2025` (MMSOD) — multi-modal SOD, less relevant
- `cmfnet2026` (CMFNet) — RGB-D SOD, less relevant
- `cmoe2026` (CMoE) — modality-missing SOD, less relevant
- `psod2025` (PSOD) — pluralistic SOD, less relevant
- `soft-moe2024` (Soft MoE) — interesting but not cited in main claims
- `mofme2024` (MoFME) — less directly relevant

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
| §5 Results §5.1 cont. | Brief observation on synthetic vs real. | 3 |
| §5 Results §5.2 | Weather-wise analysis text + Table 2. | 10 |
| **FIGURE 4** | Weather-wise bar chart (half width). | ~8 |
| **FIGURE 3** | Qualitative predictions grid (full width). | ~12 (including caption) |
| §5 Results §5.3 | Routing entropy (text only, 2–3 sentences). | 4 |
| §5 Results §5.4 | Forced-expert analysis (text callout, no table). | 6 |
| §5 Results §5.5 | Limitations (compact paragraph). | 5 |
| §6 Conclusion | 3–4 sentences. | 5 |

**Page 3 total:** ~53 lines. Two figures + one table.

### PAGE 4 (~55 lines)

| Section | Content | Approx. Lines |
|---------|---------|---------------|
| References | 6–8 entries. | ~20 |
| (Remaining space) | Blank or minimal. | ~35 |

**Page 4 total:** ~20 lines of references, rest is whitespace. This is acceptable for a 4-page paper — references naturally fall on the last page.

**Alternatively:** If references fit on page 3 (with compression), page 4 can be entirely blank, which is also acceptable. The key constraint is MAXIMUM 4 pages, not EXACTLY 4 pages of content.

---

## 7. Mathematical Equations

| Eq. | Content | Compact Form |
|-----|---------|--------------|
| 1 | Router input | r_i = [x_i ∥ DWConv₃ₓ₃(X)_i] ∈ ℝ^{2C} |
| 2 | Routing logits | h_i = W₂·GELU(W₁r_i + b₁) + b₂ ∈ ℝ^E |
| 3 | Noisy routing + top-K | T_i = topk(h_i + ε_i, K=2), g_{i,k} = softmax over T_i |
| 4 | Sparse dispatch | y_i = Σ_{k∈T_i} g_{i,k} · E_k(x_i) |
| 5 | Routing entropy | H_i = −Σ_{k∈T_i} g_{i,k} log g_{i,k}, Ĥ_i = H_i / log 2 |
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
- Verify page count is ≤ 4 before submission
- Include `\usepackage{hyperref}` for clickable references (already present)

---

*This plan is a structural blueprint. It defines WHAT to write and WHERE, not the actual prose. The manuscript will be written separately following this plan.*
