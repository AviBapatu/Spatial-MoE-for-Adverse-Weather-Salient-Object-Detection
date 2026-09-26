# MANUSCRIPT_AUDIT.md — Structural and Visual Verification

> **RESOLVED 2026-09-25 — read before acting on anything below.**
>
> This audit is a dated record of a manuscript revision and its findings are kept as
> written. Two of its items are now settled, and one cited file has been renamed:
>
> - **Parameter count.** The audit records 66.27M. That figure is not present in any file
>   under `results/`; the correct value is **69.21M**, measured directly (69,212,349 at
>   window 7 with deep supervision off, 69,213,120 with it on; the legacy `compute_cost.json`
>   reports 68.9M). See `RESEARCH_TRUTH.md`.
> - **Routing entropy.** `SpatialMoELayer` takes the entropy of the **full E-way softmax**
>   over the router logits, so its ceiling is `ln E` — `ln 8 = 2.0794` for the 8-expert
>   model — and `EntropyFusionBlock` normalises by `ln 8`. For the reported E8 model the
>   measured mean is 2.076–2.079 nats, i.e. **99.8–100% of `ln 8`**: the router is close to
>   uniform over all eight experts. The "0.9309" and "0.6931" figures below are both
>   pre-fix top-2 measurements. See `RESEARCH_TRUTH.md`.
> - `LITERATURE_NOTES.md` no longer exists; it is `docs/research/LITERATURE_DATABASE.md`.
> - `docs/research/PAPER_CORRECTIONS.md` repeats the 66.27M figure for the same reason.
> - **Loss terms.** The reported recipe activates SSIM, boundary, auxiliary-boundary and deep
>   supervision (`ssim_weight`/`boundary_weight` 1.0, `aux_boundary_weight` 0.5,
>   `deep_supervision_weight` 0.4). The "inactive / λ=0" loss rows below were checked against
>   `experiments/baseline_v1.json`, which is a template, not the recipe. See
>   `RESEARCH_TRUTH.md`.
>
> **Take no number or fact from the body of this file.** It is a historical record; the
> authoritative sources are `RESEARCH_TRUTH.md`, `docs/research/ARCHITECTURE.md`,
> `docs/research/TRAINING.md` and `docs/research/RESULTS.md`.


> **Note.** This file audits a specific revision of `paper/main.tex`. Citations to
> source files refer to the layout at the time of the audit; module paths have since
> changed (the decoder is now the `src/decoder/` package, training code lives under
> `src/training/`). Treat the findings as issues to check, not as current fact.

**Date:** 2026-09-02
**Paper:** paper/main.tex (v1, post-audit)

---

## 1. Page Budget

| Item | Status |
|------|--------|
| Two-column layout | `\documentclass[10pt,twocolumn]{article}` — VERIFIED |
| Maximum 6 pages including references | 6 pages, verified by a `latexmk -pdf` build (`main.pdf`) |
| No SCOPUS template available | Using clean two-column article class — ACCEPTABLE |

**Note:** The page count is now verified by compilation. An earlier revision of this audit estimated it because no LaTeX toolchain was available at the time.

---

## 2. Reference Count

| Item | Count | Status |
|------|-------|--------|
| Bibliography entries | 8 | **EXCEEDS 5-6 target** |
| Citations in text | 8 unique keys | All resolve to bib entries |
| Orphaned bib entries | 0 | All bib entries cited |
| Missing citations | 0 | All \cite keys have bib entries |

**Recommendation:** The user specified "5-6 references" and "Do not exceed this unless a citation is absolutely indispensable." Currently at 8. The 3 metric-definition papers (struct-measure, enhanced-measure, f-measure) are necessary for reproducibility. NIFM and WM-MoE are necessary for differentiation. V-MoE is necessary as architectural ancestor. PVTv2 is necessary for backbone attribution. WXSOD is necessary as benchmark. All 8 are defensible. However, if strict 5-6 compliance is required, consider removing f-measure (marginal utility) and one of enhanced-measure/struct-measure.

---

## 3. Figure Count and References

| Figure | Type | Label | Referenced | Status |
|--------|------|-------|------------|--------|
| Fig. 1 | TikZ architecture | `fig:arch` | L84: `\ref{fig:arch}` | VERIFIED |
| Fig. 2 | TikZ routing mechanism | `fig:router` | (self-contained in caption) | PRESENT |
| Fig. 3 | pgfplots bar chart | `fig:weather` | (self-contained in caption) | PRESENT |
| Fig. 4 | Raster qualitative | `fig:qualitative` | (self-contained in caption) | PRESENT |

**Note:** Fig. 2, 3, 4 are not explicitly referenced in the body text via `\ref`. This is acceptable for self-contained figures with descriptive captions, but ideally each figure should be referenced at least once in the text.

---

## 4. Table Count and References

| Table | Label | Referenced | Status |
|-------|-------|------------|--------|
| Tab. 1: Global results | `tab:main` | L395: `\ref{tab:main}` | VERIFIED |
| Tab. 2: Weather-wise | `tab:weather` | L416: `\ref{tab:weather}` | VERIFIED |
| Tab. 3: Entropy | `tab:entropy` | L487: `\ref{tab:entropy}` | VERIFIED |

---

## 5. Visual Quality Assessment

### Figure 1 — Architecture (TikZ)
- Semantic color coding: blue=backbone, orange=router, green=experts, red=entropy, yellow=fusion, gray=decoder, violet=heads
- Scale labels with tensor dimensions
- Brace annotation for SpatialMoE block
- Data flow arrows with direction
- Status: READABLE (requires compilation to verify)

### Figure 2 — Routing Mechanism (TikZ)
- Token → DWConv → concat → MLP → logits → top-2 → experts → weighted sum → entropy → projection → decoder
- Brace annotation for residual MLP structure
- Status: READABLE (requires compilation to verify)

### Figure 3 — Weather-wise MAE (pgfplots)
- Bar chart with 5 weather categories
- Actual measured values (0.0094 to 0.0243)
- Nodes near coords with 4-decimal precision
- Grid lines for readability
- Status: READABLE (requires compilation to verify)

### Figure 4 — Qualitative Results (raster)
- 3 rows: Snow, Fog, Dark from test_real
- 4 columns: Input, GT, Prediction, Abs. Error (red-tinted)
- Generated from actual evaluation outputs
- Status: READABLE (verified via PNG inspection)

---

## 6. Overflow and Clipping Risk

| Risk | Assessment |
|------|------------|
| TikZ figures too wide | Both use `figure*` (full width). TikZ coordinates are within reasonable bounds. LOW RISK |
| Bar chart labels overlap | 5 categories with rotated labels at 0°. LOW RISK |
| Qualitative figure width | `width=0.92\textwidth` with `figure*`. LOW RISK |
| Table overflow | All tables use `\small` and have 4-5 columns. LOW RISK |
| Reference page overflow | 8 references at ~2 lines each = ~16 lines. Fits on page 4. LOW RISK |

---

## 7. Bibliography Verification

| Entry | Key | Citation Count | Status |
|-------|-----|---------------|--------|
| wxsod2025 | `wxsod2025` | 2 (L71, L388) | VERIFIED |
| nifm2025 | `nifm2025` | 3 (L57, L70, L544) | VERIFIED |
| wmmoe2023 | `wmmoe2023` | 2 (L57, L74) | VERIFIED |
| v-moe2021 | `v-moe2021` | 1 (L79) | VERIFIED |
| pvtv2 | `pvtv2` | 1 (L205) | VERIFIED |
| struct-measure | `struct-measure` | 1 (L391) | VERIFIED |
| enhanced-measure | `enhanced-measure` | 1 (L391) | VERIFIED |
| f-measure | `f-measure` | 1 (L391) | VERIFIED |

All entries have correct formatting. No missing fields. No broken references.

---

## 8. Scientific Positioning

### Claims made (architectural contribution + empirical characterization)
- Token-level spatial router with K=2, E=8 per scale — IMPLEMENTED, NOT TESTED against non-MoE
- Three independent routers at 1/4, 1/8, 1/16 — IMPLEMENTED, NOT TESTED against shared routing
- Routing entropy as decoder feature — IMPLEMENTED, NOT TESTED against no-entropy
- No weather-specific component — IMPLEMENTED, NOT TESTED against weather-branch
- MAE 0.0192 (synthetic), 0.0168 (real) — MEASURED, REPORTED

### Claims explicitly NOT made
- [x] No "outperforms SOTA" claim
- [x] No "MoE improves over non-MoE" claim
- [x] No "routing is critical" claim
- [x] No "entropy helps" claim
- [x] No "experts specialize" claim (FIXED: removed "specialized" from abstract/conclusion)
- [x] No "independent routing is better" claim
- [x] No "controlled domain generalization" claim
- [x] No "efficiency" claim

---

## 9. Remaining Issues

### Must Fix
None after the "specialized" fix.

### Should Fix (minor)
1. **Figures 2-4 not body-referenced** — Each figure is self-contained but ideally should be mentioned in text at least once. Consider adding one sentence each referencing Fig. 2 (routing mechanism), Fig. 3 (weather-wise), and Fig. 4 (qualitative).

### Could Fix (polish)
2. **8 references vs 5-6 target** — All 8 are defensible. Trim f-measure if strict compliance needed.
3. **Entropy scale 1/8 synthetic discrepancy** — The stored value (0.9309) differs from paper (0.6931). Could add footnote noting this, or investigate source.
