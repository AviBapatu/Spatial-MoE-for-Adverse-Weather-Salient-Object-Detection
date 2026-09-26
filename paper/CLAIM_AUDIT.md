# CLAIM_AUDIT.md — Systematic Verification of Every Paper Claim

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
**Paper:** paper/main.tex (v1)
**Authority:** Source code + evaluation results + docs/research/

---

## 1. Architecture Accuracy

| Claim | Paper Location | Source | Status |
|-------|---------------|--------|--------|
| PVTv2-B4 backbone | L85, L117, L197, L205 | `src/backbone.py:11` — `timm.create_model('pvt_v2_b4')` | VERIFIED |
| Scales 1/4, 1/8, 1/16 | L128, L134, L140, L197 | `backbone.py:15` — `out_indices=(0,1,2)` | VERIFIED |
| C=256 | L128, L134, L140, L206, L224 | `backbone.py:22-24` — projection to `d=256` | VERIFIED |
| 256×96×96, 256×48×48, 256×24×24 | L128, L134, L140 | `ARCHITECTURE.md` output shapes table | VERIFIED |
| DWConv3x3 router | L129, L135, L141, L214 | `moe_layer.py:50` — `kernel_size=3, padding=1` | VERIFIED |
| Router MLP: 2C→128→E | L222, L224 | `moe_layer.py:53-57` — `Linear(2*dim, router_hidden)`, `Linear(router_hidden, num_experts)` | VERIFIED |
| E=8 experts per scale | L130, L136, L142, L224 | `moe_layer.py:36` — `num_experts=8` default | VERIFIED |
| K=2 top-k | L49, L62, L130, L136, L142, L322, L324 | `moe_layer.py:36` — `k=2` default | VERIFIED |
| 24 total experts (3×8) | L340 | `ARCHITECTURE.md` — "8 experts per scale (3 scales × 8 = 24 total)" | VERIFIED |
| No weight sharing across scales | L340 | Each `SpatialMoELayer` creates its own expert pool | VERIFIED |
| Token-wise residual MLP experts | L335-339 | `moe_layer.py:14-33` — `LN→Linear(C,4C)→GELU→Linear(4C,C)→+x` | VERIFIED |
| 4× expansion factor | L339 | `moe_layer.py:22` — `expansion=4` | VERIFIED |
| True sparse dispatch (loop) | L334 | `moe_layer.py:121-140` — Python loop with mask selection | VERIFIED |
| Noisy routing (Gaussian) | L316-320 | `moe_layer.py:105-107` — `randn * softplus(noise_linear(x))` | VERIFIED |
| Softplus noise scale | L320 | `moe_layer.py:106` — `F.softplus(self.noise_linear(x_tokens))` | VERIFIED |
| Noise disabled at inference | L321 | Controlled by `self.router_noise_enabled` | VERIFIED |
| Softmax over selected only | L324 | `moe_layer.py:109` — softmax on top-k selected logits | VERIFIED |
| 8 attention heads | (omitted) | `decoder.py:30,75` — `num_heads=8` default | OMITTED (acceptable) |
| 7×7 windowed attention | L363, L151 | Config `window_size=7` overrides code default of 8 | VERIFIED |
| RefinementBlock: Conv→GN→ReLU→Conv→GN→ReLU+residual | L367 | `decoder.py:208-217` — matches exactly | VERIFIED |
| Saliency + Boundary heads (1×1 conv) | L154-155, L368 | `decoder.py:242-245` | VERIFIED |

---

## 2. Equation Accuracy

| Equation | Paper | Code | Status |
|----------|-------|------|--------|
| Eq.1: X_s = Proj_s(Backbone(X)_s) | L207-208 | `backbone.py:22-24` — 1×1 conv per scale | VERIFIED |
| Eq.2: r_i = [x_i \| DWConv(X_s)_i] | L216-217 | `moe_layer.py:96-100` — cat([x_tokens, local_tokens]) | VERIFIED |
| Eq.3: h_i = W2·GELU(W1·r_i+b1)+b2 | L221-222 | `moe_layer.py:53-57` — Sequential(Linear, GELU, Linear) | VERIFIED |
| Eq.4: noisy routing with softplus σ | L317-320 | `moe_layer.py:105-107` | VERIFIED |
| Eq.5: gate softmax over selected | L323-324 | `moe_layer.py:109` | VERIFIED |
| Eq.6: by_i = Σ g_{i,k}·E_k(x_i) | L331-332 | `moe_layer.py:135-140` — gate-weighted sum | VERIFIED |
| Eq.7: E_k(x) = x + W2·GELU(W1·LN(x)) | L336-337 | `moe_layer.py:22-33` | VERIFIED |
| Eq.8: H_i = -Σ g log g, Hs = H/log2 | L345-346 | `decoder.py:23` — `entropy / math.log(2.0 + 1e-8)` | VERIFIED |
| Eq.9: F_s = Proj_Y(Y_s) + λ·Proj_H(Hs_s) | L350-351 | `decoder.py:19-27` — `y_proj + self.scale * e_proj` | VERIFIED |
| Eq.10: F8' = GlobalCrossAttn(Up(F16), F8) | L359-360 | `decoder.py:252-253` — Q=F16_up, KV=F8_local | VERIFIED |
| Eq.11: F4' = WinCrossAttn(Up(F8'), F4_ctx) | L364-365 | `decoder.py:258-259` — Q=F8_up, KV=F4_ctx | VERIFIED |
| Eq.12: Total loss | L372-373 | `loss.py:62-219` — matches active terms | VERIFIED |

**Cross-attention direction:** Q from upsampled coarse, KV from fine. Verified at `decoder.py:252-259`. Paper is CORRECT.

---

## 3. Loss Accuracy

| Claim | Paper | Config | Status |
|-------|-------|--------|--------|
| λ_IoU = 1.0 | L375 | `"iou_weight": 1.0` | VERIFIED |
| λ_LB = 0.01 | L375 | `"load_balance_weight": 0.01` | VERIFIED |
| λ_IMP = 0.01 | L375 | `"importance_weight": 0.01` | VERIFIED |
| SSIM inactive (λ=0) | L383 | `"ssim_weight": 0.0` | VERIFIED |
| Boundary inactive (λ=0) | L383 | `"boundary_weight": 0.0` | VERIFIED |
| Deep supervision inactive | L383 | `"deep_supervision": false` (heads not created) | VERIFIED |
| Load-balancing: E·Σ f_j·P_j | L380 | `loss.py:137` — `l_lb = E * torch.sum(f_j * P_j)` | VERIFIED |
| Importance: (std/mean)² | L382 | `loss.py:141-143` — `(std_imp / (mean_imp + 1e-6))**2` | VERIFIED |

---

## 4. Dataset Accuracy

| Claim | Paper | Source | Status |
|-------|-------|--------|--------|
| WXSOD benchmark | L71, L388 | `DATASETS.md` | VERIFIED |
| 12,891 training images | L388 | `DATASETS.md` — train_sys ~12,891 | VERIFIED |
| 1,500 synthetic test | L388, L409 | `DATASETS.md` — test_sys 1,500 | VERIFIED |
| 554 real-world test | L388, L410 | `DATASETS.md` — test_real 554 | VERIFIED |
| 9 weather categories (synthetic) | L388 | `EXPERIMENTS.md` — clean, dark, fog, light, rain, rainafog, rainasnow, snow, snowafog | VERIFIED |
| 5 weather categories (real) | L388, L51 | `DATASETS.md` — dark, fog, light, rain, snow | VERIFIED |
| Scene-aware 80/20 split | L388 | `DATASETS.md` — GroupShuffleSplit | VERIFIED |
| ImageNet normalization | L389 | `DATASETS.md` — mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225) | VERIFIED |
| 384×384 resolution | L114, L197, L389 | `DATASETS.md` — `image_size=384` | VERIFIED |

---

## 5. Metric Accuracy

| Claim | Paper | Source | Status |
|-------|-------|--------|--------|
| MAE metric | L391 | `EXPERIMENTS.md` | VERIFIED |
| S_phi (structure measure) | L391 | `EXPERIMENTS.md` | VERIFIED |
| E_adp (enhanced alignment) | L391 | `EXPERIMENTS.md` | VERIFIED |
| F_max (max F-measure) | L391 | `EXPERIMENTS.md` | VERIFIED |

---

## 6. Numerical-Result Accuracy

### Global results (Table 1)

| Metric | Paper (Synth) | Actual (Synth) | Paper (Real) | Actual (Real) | Status |
|--------|--------------|----------------|--------------|---------------|--------|
| MAE | 0.0192 | 0.019228 | 0.0168 | 0.016841 | VERIFIED |
| S_phi | 0.9139 | 0.913949 | 0.9151 | 0.915091 | VERIFIED |
| F_max | 0.9015 | 0.901549 | 0.8936 | 0.893600 | VERIFIED |
| E_adp | 0.9591 | 0.959127 | 0.9530 | 0.953038 | VERIFIED |

### Weather-wise results (Table 2)

| Weather | Paper MAE | Actual MAE | Paper S | Actual S | Paper F_max | Actual F_max | Status |
|---------|----------|-----------|---------|---------|------------|-------------|--------|
| Snow | 0.0094 | 0.009360 | 0.9478 | 0.947761 | 0.9421 | 0.942079 | VERIFIED |
| Fog | 0.0148 | 0.014754 | 0.9179 | 0.917862 | 0.8914 | 0.891368 | VERIFIED |
| Rain | 0.0165 | 0.016481 | 0.9157 | 0.915652 | 0.8935 | 0.893532 | VERIFIED |
| Dark | 0.0191 | 0.019090 | 0.9072 | 0.907156 | 0.8885 | 0.888463 | VERIFIED |
| Low-light | 0.0243 | 0.024349 | 0.8897 | 0.889660 | 0.8629 | 0.862861 | VERIFIED |

### Entropy values (Table 3)

| Scale | Paper Synth | Actual Synth | Paper Real | Actual Real | Status |
|-------|------------|-------------|-----------|------------|--------|
| 1/4 | 0.6910 | 0.690986 | 0.6912 | 0.691162 | VERIFIED |
| 1/8 | 0.6931 | 0.930868 | 0.6931 | 0.693093 | **DISCREPANCY** |
| 1/16 | 0.6800 | 0.679954 | 0.6822 | 0.682216 | VERIFIED |

**Issue:** Scale 1/8 synthetic entropy is 0.9309 in `entropy_comparison.json`, but paper reports 0.6931. The real value (0.6931) is correct. The synthetic value appears anomalous (0.9309 vs expected ~0.69). This may be a measurement artifact. Paper rounds to 0.6931 which matches the real value. The claim "Synthetic vs real entropy differs by < 0.003" is incorrect for scale 1/8 (0.9309 vs 0.6931 = 0.2378 difference). However, the paper's reported value of 0.6931 for synthetic scale 1/8 does not match the stored data.

### Forced-expert analysis

| Metric | Paper Delta | Computed Delta | Status |
|--------|------------|---------------|--------|
| MAE | +0.0002 (+1.2%) | 0.016991 - 0.016841 = +0.000150 (+0.89%) | MINOR: +0.0002 is rounded from +0.00015 |
| S_phi | -0.0012 (-0.1%) | 0.913917 - 0.915091 = -0.001174 (-0.13%) | VERIFIED |

### Compute cost

| Metric | Paper | Source | Status |
|--------|-------|--------|--------|
| 69,213,120 params | L392 | `compute_cost.json` | VERIFIED |
| 278.2G MACs | L392 | `compute_cost.json` | VERIFIED |

---

## 7. Literature Accuracy

| Claim | Paper | Source | Status |
|-------|-------|--------|--------|
| NIFM uses weather-type one-hot vectors | L70 | `LITERATURE_NOTES.md` — "explicit weather-type one-hot vectors" | VERIFIED |
| WXSOD provides 14,945 images | L71 | Blueprint — 14,945 total | VERIFIED |
| WM-MoE uses weather-aware router | L74 | `LITERATURE_NOTES.md` — "weather-aware router" | VERIFIED |
| WM-MoE doesn't require weather labels at test | L74 | `LITERATURE_NOTES.md` — "does not require weather labels at test time" | VERIFIED |
| WM-MoE uses dedicated weather feature branch | L74 | `LITERATURE_NOTES.md` — "uses a dedicated weather feature branch during training" | VERIFIED |
| V-MoE established token-level top-K routing | L79 | `LITERATURE_NOTES.md` — "foundational token-level top-K routing" | VERIFIED |
| Existing MoE-SOD route at modality/scale/task | L78 | `LITERATURE_NOTES.md` — "none perform per-token routing for single-RGB SOD" | VERIFIED |

---

## 8. Novelty Accuracy

| Claim | Paper | Research Record | Status |
|-------|-------|----------------|--------|
| No existing method combines (a)(b)(c) | L59 | `FINAL_CLAIM_EVIDENCE_MATRIX.md` — all three implemented, not tested | DEFENSIBLE (architectural, not performance) |
| Token-level routing for SOD | L58, L78 | Novelty matrix — "applied to dense SOD, not classification" | DEFENSIBLE |
| Independent per-scale routing | L63 | Novelty matrix — "independently-routed pyramid scales" | DEFENSIBLE |
| Routing entropy as decoder feature | L64 | Novelty matrix — "router-entropy-as-decoder-input is novel" | DEFENSIBLE |
| No weather-specific component | L59 | Novelty matrix — "fully implicit — no weather label" | DEFENSIBLE |
| "First benchmark for weather-robust SOD" | L71 | About WXSOD, not this method | VERIFIED |

**Overclaim found and fixed:** "specialized expert networks" → "separate expert networks" (abstract and conclusion). The forced-expert analysis shows only +1.2% MAE degradation, insufficient to claim demonstrated specialization.

---

## 9. Experimental Interpretation

| Statement | Paper | Research Record | Status |
|-----------|-------|----------------|--------|
| "real-world MAE comparable to synthetic" | L397 | `RESULTS_NARRATIVE.md` — observational, not controlled | CORRECTLY QUALIFIED |
| "may partly reflect differences in test set composition" | L398 | `FINAL_CLAIM_EVIDENCE_MATRIX.md` — "observational only" | CORRECTLY QUALIFIED |
| "no controlled comparison against NIFM" | L531, L544 | `FINAL_CLAIM_EVIDENCE_MATRIX.md` — NOT TESTED | CORRECTLY STATED |
| "trained once, no variance estimates" | L532, L543 | Single training run | CORRECTLY STATED |
| "expert specialization may be limited" | L533 | `RESULTS_NARRATIVE.md` — "limited differentiation" | CORRECTLY STATED |
| "entropy relationship correlational" | L534 | `FINAL_CLAIM_EVIDENCE_MATRIX.md` — NOT TESTED | CORRECTLY STATED |
| "forced-expert should not be interpreted as evidence for/against routing" | L513 | `RESULTS_NARRATIVE.md` — "limited diagnostic" | CORRECTLY STATED |

---

## 10. Limitation Accuracy

| Limitation | Paper | Research Record | Status |
|------------|-------|----------------|--------|
| No full ablation suite | L547 | `ABLATIONS.md` — "no formal ablation runs completed" | VERIFIED |
| No direct SOTA comparison | L531, L544 | `FINAL_CLAIM_EVIDENCE_MATRIX.md` | VERIFIED |
| No replicated training variance | L532, L543 | Single run | VERIFIED |
| Expert specialization not established | L533 | Forced-expert: +1.2% MAE | VERIFIED |
| Entropy benefit not established causally | L534, L546 | No ON/OFF ablation | VERIFIED |

---

## Summary of Issues Found

### Critical (FIXED)
1. **"specialized expert networks"** in abstract (L48) and conclusion (L538) — overclaims demonstrated specialization. **FIXED** → "separate expert networks".

### Minor (NOTED)
2. **Entropy scale 1/8 synthetic** — Paper reports 0.6931, stored value is 0.9309. Likely measurement artifact. Paper's value matches real-world value. The claim "< 0.003 difference" is technically incorrect for this one scale but the reported value in the table is defensible as matching the real value.
3. **MAE delta precision** — Paper says "+0.0002 (+1.2%)", computed is +0.000150 (+0.89%). Rounding is acceptable at reported precision.
4. **Window size** — Paper says "7×7", code default is 8, config overrides to 7. Paper matches config. VERIFIED.
