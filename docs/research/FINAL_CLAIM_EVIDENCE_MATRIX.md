# FINAL CLAIM → EVIDENCE MATRIX

**Audit date:** 2026-09-02
**Purpose:** Map every major paper claim to its supporting/countervailing evidence.

---

## Claim A: Token-level spatial routing is useful for SOD

| Evidence Type | Status | Details |
|---------------|--------|---------|
| Implementation | EXISTS | Token-level router implemented: DWConv3x3 + MLP, noisy top-k (k=2), true sparse dispatch (`src/moe_layer.py:49-141`) |
| Literature | EXISTS | V-MoE (NeurIPS 2021) established token-level routing for ViT classification; M³ViT (NeurIPS 2022) for multi-task |
| Experiment | **NOT TESTED** | No ablation comparing routed vs non-routed model. No non-MoE baseline exists. |
| Counterevidence | NONE | — |
| **Status** | **NOT TESTED** | |
| **Recommended wording** | "We design a token-level spatial router... [describe architecture]. Token-level routing has shown efficacy in classification (V-MoE) and multi-task learning (M³ViT); we adapt it to dense SOD." — Present as design choice, not empirically validated contribution. |

---

## Claim B: No weather-specific component is useful/sufficient

| Evidence Type | Status | Details |
|---------------|--------|---------|
| Implementation | EXISTS | Model has no weather label input at train or test time (`src/model.py` — no weather conditioning) |
| Literature | EXISTS | WM-MoE (2023), MoWE (2023), Zhu et al. (CVPR 2023) all use weather-conditioned routing |
| Experiment | **NOT TESTED** | No ablation comparing weather-conditioned vs label-free routing |
| Counterevidence | NONE | — |
| **Status** | **NOT TESTED** | |
| **Recommended wording** | "Unlike WM-MoE and MoWE, our method does not require weather labels at train or test time." — Present as architectural design, not as empirically validated superiority. |

---

## Claim C: Independent per-scale routing is useful

| Evidence Type | Status | Details |
|---------------|--------|---------|
| Implementation | EXISTS | Three independent `SpatialMoELayer` instances at scales 1/4, 1/8, 1/16 (`src/model.py:14-16`) |
| Literature | EXISTS | Multi-scale MoE precedent in M³ViT (task-conditioned); no prior work does independent per-scale routing for SOD |
| Experiment | **NOT TESTED** | No ablation comparing independent vs shared routing. No single-scale routing baseline. |
| Counterevidence | NONE | — |
| **Status** | **NOT TESTED** | |
| **Recommended wording** | "We deploy independent routers at each pyramid scale... [describe architecture]." — Present as design choice, not empirically validated. |

---

## Claim D: Routing entropy in the decoder is useful

| Evidence Type | Status | Details |
|---------------|--------|---------|
| Implementation | EXISTS | `EntropyFusionBlock` fuses entropy channel into decoder (`src/decoder.py:14-27`). Learnable scale parameter initialized to 0.1. |
| Literature | NO PRECEDENT | Router-entropy-as-decoder-input is claimed as novel; no prior work found |
| Experiment | **NOT TESTED** | No ablation with/without entropy fusion. The `scale` parameter was not analyzed post-training. |
| Counterevidence | NONE | — |
| **Status** | **NOT TESTED** | |
| **Recommended wording** | "We incorporate routing entropy as an auxiliary channel in the decoder... [describe mechanism]." — Present as novel design, explicitly note that ablation to validate its contribution was not performed. |

---

## Claim E: MoE is useful for adverse-weather SOD

| Evidence Type | Status | Details |
|---------------|--------|---------|
| Implementation | EXISTS | Full MoE system: 8 experts per scale, sparse dispatch, load-balance + importance losses |
| Literature | EXISTS | WM-MoE, MoWE, Complexity Experts all show MoE benefits for weather tasks |
| Experiment | **NOT TESTED** | No non-MoE baseline (shared MLP) exists. No dense MoE baseline. No matched-FLOPs comparison. |
| Counterevidence | The forced-expert ablation shows only marginal degradation (ΔMAE=+0.0002), which could be interpreted as: (a) the router is not critical, or (b) the single forced expert is already competent |
| **Status** | **NOT TESTED** | |
| **Recommended wording** | "We apply mixture-of-experts to SOD... [describe architecture]." — Present as novel application, explicitly state no ablation against non-MoE baseline was performed. |

---

## Claim F: Experts specialize by weather/artifact type

| Evidence Type | Status | Details |
|---------------|--------|---------|
| Implementation | EXISTS | `WeatherAnalyzer` in `src/diagnostics.py:80-177` computes P(weather\|expert) — but **disabled in training** (`train_ddp.py:560`) |
| Literature | EXISTS | Complexity Experts (CVPR 2025) shows implicit specialization; M³ViT shows task-based clustering |
| Experiment | **NOT TESTED** | No per-expert assignment statistics. No weather\|expert distributions. No t-SNE/UMAP. |
| Counterevidence | The forced-expert ablation shows minimal degradation, which **contradicts** the intuition that specialists are needed — if one expert handles all tokens with only 1.2% MAE increase, specialization may not be occurring |
| **Status** | **NOT TESTED** (and counterevidence exists) | |
| **Recommended wording** | DO NOT CLAIM. The forced-expert result actually suggests experts may NOT be specializing by weather. This claim requires explicit per-expert assignment analysis before it can be made. |

---

## Claim G: The model is robust across weather conditions

| Evidence Type | Status | Details |
|---------------|--------|---------|
| Implementation | — | — |
| Literature | — | — |
| Experiment | **PARTIALLY SUPPORTED** | MAE range on test_real: 0.0094 (snow) to 0.0243 (light) — a 2.6× spread. On test_sys: 0.0142 (clean) to 0.0237 (rainafog) — a 1.7× spread. |
| Counterevidence | "Light" (low-light) conditions show substantially worse performance (MAE=0.0243, S=0.8897). This is the worst-performing category and may indicate a systematic weakness. |
| **Status** | **PARTIALLY SUPPORTED** | |
| **Recommended wording** | "The model achieves MAE between 0.0094 and 0.0243 across real-world weather categories... [report range]. Performance is strongest on snow and weakest on low-light conditions." — Report the range honestly. Do not claim uniform robustness. |

---

## Claim H: The model generalizes from synthetic to real weather

| Evidence Type | Status | Details |
|---------------|--------|---------|
| Implementation | — | — |
| Literature | — | — |
| Experiment | **PARTIALLY SUPPORTED** | Real-world MAE (0.0168) ≤ synthetic MAE (0.0192) on matching weather categories. But this is observational, not a controlled experiment. |
| Counterevidence | Not a controlled domain generalization test. The real-world test set may be inherently easier. No clean-source baseline exists. |
| **Status** | **PARTIALLY SUPPORTED** (observational only) | |
| **Recommended wording** | "When trained on synthetic data, the model achieves comparable or better performance on real-world weather (MAE 0.0168 vs 0.0192), suggesting effective transfer of weather-robust features." — Note this is an observation, not a controlled generalization experiment. |

---

## Claim I: The model is computationally efficient

| Evidence Type | Status | Details |
|---------------|--------|---------|
| Implementation | — | — |
| Literature | — | — |
| Experiment | **NOT TESTED** | 66.27M params, 278.2G MACs, 3.83 FPS reported — but no comparison to non-MoE baseline, no standardized hardware, no memory analysis |
| Counterevidence | Without a baseline comparison, "efficient" is meaningless. 3.83 FPS may or may not be good depending on the baseline. |
| **Status** | **NOT TESTED** | |
| **Recommended wording** | "The model contains 66.27M parameters and 278.2G MACs." — Report the numbers without claiming efficiency. |

---

## Summary Matrix

| Claim | Implementation | Literature | Experiment | Status |
|-------|---------------|------------|------------|--------|
| A. Token routing useful | ✓ Exists | ✓ Precedent | ✗ Not tested | NOT TESTED |
| B. No weather labels needed | ✓ Exists | ✓ Precedent | ✗ Not tested | NOT TESTED |
| C. Independent per-scale routing useful | ✓ Exists | ✓ Novel position | ✗ Not tested | NOT TESTED |
| D. Entropy fusion useful | ✓ Exists | ✗ Novel claim | ✗ Not tested | NOT TESTED |
| E. MoE useful for SOD | ✓ Exists | ✓ Precedent | ✗ Not tested | NOT TESTED |
| F. Experts specialize by weather | ✓ Diagnostics exist (disabled) | ✓ Precedent | ✗ Not tested (+ counterevidence) | NOT TESTED |
| G. Robust across weather | — | — | ~ Partial | PARTIALLY SUPPORTED |
| H. Synthetic→real transfer | — | — | ~ Observational | PARTIALLY SUPPORTED |
| I. Computationally efficient | — | — | ✗ Not tested | NOT TESTED |

---

## What This Means for the Paper

**The completed experiments support exactly TWO categories of claims:**

1. **Performance reporting:** "Our model achieves X on test set Y under condition Z." — Fully supported.

2. **Architectural description:** "Our model uses token-level routing with K experts at each of 3 scales..." — Fully supported by implementation.

**The completed experiments do NOT support any causal or comparative claims:**
- Cannot claim any design choice causes improvement
- Cannot claim MoE is better than non-MoE
- Cannot claim routing is better than no routing
- Cannot claim entropy fusion helps
- Cannot claim experts specialize
- Cannot claim efficiency
- Cannot claim SOTA performance

**The paper must be written as an architectural contribution with empirical performance characterization, NOT as an empirically validated design.**
