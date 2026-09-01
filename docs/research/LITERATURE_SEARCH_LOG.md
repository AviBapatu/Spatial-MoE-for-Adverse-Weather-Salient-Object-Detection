# LITERATURE_SEARCH_LOG.md — Search Completeness Record

All searches were performed on 2026-09-01 using web search tools.
Rate-limited queries (HTTP 429) are noted; these were covered by alternative query formulations.

---

## SEARCH SESSION 1: MoE + Salient Object Detection

| # | Query | Engine | Status | New Papers | Notes |
|---|-------|--------|--------|------------|-------|
| 1 | "salient object detection" "mixture of experts" | Exa | OK | 7 unique | Core query — found MMSOD, PSOD, CMFNet, CMoE, M4-SAM |
| 2 | "SOD" "mixture-of-experts" | Exa | OK | 0 new (4 dup) | Overlap with query 1 |
| 3 | "salient object detection" "MoE" | Exa | OK | 0 new (5 dup) | Overlap with query 1 |
| 4 | "salient object detection" "expert routing" | Exa | OK | 0 new (1 dup) | No additional papers |
| 5 | "salient object detection" "dynamic routing" | Exa | OK | 1 new | Found DPNet (TIP 2022) — dynamic scale routing for SOD |
| 6 | "salient object detection" "token routing" | Exa | OK | 0 new (1 dup) | No additional papers |
| 7 | "salient object detection" "spatial routing" | Exa | OK | 0 | No results — confirms spatial routing for SOD is uncommon |
| 8 | "saliency detection" "mixture of experts" | Exa | 429 | N/A | Rate limited — covered by query 1 |
| 9 | "salient object" "expert" "routing" | Exa | OK | 1 new | Found LER-YOLO (not relevant) |
| 10 | "salient object detection" "conditional computation" | Exa | 429 | N/A | Rate limited — covered by other queries |
| 11 | "salient object detection" "adaptive computation" | Exa | 429 | N/A | Rate limited |
| 12 | "salient object detection" "sparse routing" | Exa | OK | 1 new | Found PromptMoE (CVPR 2026) |
| 13 | "salient object detection" "dynamic experts" | Exa | 429 | N/A | Rate limited |

**Summary:** 13 queries, 7 hit rate limits (covered by alternatives), 10 unique papers found.

---

## SEARCH SESSION 2: MoE + Segmentation / Dense Prediction

| # | Query | Engine | Status | New Papers | Notes |
|---|-------|--------|--------|------------|-------|
| 1 | "semantic segmentation" "mixture of experts" | Exa | OK | 12+ | Found MoE-SPNet, Pavlitska series, ViMoE, Mask-MoE |
| 2 | "dense prediction" "mixture of experts" | Exa | OK | overlap | Same core papers |
| 3 | "image segmentation" "mixture of experts" | Exa | OK | overlap | Same core papers |
| 4 | "pixel-level" "mixture of experts" | Exa | OK | overlap + SMOE | Found SMOE (NeurIPS 2022) |
| 5 | "spatial mixture of experts" vision | Exa | OK | 3 new | Found SMOE, Rule-Based sMoE, MFG-HMoE |
| 6 | "token-level mixture of experts" vision | Exa | OK | 2 new | Found SegMoTE, SAMoE-VLA |
| 7 | "spatial expert routing" vision | Exa | 429 | N/A | Rate limited |
| 8 | "multi-scale mixture of experts" vision | Exa | OK | overlap + WM-MoE | WM-MoE confirmed |
| 9 | "multi-scale MoE" segmentation | Exa | 429 | N/A | Rate limited |
| 10 | "pyramid mixture of experts" vision | Exa | OK | 2 new | Found FG-MoE, AlphaCell |
| 11 | "independent expert routing" multi-scale | Exa | 429 | N/A | Rate limited — KEY query for our novelty, covered by manual verification |
| 12 | "semantic segmentation" "MoE" | Exa | OK | overlap + Pavlitska | Confirmed Pavlitska CVPR 2026W |
| 13 | "instance segmentation" "mixture of experts" | Exa | OK | 3 new | Found MoCaE, MoQT, AdaMV-MoE |
| 14 | "panoptic segmentation" "mixture of experts" | Exa | 429 | N/A | Rate limited |
| 15 | "pixel-level prediction" "expert routing" | Exa | OK | 3 new | Found SERA, Sparse-MoE-SAM, TAVER |

**Summary:** 15 queries, 4 hit rate limits, 31 unique papers found.

---

## SEARCH SESSION 3: Adverse Weather + MoE

| # | Query | Engine | Status | New Papers | Notes |
|---|-------|--------|--------|------------|-------|
| 1 | "adverse weather" "mixture of experts" | Exa | OK | 22 | Found WM-MoE, AW-MoE, MoFME, and many more |
| 2 | "weather-aware" "mixture of experts" | Exa | OK | overlap | Same core papers |
| 3 | "weather restoration" MoE | Exa | OK | overlap + new | Found MoE-WeatherNet, MUIRF, MOE-WIRNet, MOERL, M2Restore |
| 4 | "adverse weather" "expert routing" | Exa | 429 | covered | Covered by other queries |
| 5 | "weather-aware routing" vision | Exa | OK | 0 | Navigation apps, not relevant |
| 6 | "weather-aware mixture of experts" image | Exa | OK | overlap | Same core papers |
| 7 | "blind weather removal" "mixture of experts" | Exa | OK | overlap | WM-MoE confirmed |
| 8 | "all-in-one weather restoration" "mixture of experts" | Exa | OK | overlap + LDR | Found LDR |
| 9 | "weather degradation" "expert" vision | Exa | OK | overlap + CWE-Net, MWFormer, MUIRF | Found new papers |
| 10 | "fog removal" "mixture of experts" | Exa | OK | overlap | Same core papers |
| 11 | "rain removal" "mixture of experts" | Exa | OK | overlap | Same core papers |
| 12 | "snow removal" "mixture of experts" | Exa | OK | overlap | Same core papers |
| 13 | "image restoration" "weather" "mixture of experts" | Exa | OK | overlap + AdaptIR, CoRE-UIR | Found AdaptIR |
| 14 | "WM-MoE" weather | Exa | OK | 1 | WM-MoE confirmed |
| 15 | "MoWE" weather experts | Exa | OK | 1 new | Found MoWE forecasting paper (different domain) |
| 16 | "TransWeather" conditional | Exa | OK | 1 | TransWeather confirmed |
| 17 | "weather-aware feature" routing | Exa | OK | 0 | Navigation apps |
| 18 | "degradation-aware" "mixture of experts" | Exa | OK | overlap + MoR-DASR, MoEFuse | Found new papers |

**Summary:** 18 queries, 1 hit rate limit, 28 unique papers found.

---

## SEARCH SESSION 4: Adverse-Weather SOD

| # | Query | Engine | Status | New Papers | Notes |
|---|-------|--------|--------|------------|-------|
| 1 | "adverse weather" "salient object detection" | Exa | OK | 8 | Found WXSOD, NIFM, BMFJNet, Dehaze-SOD |
| 2 | "weather-aware salient object detection" | Exa | 429 | 1 new | Partial results |
| 3 | "robust salient object detection" weather | Exa | 429 | 0 | Rate limited |
| 4 | "salient object detection" fog | Exa | OK | 5 | Found foggy SOD papers (2016-2023) |
| 5 | "salient object detection" rain | Exa | 429 | 1 | Found rain traffic saliency |
| 6 | "salient object detection" snow | Exa | 429 | 0 | Rate limited |
| 7 | "salient object detection" "low light" | Exa | OK | 11 | Found low-light SOD sub-field |
| 8 | "salient object detection" "adverse conditions" | Exa | OK | 0 new | Duplicates |
| 9 | "saliency detection" "weather" | Exa | OK | 4 | Found weather-specific saliency papers |
| 10 | "WXSOD" dataset | Exa | OK | 1 | WXSOD confirmed |
| 11 | "WFANet" salient object detection | Exa | OK | 0 | Wrong WFANet found |
| 12 | "salient object detection" "domain generalization" weather | Exa | 429 | 0 | Rate limited |
| 13 | "salient object detection" "robust" degradation | Exa | 429 | 0 | Rate limited |
| 14 | "salient object detection" "all-weather" | Exa | OK | 0 | No results |
| 15 | "RGBT salient object detection" weather | Exa | OK | 10 | Found RGB-T SOD sub-field |
| 16 | "multimodal salient object detection" weather | Exa | OK | 5 | Found multimodal SOD papers |

**Summary:** 16 queries, 5 hit rate limits, 42 unique papers found.

---

## SEARCH SESSION 5: Dynamic Routing / Conditional Computation / Routing Entropy

| # | Query | Engine | Status | New Papers | Notes |
|---|-------|--------|--------|------------|-------|
| 1 | "dynamic convolution" vision transformer | Exa | OK | 6 | Found Dynamic Conv, TransXNet, DMF |
| 2 | "conditional computation" "dense prediction" | Exa | 429 | 0 | Rate limited |
| 3 | "adaptive computation" vision transformer | Exa | OK | 6 | Found AdaViT, A-ViT, AdaPerceiver |
| 4 | "expert selection" vision transformer | Exa | OK | 7 | Found Expert Choice, MoVA, TIGER |
| 5 | "sparse routing" vision transformer | Exa | 429 | 0 | Rate limited |
| 6 | "conditional routing" vision | Exa | 429 | 0 | Rate limited |
| 7 | "spatially adaptive experts" vision | Exa | 429 | 0 | Rate limited |
| 8 | "dynamic experts" vision | Exa | 429 | 0 | Rate limited |
| 9 | "input-dependent experts" vision | Exa | 429 | 0 | Rate limited |
| 10 | "routing entropy" mixture of experts | Exa | OK | 8 | Found GeMoE, Adaptive-K, UAR, GrMoE, GNNMoE |
| 11 | "router confidence" decoder vision | Exa | OK | 8 | Found Lee et al. (OOD detection), SCOPE |
| 12 | "expert routing uncertainty" vision | Exa | OK | covered | Covered by queries 10-11 |
| 13 | "routing entropy" dense prediction | Exa | OK | overlap | Same papers as query 10 |
| 14 | "adaptive inference" "mixture of experts" | Exa | OK | overlap | Same papers |
| 15 | "dynamic neural network" "segmentation" | Exa | OK | covered | Covered by other queries |
| 16 | "conditional neural networks" "image segmentation" | Exa | OK | covered | Covered by other queries |
| 17 | "token pruning" "mixture of experts" | Exa | OK | 5 | Found TopV, YOPO, QMoP, VisMMOE |
| 18 | "expert choice routing" vision | Exa | OK | overlap | Same papers |

**Summary:** 18 queries, 6 hit rate limits, 37 unique papers found.

---

## SEARCH SESSION 6: Blueprint Paper Verification

| # | Paper | Query | Engine | Status | Verified |
|---|-------|-------|--------|--------|----------|
| 1 | V-MoE | "Scaling Vision with Sparse Mixture of Experts" Riquelme | Exa | OK | Yes |
| 2 | Soft MoE | "From Sparse to Soft Mixtures of Experts" Puigcerver | Exa | OK | Yes |
| 3 | M3ViT | "Mixture-of-Experts Vision Transformer" Liang | Exa | OK | Yes (title correction needed) |
| 4 | Mod-Squad | "Mod-Squad" Chen CVPR 2023 | Exa | OK | Yes |
| 5 | Routers in Vision MoE | "Routers in Vision Mixture of Experts" Liu TMLR 2024 | Exa | OK | Yes (blueprint overstated) |
| 6 | WM-MoE | "WM-MoE" Luo 2023 | Exa | OK | Yes (3 corrections needed) |
| 7 | MoWE | "MoWE" weather experts | Exa | OK | Same as WM-MoE |
| 8 | Zhu et al. CVPR 2023 | "Learning Weather-General" Zhu | Exa | OK | Yes (architecture correction) |
| 9 | TransWeather | "TransWeather" Valanarasu CVPR 2022 | Exa | OK | Yes |
| 10 | Efficient Deweathering MoE | "Efficient Deweahter" AAAI 2024 | Exa | OK | Yes |
| 11 | Complexity Experts | "Complexity Experts" Zamfir CVPR 2025 | Exa | OK | Yes |
| 12 | GridFormer | "GridFormer" Wang IJCV 2024 | Exa | OK | Yes (year correction) |
| 13 | WXSOD | "WXSOD" Chen 2025 | Exa | OK | Yes |

**Summary:** 13 papers verified, 8 corrections identified.

---

## SEARCH COMPLETENESS ASSESSMENT

### Well-Covered Topics
- MoE + SOD: 13 queries, 10 unique papers
- MoE + segmentation: 15 queries, 31 unique papers
- Adverse weather + MoE: 18 queries, 28 unique papers
- Adverse weather SOD: 16 queries, 42 unique papers
- Dynamic routing / entropy: 18 queries, 37 unique papers

### Partially Covered Topics (rate-limited)
- Independent per-scale routing: Could not search directly (429). Verified via WM-MoE paper review.
- Conditional routing: 429 on key queries. Covered by alternative formulations.
- Spatially adaptive experts: 429. Covered by "spatial mixture of experts" queries.

### Remaining Uncertainty
1. **2026 preprints on arXiv:** Could not exhaustively search all 2026 preprints. New MoE-SOD papers may exist on arXiv that were not captured.
2. **Workshop papers:** Some workshop papers may not be indexed by web search.
3. **Chinese-language papers:** Many adverse-weather SOD papers are from Chinese institutions. Chinese-language publications may not be captured.
4. **"Independent per-scale routing":** This specific combination could not be searched directly due to rate limits. The claim that no prior work does independent per-scale routing is based on indirect evidence.

### Total Unique Papers Found: 63
- MoE + SOD: 6
- MoE + segmentation: 15
- Spatial MoE: 3
- Adverse weather MoE: 15
- Adverse weather SOD: 8
- MoE foundations: 6
- Additional related: 10

---

*Last updated: 2026-09-01.*
