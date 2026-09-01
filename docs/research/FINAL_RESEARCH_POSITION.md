# FINAL_RESEARCH_POSITION.md — Defensible Research Position

Based on: ACTUAL IMPLEMENTATION + PRIMARY LITERATURE + ELICIT EVIDENCE + ADVERSARIAL NOVELTY ANALYSIS

---

## 1. Problem Definition

Salient object detection (SOD) under adverse weather conditions (fog, rain, snow, low-light) is an open problem. Existing approaches either (a) use explicit weather-type labels to condition the model [NIFM, WFANet], (b) add a dedicated weather prediction branch [WFANet], or (c) apply MoE to image restoration rather than SOD [WM-MoE]. No existing method combines spatial token-level MoE routing with SOD under adverse weather without any weather-specific supervision or architectural component.

---

## 2. Established Prior Art

| Paper | Task | Key contribution | Limitation vs. our work |
|-------|------|-----------------|------------------------|
| WM-MoE (2023) | Weather restoration | Token-level MoE with weather branch | Restoration, not SOD; has weather branch |
| MoFME (AAAI 2024) | Weather restoration | Uncertainty-aware token-level MoE | Restoration, not SOD; no multi-scale |
| NIFM (2025) | Adverse-weather SOD | Weather-type encoding for SOD | Explicit weather labels, no MoE |
| WFANet (2025) | Adverse-weather SOD | Two-branch baseline | Image-level, no MoE, requires weather labels |
| MMSOD (IJCNN 2025) | Multimodal SOD | MoE for SOD | Modality-level routing, not spatial tokens |
| SegMoTE (CVPR 2026) | Medical segmentation | Token-level MoE | Medical imaging, not SOD |
| Pavlitska et al. (CVPR 2026W) | Semantic segmentation | Sparse MoE in CNNs | Patch-level, not token-level; no weather |
| HoME (2025) | 3D medical segmentation | Hierarchical Soft MoE | 3D medical, not 2D SOD |
| Pavlitska et al. (ICCVW 2025) | Semantic segmentation | Uncertainty from MoE | OOD detection, not decoder input |

---

## 3. Closest Competing Methods

### 3.1 WM-MoE (Luo et al., 2023)
- **Architecture:** Token-level routing via WEAR router with weather feature branch (WGF-CL contrastive learning)
- **Routing:** Per-patch, conditioned on decoupled content + weather features
- **Weather labels:** Not required at test time, but used during training for contrastive learning
- **Multi-scale:** Multi-scale experts (DWConv 1/3/5/7) but single router
- **Key difference from us:** (1) Restoration, not SOD. (2) Has dedicated weather branch. (3) Single router across scales.

### 3.2 NIFM (Chen et al., 2025)
- **Architecture:** Noise Indicator Fusion Module with one-hot weather-type encoding
- **Routing:** None — uses explicit weather-type vector as encoder conditioning
- **Weather labels:** Required (one-hot encoding)
- **Multi-scale:** No
- **Key difference from us:** (1) Explicit weather labels. (2) No MoE routing. (3) Global weather vector, not spatial routing.

### 3.3 WFANet (Chen et al., 2025)
- **Architecture:** Two-branch (weather prediction + saliency detection)
- **Routing:** None — image-level weather prediction branch
- **Weather labels:** Required (trained with weather labels)
- **Multi-scale:** No
- **Key difference from us:** (1) Image-level, not spatial. (2) Requires weather labels. (3) No MoE.

### 3.4 MMSOD (Cen et al., 2025)
- **Architecture:** MoE with adapter for multimodal SOD
- **Routing:** Modality-level gating
- **Weather labels:** No
- **Multi-scale:** Yes (multiscale fusion adapter)
- **Key difference from us:** (1) Modality-level, not spatial token-level. (2) Multimodal (RGB-D), not single RGB.

### 3.5 Pavlitska et al. (ICCVW 2025)
- **Architecture:** MoE for semantic segmentation
- **Routing:** Token-level with gating network
- **Uncertainty:** Computes gate entropy per token for OOD detection
- **Key difference from us:** Uses entropy for OOD detection, NOT as decoder input feature.

---

## 4. What Our Implementation Actually Contributes

From verified source code (RESEARCH_TRUTH.md):

1. **Token-level spatial routing:** Each spatial token independently selects top-2 experts from a pool of 8 via noisy top-k routing with DWConv3x3-enhanced router input (`moe_layer.py:96-112`)

2. **Independent per-scale routing:** Three completely separate `SpatialMoELayer` instances (one per pyramid scale), each with its own router MLP, expert pool, and noise parameters (`model.py:14-16`)

3. **Entropy-decoder integration:** Per-token routing entropy is normalized by log(2), projected via Conv2d, and added to features with a learnable scale parameter in `EntropyFusionBlock` (`decoder.py:14-27`)

4. **No weather-specific components:** Router operates purely on spatial content features (DWConv3x3 + MLP). No weather branch, no weather labels, no CLIP features at any stage.

5. **SOD-specific design:** Boundary head, deep supervision, cross-attention fusion — designed for pixel classification with topology constraints, not pixel regression.

---

## 5. Strongest Defensible Contribution

**Routing entropy as decoder input feature (Claim D).**

This is the strongest novelty claim because:
- No prior paper feeds routing entropy as an explicit spatial feature channel into a dense prediction decoder
- Pavlitska et al. compute gate entropy but use it for OOD detection, not decoder input
- UGRAN uses uncertainty maps for decoder refinement but these are prediction uncertainty, not routing entropy
- The specific architectural wiring (entropy → EntropyFusionBlock → decoder features) is a concrete, verifiable design choice
- The ablation (removing EntropyFusionBlock) is straightforward to implement and evaluate

**Recommended claim:** "We feed per-token routing entropy as an explicit feature channel into the decoder, providing the model with an explicit uncertainty signal about routing decisions at each spatial location."

---

## 6. Secondary Contributions

### 6.1 Token-level MoE for SOD (Claim A)
- **Strength:** No existing MoE-SOD paper does spatial token-level routing
- **Qualification:** Must cite WM-MoE (restoration) and SegMoTE (medical) as prior token-level MoE in adjacent tasks
- **Recommended claim:** "We apply token-level spatial MoE routing to salient object detection, extending MoE from classification [V-MoE] and restoration [WM-MoE] to dense segmentation."

### 6.2 Independent per-scale routing (Claim C)
- **Strength:** No prior work uses independent routers with separate expert pools at each pyramid scale
- **Qualification:** Must distinguish from WM-MoE (single router) and HoME (hierarchical)
- **Recommended claim:** "Each pyramid scale has its own independent router and expert pool, allowing different routing decisions at different receptive fields."

### 6.3 No weather-specific component (Claim B)
- **Strength:** Distinct from WM-MoE (weather branch) and NIFM (weather labels)
- **Qualification:** Must NOT claim "first label-free" — WM-MoE also routes without weather labels at test time. Claim instead: "no weather-specific architectural component or supervision"
- **Recommended claim:** "Unlike prior weather-aware methods that require a dedicated weather branch [WM-MoE] or explicit weather labels [NIFM, WFANet], our router operates purely on spatial content features with no weather-specific component."

### 6.4 MoE for adverse-weather SOD (Claim E)
- **Strength:** Intersection of MoE + SOD + adverse weather is novel
- **Qualification:** Must distinguish from WM-MoE (restoration) and NIFM (no MoE)
- **Recommended claim:** "We apply mixture-of-experts specifically to salient object detection under adverse weather conditions, a setting not previously addressed with MoE routing."

---

## 7. Claims That Must Be Qualified

| Original Claim | Problem | Required Qualification |
|----------------|---------|----------------------|
| "First label-free weather routing" | WM-MoE also routes without weather labels at test time | "No weather-specific architectural component or supervision at any stage" |
| "Novel routing entropy contribution" | Pavlitska et al. compute gate entropy; UGRAN uses uncertainty for refinement | "No prior work feeds routing entropy as an explicit feature channel into the decoder" |
| "Novel multi-scale MoE" | WM-MoE has multi-scale experts | "Independent per-scale routers with separate expert pools, not a single router across scales" |

---

## 8. Claims That Must Be Removed

| Claim | Reason |
|-------|--------|
| "First MoE for SOD" | MMSOD, CMFNet, CMoE, M4-SAM, PSOD all apply MoE to SOD |
| "First label-free routing" | WM-MoE routes without weather labels at test time |
| "WM-MoE uses feature-map-level routing" | WM-MoE routes at token level (corrected) |
| "WM-MoE requires weather labels at test time" | WM-MoE does not (corrected) |
| "WM-MoE and MoWE are separate papers" | They are the same paper (corrected) |

---

## 9. Experiments Required to Support Each Contribution

| Contribution | Required Experiment | Status |
|-------------|-------------------|--------|
| A. Token-level MoE for SOD | Compare against modality-level MoE baselines (MMSOD, CMFNet) | NOT RUN |
| B. No weather-specific component | Ablation: add weather branch, show no improvement or degradation | NOT RUN |
| C. Independent per-scale routing | Ablation: share router across scales, compare performance | NOT RUN |
| D. Routing entropy as decoder input | Ablation: remove EntropyFusionBlock, measure impact on metrics | NOT RUN |
| E. MoE for adverse-weather SOD | Compare against NIFM and WFANet on WXSOD | NOT RUN |
| Expert specialization | t-SNE/UMAP of token embeddings by expert vs weather category | NOT RUN |
| Entropy-boundary correlation | Plot entropy maps overlaid on images, compute boundary correlation | NOT RUN |
| Routing stability | Analyze routing consistency across weather severity | NOT RUN |

---

## 10. Remaining Uncertainties

1. **No ablation results exist.** All claims about the value of specific design choices (entropy fusion, independent routing, token-level routing) are unvalidated. The code exists but no ablations have been run.

2. **No SOTA comparison exists.** We have not compared against NIFM, WFANet, or other adverse-weather SOD methods on WXSOD.

3. **No interpretability analysis exists.** We cannot claim expert specialization without t-SNE/UMAP analysis and per-expert heatmaps.

4. **No entropy-effectiveness analysis exists.** We cannot claim that entropy helps without the ablation removing EntropyFusionBlock.

5. **WM-MoE's weather branch details need manual verification.** The exact mechanism of WGF-CL contrastive learning and whether it constitutes "implicit weather labeling" requires reading the full WM-MoE paper.

6. **2026 arXiv preprints may contain relevant work.** The literature search was comprehensive but cannot claim exhaustiveness for very recent preprints.

7. **The "independent per-scale routing" claim could not be directly searched** due to rate limits. The claim is based on indirect evidence from WM-MoE (single router) and HoME (hierarchical, not independent).

---

## Appendix: Elicit Material Assessment

### Papers found by Elicit but NOT in our database (new additions):

| Paper | Relevance | Assessment |
|-------|-----------|------------|
| HEHP (Zha et al., 2025) | High — heterogeneous experts for underwater SOD | Related but different domain (underwater) |
| Dynamic Neural Network for RGB-D SOD (Li et al., 2026) | Medium — dynamic routing for RGB-D SOD | Different routing mechanism (mutual information) |
| Feature Quality-Based Dynamic Feature Selection (Naqvi et al., 2016) | Low — early non-MoE precursor | Historical context only |
| Learning Dynamic Routing for Semantic Segmentation (Li et al., 2020) | Medium — differentiable soft conditional gates | Path-level routing, not token-level MoE |
| Diversified Dynamic Routing (Csaba et al., 2022) | Medium — addresses router collapse | Useful for MoE failure mode discussion |
| VST (Liu et al., 2021) | Low — transformer for SOD | Not MoE, not adaptive routing |
| TSVT (Gao et al., 2024) | Medium — dynamic sparse tokens for RGB-D SOD | Token sparsification, not expert routing |
| HI-MoE (Vashkelis & Trukhina, 2026) | Medium — hierarchical instance-conditioned MoE for detection | Relevant for hierarchical routing concept |
| HoME (Płotka et al., 2025) | High — hierarchical Soft MoE for 3D medical segmentation | Closest architectural match for hierarchical routing |
| MLoRE (Yang et al., CVPR 2024) | Medium — multi-task dense prediction via MoE | Relevant for decoder MoE design |
| TaskExpert (Ye & Xu, ICCV 2023) | Medium — dynamic task-specific MoE | Relevant for task-conditioned routing |
| ShapeMoE (Li et al., 2025) | Low — shape-specific MoE for amodal segmentation | Different conditioning variable |
| Uncertainty-Aware Deep Calibrated SOD (Zhang et al., 2020) | High — uncertainty calibration for SOD | Relevant for uncertainty discussion but different from routing entropy |
| UDNet (Fang et al., 2022) | High — uncertainty-aware SOD with contour uncertainty | Different uncertainty type (contour, not routing) |
| UGRAN (Yuan et al., TIP 2025) | High — uncertainty-guided refinement for SOD | Uses prediction uncertainty, not routing entropy — key distinction |
| Graph-Based Uncertainty Modeling (Xiong et al., 2025) | Medium — dynamic uncertainty propagation | Different uncertainty mechanism |

### Elicit's strongest recommendations vs. our actual design:

| Elicit Recommendation | Our Implementation | Match? |
|----------------------|-------------------|--------|
| "Spatial, multi-scale MoE routing" | Independent per-scale token-level MoE | YES |
| "Weather router with soft weather mixture" | No weather router — label-free | NO (we go further) |
| "Condition experts for haze/rain/snow/clean" | Implicit specialization via label-free routing | NO (we let specialization emerge) |
| "Saliency-region router after initial coarse map" | Per-token router on backbone features | Different timing |
| "Routing entropy to decide expert capacity" | Routing entropy fused into decoder features | Partial match |
| "Weather classification as auxiliary supervision" | No weather labels at all | NO (we are fully label-free) |
| "Diversity/orthogonality loss" | Load balance + importance losses | Partial match |

### Key Elicit insight that strengthens our position:

Elicit's file "For dense prediction, a useful design is an MoE decoder that routes spatial tokens..." explicitly recommends using routing entropy to decide how much expert capacity to spend on each region. This supports our entropy-decoder integration as a principled design choice, even though Elicit frames it as a recommendation rather than citing a specific prior paper that does it.

### Key Elicit finding that threatens our position:

Elicit found that WM-MoE already routes without weather labels at test time via a weather feature branch. This means our "label-free" claim must be carefully qualified. The distinction is real (no weather branch in our architecture) but narrower than originally claimed.

---

*Last updated: 2026-09-01. Based on adversarial analysis of implementation + literature + Elicit evidence.*
