# FINAL_CLAIM_EVIDENCE_MATRIX.md — Claim-by-Claim Evidence Assessment

Each row: claim → our evidence → prior evidence → closest counterexample → exact difference → required experiment → confidence → recommended wording.

---

## CLAIM A: Token-level spatial MoE for single-RGB SOD

**Our implementation evidence:**
- `src/moe_layer.py:62-165`: Each spatial token gets independent routing decision via top-k selection
- `src/moe_layer.py:111-112`: `torch.topk(noisy_logits, k=self.k, dim=-1)` operates on `[B, H*W, E]`
- `src/model.py:14-16`: Three independent `SpatialMoELayer` instances, one per scale
- RESEARCH_TRUTH.md lines 20-26

**Prior-work evidence:**
- MMSOD (IJCNN 2025): MoE for SOD, routes at modality level
- CMFNet (IEEE TMM 2026): MoE for RGB-D SOD, routes at scale level
- CMoE (AAAI 2026): MoE for SOD, routes at modality level
- M4-SAM (CVPR 2026): MoE for video SOD, routes at modality level
- PSOD (IEEE TIP 2025): MoE for SOD, routes at task level
- WM-MoE (2023): Token-level routing for image restoration, not SOD
- SegMoTE (CVPR 2026): Token-level MoE for medical segmentation, not SOD
- MoE-SPNet (2018): Pixel-wise gating for scene parsing, not SOD

**Closest counterexample:** WM-MoE performs token-level routing but for image restoration (pixel regression), not SOD (pixel classification with topology constraints).

**Exact technical difference:** We route individual spatial tokens in a single-RGB-modality SOD pipeline. All existing MoE-SOD papers route at modality, task, or scale level. WM-MoE routes tokens but targets a different task.

**Required experiment:** None for the claim itself — the implementation is verified. For the paper, compare against MoE-SOD baselines (MMSOD, CMFNet) to demonstrate the value of token-level routing.

**Confidence:** HIGH

**Recommended wording:** "We apply token-level spatial MoE routing to salient object detection, where each spatial position independently selects experts based on local feature statistics. Prior MoE work in SOD routes at the modality or task level [refs], while token-level routing has been explored for image restoration [WM-MoE] and medical segmentation [SegMoTE] but not for SOD."

---

## CLAIM B: Routing without explicit weather labels or a dedicated weather-specific branch

**Our implementation evidence:**
- `src/moe_layer.py:96-112`: Router uses only spatial content features (DWConv3x3 + MLP). No weather branch, no weather labels, no CLIP features.
- `experiments/baseline_v1.json`: No weather-related config fields
- RESEARCH_TRUTH.md: No weather labels used in training pipeline

**Prior-work evidence:**
- WM-MoE (2023): Routes without weather labels at test time, BUT uses a dedicated weather feature branch trained via WGF-CL contrastive learning with weather-type labels during training
- MoFME (AAAI 2024): Uncertainty-aware routing without weather labels
- MoE-WeatherNet (2025): Decoupled content/weather gating without explicit labels
- CLIP-based methods (LDR, DA2Diff, M2Restore): Use CLIP degradation priors

**Closest counterexample:** WM-MoE achieves label-free routing at test time via a weather feature branch. However, WM-MoE uses weather labels during training (via contrastive learning) and has a dedicated weather-specific branch. Our method has NO weather-specific component at all.

**Exact technical difference:** WM-MoE: weather branch (contrastive learning with weather labels) → router. Our method: spatial content features only → router. The distinction is: (1) no weather branch exists in our architecture, (2) no weather labels are used at any stage (train or test), (3) routing emerges purely from spatial content statistics.

**Required experiment:** Ablation: add a weather branch (like WM-MoE) and show it doesn't help or hurts. Also: analyze whether the router learns weather-relevant features without supervision (interpretability analysis).

**Confidence:** MEDIUM-HIGH

**Recommended wording:** "Unlike WM-MoE [ref], which requires a dedicated weather feature branch trained via contrastive learning with weather-type labels, our router operates purely on spatial content features with no weather-specific component. No weather labels are used at any stage of training or inference."

---

## CLAIM C: Independent routers and independent expert pools at multiple feature scales

**Our implementation evidence:**
- `src/model.py:14-16`: `self.moe_4 = SpatialMoELayer(...)`, `self.moe_8 = SpatialMoELayer(...)`, `self.moe_16 = SpatialMoELayer(...)` — three completely independent instances
- Each has its own router weights, expert pool, and noise parameters
- `src/model.py:38-40`: Each processes features at its own scale independently

**Prior-work evidence:**
- WM-MoE (2023): Multi-scale experts (DWConv 1/3/5/7) but single WEAR router for all scales
- M3KE (2026): Hierarchical routing (image→patch→pixel), not independent per-scale
- HoME (2025): Two-level routing (local groups → global aggregation), but in 3D medical segmentation
- MoE-SPNet (2018): Pixel-wise gating for multi-level features, but single gating mechanism

**Closest counterexample:** WM-MoE has multi-scale experts but uses a single router. HoME has hierarchical routing (local→global) but operates in 3D medical imaging with a fundamentally different architecture.

**Exact technical difference:** We have 3 completely independent `SpatialMoELayer` instances, each with its own router MLP, expert pool, and noise parameters. Prior work either uses a single router across scales (WM-MoE) or hierarchical routing (HoME, M3KE) where later stages depend on earlier ones.

**Required experiment:** Ablation: share one router across all three scales and compare. Also: share expert pools across scales and compare.

**Confidence:** HIGH

**Recommended wording:** "Each pyramid scale has its own independent router and expert pool, allowing different routing decisions at different receptive fields. Prior multi-scale MoE work uses a single router across scales [WM-MoE] or hierarchical routing [HoME, M3KE], not independent per-scale routing."

---

## CLAIM D: Routing entropy used explicitly by the decoder

**Our implementation evidence:**
- `src/moe_layer.py:151-152`: `entropy = -torch.sum(topk_gates * torch.log(topk_gates + 1e-9), dim=-1)` → `[B, 1, H_s, W_s]`
- `src/decoder.py:14-27`: `EntropyFusionBlock` normalizes entropy by log(2), projects via Conv2d, adds to features with learnable scale
- `src/decoder.py:247-250`: Each scale's features are fused with entropy before cross-attention
- `src/model.py:44-46`: Entropy maps passed from MoE layers to decoder

**Prior-work evidence:**
- Pavlitska et al. (ICCVW 2025): Computes gate entropy per token for uncertainty estimation in MoE semantic segmentation. Used for OOD detection, NOT as decoder input feature.
- UGRAN (Yuan et al., TIP 2025): Uses uncertainty maps for decoder refinement in SOD. But this is PREDICTION uncertainty (from saliency maps), NOT routing entropy.
- GeMoE (2026): Uses gating entropy for K-selection. Not decoder input.
- Adaptive-K (2026): Uses routing entropy for dynamic K. Not decoder input.
- Lee et al. (ASE 2025): Routing entropy for OOD detection. Not decoder input.

**Closest counterexample:** Pavlitska et al. compute gate entropy per token and use it for uncertainty estimation. UGRAN uses uncertainty maps for decoder refinement. However, neither feeds routing entropy as an explicit spatial feature channel into the decoder.

**Exact technical difference:** Our `EntropyFusionBlock` takes the per-token routing entropy map, normalizes it, projects it via Conv2d, and adds it to the feature map with a learnable scale parameter. This is a specific architectural wiring choice where routing uncertainty becomes an explicit input to downstream dense prediction processing. Prior work uses entropy for OOD detection, K-selection, or routing control — not as a decoder feature.

**Required experiment:** Ablation: remove EntropyFusionBlock entirely and measure impact on SOD metrics, especially boundary performance.

**Confidence:** HIGH

**Recommended wording:** "We feed per-token routing entropy as an explicit feature channel into the decoder via a learnable fusion block. Prior work uses routing entropy for out-of-distribution detection [Pavlitska et al.] or dynamic K-selection [GeMoE, Adaptive-K], but not as an input to downstream dense prediction."

---

## CLAIM E: MoE specifically for adverse-weather SOD

**Our implementation evidence:**
- `src/model.py`: SpatialMoESODNet architecture
- `src/dataset.py`: WXSOD dataset with weather labels (used only for evaluation, not routing)
- `evaluation/`: Results on test_sys (synthetic weather) and test_real (real weather)
- RESEARCH_TRUTH.md: Only WXSOD tested

**Prior-work evidence:**
- NIFM (2025): Adverse-weather SOD on WXSOD, uses explicit weather-type encoding (not MoE)
- WFANet (2025): Two-branch baseline for adverse-weather SOD (not MoE)
- WM-MoE (2023): MoE for weather restoration (not SOD)
- BMFJNet (2024): Adverse-weather SOD for remote sensing (not MoE)
- Dehaze-SOD (2024): Joint dehazing + SOD (not MoE)

**Closest counterexample:** WM-MoE combines MoE with adverse weather but targets image restoration, not SOD. NIFM targets adverse-weather SOD but uses explicit weather encoding, not MoE.

**Exact technical difference:** We are the first to apply MoE specifically to SOD under adverse weather conditions. WM-MoE targets pixel regression (restoration); we target pixel classification with topology constraints (SOD). NIFM uses explicit weather-type labels; we are label-free.

**Required experiment:** Compare against NIFM and WFANet on WXSOD to demonstrate the value of MoE routing for adverse-weather SOD.

**Confidence:** HIGH

**Recommended wording:** "We apply mixture-of-experts specifically to salient object detection under adverse weather conditions. Prior MoE work targets image restoration [WM-MoE, MoFME], while prior adverse-weather SOD work uses explicit weather encoding [NIFM] or two-branch architectures [WFANet] without MoE routing."

---

## Summary: Claim Confidence and Recommended Wording

| Claim | Classification | Confidence | Key Qualification |
|-------|---------------|------------|-------------------|
| A. Token-level MoE for SOD | DEFENSIBLE | HIGH | Must cite WM-MoE (restoration) and SegMoTE (medical) as prior token-level MoE work in adjacent tasks |
| B. Label-free routing | DEFENSIBLE WITH QUALIFICATION | MEDIUM-HIGH | Must explicitly distinguish from WM-MoE's weather branch. Do NOT claim "first label-free" — claim "no weather-specific component" |
| C. Independent per-scale routing | DEFENSIBLE | HIGH | Must distinguish from WM-MoE (single router) and HoME (hierarchical) |
| D. Routing entropy as decoder input | DEFENSIBLE | HIGH | Must distinguish from Pavlitska et al. (OOD detection) and UGRAN (prediction uncertainty) |
| E. MoE for adverse-weather SOD | DEFENSIBLE | HIGH | Must distinguish from WM-MoE (restoration) and NIFM (explicit weather encoding) |

---

*Last updated: 2026-09-01. Based on adversarial analysis of implementation + literature + Elicit evidence.*
