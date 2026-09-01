# LITERATURE_VERIFICATION_QUEUE.md — Updated Verification Status

Items from the blueprint that required external literature verification.
Status updated 2026-09-01.

---

## Priority 1: Novelty Claims

### 1.1 "First token-level MoE routing for SOD"
- **Claim:** No prior work applies token-level (per-spatial-position) MoE routing to salient object detection.
- **Status:** LIKELY VERIFIED
- **Evidence:** Searched 13 queries on MoE + SOD. Found 6 papers combining MoE with SOD (MMSOD, Multi-Scale MoE KAN, CMFNet, CMoE, M4-SAM, PSOD). ALL route at modality/task/scale level, NOT at spatial token level for single-RGB SOD. WM-MoE does token-level routing but for image restoration, not SOD. SegMoTE (CVPR 2026) does token-level MoE for medical segmentation, not SOD.
- **Caveat:** Cannot prove negative with certainty. 2026 preprints may exist.
- **Source:** LITERATURE_DATABASE.md (DB-01 through DB-06, DB-19, DB-12)

### 1.2 "First label-free weather routing"
- **Claim:** Prior weather-aware MoE methods (WM-MoE, MoWE) use weather-type labels/embeddings at train or test time. Our method routes without any weather label.
- **Status:** PARTIALLY DISPROVEN
- **Evidence:** WM-MoE (arXiv:2303.13739) does NOT require weather labels at test time. It uses a dedicated weather feature branch via contrastive learning (WGF-CL). At inference, the weather branch runs automatically — no explicit weather labels needed. Additionally, MoFME uses uncertainty-aware routing (no labels), and CLIP-based methods (LDR, DA2Diff, M2Restore) use CLIP degradation priors (no training labels).
- **Corrected claim:** Our method is DISTINCT from WM-MoE in that we have NO weather-specific branch at all — routing emerges purely from spatial content features. WM-MoE's weather branch constitutes implicit weather supervision via contrastive learning, whereas our method has no weather-specific component.
- **Source:** LITERATURE_CLAIMS.md (CLAIM-06, CLAIM-07)

### 1.3 "First independent multi-scale MoE routing"
- **Claim:** No prior work routes independently at each pyramid scale with separate routers and expert pools.
- **Status:** LIKELY VERIFIED
- **Evidence:** Searched 15+ queries on multi-scale MoE. WM-MoE has multi-scale experts but uses a single router. M3KE has hierarchical routing (image->patch->pixel), not independent per-scale. MoE-SPNet uses pixel-wise gating for multi-level features but not independent routers. Rate-limited on "independent expert routing" query — verified via indirect evidence.
- **Caveat:** Could not search "independent per-scale routing" directly due to rate limits.
- **Source:** LITERATURE_CLAIMS.md (CLAIM-24)

### 1.4 "Router entropy as decoder input is novel"
- **Claim:** No prior work feeds routing entropy/confidence back into the decoder as an explicit feature channel.
- **Status:** LIKELY VERIFIED
- **Evidence:** Searched 18 queries on routing entropy, router confidence, expert routing uncertainty. Found papers using entropy for OOD detection (Lee et al. 2025), K-selection (GeMoE 2026, Adaptive-K 2026), routing control (UAR 2026, GrMoE 2026), but NONE feed entropy as an explicit spatial feature map into a dense prediction decoder.
- **Caveat:** Cannot prove negative with certainty. Manual verification of closest papers recommended.
- **Source:** LITERATURE_CLAIMS.md (CLAIM-21, CLAIM-22)

### 1.5 "MoE for adverse-weather SOD is novel"
- **Claim:** No prior work applies MoE specifically to weather-robust SOD.
- **Status:** LIKELY VERIFIED
- **Evidence:** Found 8 papers targeting adverse-weather SOD (WXSOD/WFANet, NIFM, BMFJNet, Dehaze-SOD, plus low-light and RGB-T papers). NONE use MoE. NIFM uses explicit weather-type encoding. WFANet uses two-branch architecture.
- **Source:** LITERATURE_CLAIMS.md (CLAIM-13, CLAIM-14)

---

## Priority 2: Dataset Claims

### 2.1 WXSOD dataset size
- **Claim:** "14,945 RGB images with pixel-wise saliency ground truth plus per-image weather labels"
- **Status:** VERIFIED
- **Evidence:** Confirmed via arXiv:2508.12250 and our repository data (train_sys=12,891, test_sys=1,500, test_real=554 = 14,945 total).
- **Source:** LITERATURE_CLAIMS.md (CLAIM-25, CLAIM-26)

### 2.2 WXSOD baseline (WFANet)
- **Claim:** "Their baseline, WFANet, is a two-branch (restore + segment) network"
- **Status:** VERIFIED
- **Evidence:** WFANet is a fully supervised two-branch architecture (weather prediction branch + saliency detection branch).
- **Source:** LITERATURE_CLAIMS.md (CLAIM-12)

### 2.3 XWOD dataset
- **Claim:** "Real-world extreme-weather detection benchmark (rain/snow/fog/haze-sand-dust/flood/tornado/wildfire, 10K images)"
- **Status:** UNVERIFIED
- **Evidence:** Described in blueprint as detection, not SOD. Could not verify exact paper details via web search.
- **Action needed:** Manual verification of XWOD paper.

---

## Priority 3: Method Comparison Claims

### 3.1 WM-MoE routing granularity
- **Claim:** "WM-MoE routes primarily on (multi-scale) feature maps with weather-conditioned gating"
- **Status:** DISPROVEN — CORRECTION REQUIRED
- **Evidence:** WM-MoE routes at TOKEN-LEVEL (per-patch via WEAR), not "feature-map level." The blueprint mischaracterizes the routing granularity.
- **Corrected claim:** WM-MoE routes at token level with weather-aware conditioning via a learned weather feature branch.
- **Source:** LITERATURE_CLAIMS.md (CLAIM-05)

### 3.2 MoWE routing granularity
- **Claim:** "MoWE: image/patch level" routing
- **Status:** CORRECTION REQUIRED — MoWE IS WM-MoE
- **Evidence:** MoWE and WM-MoE are the SAME paper (arXiv:2303.13739). Same authors, same content. Paper was renamed during revision.
- **Corrected claim:** Remove MoWE as a separate entry. Reference only as WM-MoE.
- **Source:** LITERATURE_CLAIMS.md (CLAIM-07)

### 3.3 Zhu et al. (CVPR 2023) routing
- **Claim:** "Channel (branch-level)" routing
- **Status:** PARTIALLY DISPROVEN — CORRECTION REQUIRED
- **Evidence:** Zhu et al. CVPR 2023 is NOT a two-branch channel-level split. It uses a two-stage training strategy on a U-Net backbone: Stage 1 learns weather-general features; Stage 2 adaptively expands weather-specific parameters at automatically learned positions. This is parameter expansion, not channel-level routing.
- **Corrected claim:** Zhu et al. uses two-stage parameter expansion in a shared U-Net, not a fixed two-branch architecture.
- **Source:** LITERATURE_CLAIMS.md (CLAIM-18)

### 3.4 Complexity Experts (CVPR 2025)
- **Claim:** "Experts are specialized by computational complexity rather than task/weather label"
- **Status:** VERIFIED
- **Evidence:** Zamfir et al. CVPR 2025 introduces "complexity experts" with varying computational complexity and receptive fields. Complexity-aware allocation with bias toward lower-complexity experts.
- **Source:** LITERATURE_CLAIMS.md (CLAIM-10)

---

## Priority 4: Technical Claims

### 4.1 Soft MoE eliminates load-balancing losses
- **Claim:** Soft MoE "replaces hard top-k assignment with a fully differentiable weighted combination, removing load-balancing losses and token-dropping issues entirely"
- **Status:** VERIFIED
- **Evidence:** Soft MoE (ICLR 2024) explicitly avoids sort or top-k operations. Fully differentiable weighted combinations eliminate the need for load-balancing losses.
- **Source:** LITERATURE_CLAIMS.md (CLAIM-17)

### 4.2 Routers in Vision MoE failure modes
- **Claim:** Paper "documents known MoE failure modes (router collapse, expert underutilization)"
- **Status:** OVERSTATED — CORRECTION REQUIRED
- **Evidence:** The paper is a comparative benchmark of 6 router designs, not a failure-mode analysis. It mentions mode collapse in passing but does not systematically document failure modes.
- **Corrected claim:** The paper compares 6 router designs head-to-head. Expert Choice routers generally outperform Token Choice; Soft MoE outperforms both sparse variants.
- **Source:** LITERATURE_CLAIMS.md (CLAIM-19)

### 4.3 GridFormer year
- **Claim:** "GridFormer (2023)"
- **Status:** CORRECTION REQUIRED
- **Evidence:** Published in IJCV 2024, not 2023. arXiv preprint was 2023.
- **Corrected claim:** GridFormer (IJCV 2024)
- **Source:** LITERATURE_CLAIMS.md (DB-31)

---

## Summary: Verification Status

| Priority | # Items | Verified | Disproven/Corrected | Pending |
|----------|---------|----------|---------------------|---------|
| CRITICAL | 2 | 1 (token-level MoE for SOD: likely verified) | 1 (label-free routing: partially disproven) | 0 |
| HIGH | 5 | 2 (independent per-scale: likely verified; router entropy: likely verified) | 3 (WM-MoE granularity, MoWE duplicate, Routers paper overstated) | 0 |
| MEDIUM | 6 | 4 (WXSOD size, WFANet, Soft MoE, Complexity Experts) | 2 (Zhu et al. architecture, GridFormer year) | 1 (XWOD) |
| LOW | 2 | 1 (PVTv2-B4 specs) | 0 | 1 (Swin-B not tested) |
| **Total** | **15** | **8** | **6** | **2** |

**Key corrections required to blueprint:**
1. WM-MoE and MoWE are the same paper
2. WM-MoE routes at token level, not feature-map level
3. WM-MoE does not require weather labels at test time
4. Zhu et al. is parameter expansion, not two-branch
5. Routers in Vision MoE is a benchmark, not failure-mode analysis
6. GridFormer was published in IJCV 2024

---

*Last updated: 2026-09-01.*
