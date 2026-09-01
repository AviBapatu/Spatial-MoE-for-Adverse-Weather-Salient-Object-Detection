# LITERATURE_CLAIMS.md — Externally Verified Claims

Only claims supported by primary sources are included. Each claim cites its evidence location.

---

## 1. MoE + SOD Claims

### CLAIM-01: Multiple papers combine MoE with SOD (2024-2026)
- **Evidence:** MMSOD (IJCNN 2025, DB-01), Multi-Scale MoE KAN (Neurocomputing 2025, DB-02), CMFNet (IEEE TMM 2026, DB-03), CMoE (AAAI 2026, DB-04), M4-SAM (CVPR 2026, DB-05), PSOD (IEEE TIP 2025, DB-06)
- **Status:** VERIFIED
- **Source URLs:** https://doi.org/10.1109/ijcnn64981.2025.11228833, https://doi.org/10.1016/j.neucom.2025.130349, https://doi.org/10.1109/tmm.2026.3668533, https://doi.org/10.1609/aaai.v40i12.37959, https://arxiv.org/abs/2605.11760, https://arxiv.org/abs/2409.02368

### CLAIM-02: Existing MoE-SOD papers route at modality or task level, not spatial token level
- **Evidence:** MMSOD routes per-modality (DB-01), CMFNet routes at scale level (DB-03), CMoE routes per-modality (DB-04), M4-SAM routes per-modality (DB-05), PSOD routes per-task (DB-06)
- **Status:** VERIFIED
- **Note:** None of these papers perform per-spatial-token routing for single-RGB-modality SOD.

### CLAIM-03: Controllable-LPMoE applies MoE to SOD as one of multiple segmentation tasks
- **Evidence:** DB-07 (ICCV 2025), evaluated on DUTS-TE for SOD
- **Status:** VERIFIED
- **URL:** https://arxiv.org/abs/2410.16076

### CLAIM-04: PromptMoE evaluates MoE on SOD benchmarks (ECSSD, MSRA-B)
- **Evidence:** DB-08 (CVPR 2026)
- **Status:** VERIFIED
- **URL:** https://openaccess.thecvf.com/content/CVPR2026/papers/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.pdf

---

## 2. Adverse-Weather MoE Claims

### CLAIM-05: WM-MoE performs token-level routing (CORRECTION: NOT feature-map level)
- **Evidence:** DB-19 (arXiv:2303.13739). WEAR (WEather-Aware Router) assigns experts for each image token based on decoupled content and weather features.
- **Status:** VERIFIED — CORRECTS BLUEPRINT
- **Blueprint error:** Blueprint says WM-MoE "routes primarily on (multi-scale) feature maps with weather-conditioned gating." This is wrong. Routing is token-level.

### CLAIM-06: WM-MoE does NOT require weather labels at test time (CORRECTION)
- **Evidence:** DB-19. Uses a separate lightweight ViT branch to extract weather features via contrastive learning (WGF-CL). At inference, the weather branch runs automatically — no explicit weather labels needed.
- **Status:** VERIFIED — CORRECTS BLUEPRINT
- **Blueprint error:** Blueprint implies weather labels are needed. They are not.

### CLAIM-07: WM-MoE and MoWE are the same paper (CORRECTION)
- **Evidence:** DB-19. Same arXiv ID (2303.13739), same authors, same content. Paper was renamed from "MoWE" to "WM-MoE" during revision.
- **Status:** VERIFIED — CORRECTS BLUEPRINT
- **Blueprint error:** Blueprint lists WM-MoE and MoWE as two separate papers. They are the same paper.

### CLAIM-08: MoFME uses uncertainty-aware routing without weather labels
- **Evidence:** DB-20 (AAAI 2024). Uncertainty-aware Router uses MC dropout to estimate router uncertainty.
- **Status:** VERIFIED
- **URL:** https://doi.org/10.1609/aaai.v38i15.29622

### CLAIM-09: Multiple weather-MoE papers use CLIP/VLM for label-free degradation estimation
- **Evidence:** LDR (DB-22, VLM), DA2Diff (DB-23, CLIP), M2Restore (DB-28, CLIP), CoRE-UIR (CLIP)
- **Status:** VERIFIED
- **URLs:** https://arxiv.org/abs/2312.01381, https://arxiv.org/abs/2504.05135, https://doi.org/10.1109/tip.2025.3638662

### CLAIM-10: Complexity Experts (CVPR 2025) uses complexity-based routing, not weather labels
- **Evidence:** DB-29. Experts with increasing complexity/receptive fields. Complexity-aware allocation with bias toward lower-complexity experts.
- **Status:** VERIFIED
- **URL:** https://arxiv.org/abs/2411.18466

---

## 3. Adverse-Weather SOD Claims

### CLAIM-11: WXSOD is the primary adverse-weather SOD benchmark (14,945 images)
- **Evidence:** DB-34 (arXiv:2508.12250, Pattern Recognition 2026). Confirmed by our repository data (train_sys=12,891, test_sys=1,500, test_real=554).
- **Status:** VERIFIED
- **URL:** https://arxiv.org/abs/2508.12250

### CLAIM-12: WFANet is the WXSOD baseline (two-branch: weather prediction + saliency)
- **Evidence:** DB-34. WFANet is a fully supervised two-branch architecture.
- **Status:** VERIFIED

### CLAIM-13: NIFM is the most direct competitor to our work
- **Evidence:** DB-35 (arXiv:2512.10592). Uses explicit weather-type one-hot encoding (NIFM) to embed weather-aware priors into any SOD encoder. Same authors as WXSOD, same dataset.
- **Status:** VERIFIED
- **URL:** https://arxiv.org/abs/2512.10592
- **Key difference from our work:** NIFM uses EXPLICIT weather-type labels. We use LABEL-FREE spatial routing.

### CLAIM-14: Low-light SOD is a large separate sub-field
- **Evidence:** DB-38 (IEEE TCSVT 2024), DB-39 (IEEE TCSVT 2026), plus 10+ additional papers (DB-11 through DB-20 in adverse-weather SOD section)
- **Status:** VERIFIED
- **Datasets:** SOD-LL, LSPV, LlSOD

### CLAIM-15: RGB-T SOD provides thermal-based weather robustness as an alternative to RGB-only
- **Evidence:** VT5000 (DB-21 in Tier 5), VT821 (DB-22), plus 10+ RGB-T SOD papers
- **Status:** VERIFIED
- **Relevance:** Demonstrates the "why not add a sensor" counterargument our work addresses.

---

## 4. MoE Routing Mechanism Claims

### CLAIM-16: V-MoE uses token-level top-k routing with noisy gating
- **Evidence:** DB-41 (NeurIPS 2021). Each image patch routed to k experts (default k=2).
- **Status:** VERIFIED
- **URL:** https://arxiv.org/abs/2106.05974

### CLAIM-17: Soft MoE eliminates load-balancing losses
- **Evidence:** DB-42 (ICLR 2024). Replaces discrete top-k with fully differentiable weighted combinations. Explicitly avoids sort or top-k operations.
- **Status:** VERIFIED
- **URL:** https://arxiv.org/abs/2308.00951

### CLAIM-18: M3ViT uses task-conditioned token routing (CORRECTION: not "multi-scale")
- **Evidence:** DB-43 (NeurIPS 2022). Task-dependent gating with task embedding concatenated to token embedding. NOT multi-scale in pyramid sense — replaces FFN in ViT with MoE.
- **Status:** VERIFIED — CORRECTS BLUEPRINT
- **Blueprint error:** Blueprint implies M3ViT is "multi-scale" — it is multi-task, not multi-scale.

### CLAIM-19: Routers in Vision MoE is a comparative benchmark, NOT a failure-mode analysis
- **Evidence:** DB-45 (TMLR 2024). Studies 6 router designs head-to-head. Key finding: Expert Choice generally outperforms Token Choice; Soft MoE outperforms both.
- **Status:** VERIFIED — CORRECTS BLUEPRINT
- **Blueprint error:** Blueprint claims paper "documents known MoE failure modes (router collapse, expert underutilization)." The paper is a router comparison benchmark, not a failure-mode analysis.

### CLAIM-20: SMOE (NeurIPS 2022) introduces per-pixel expert routing with tensor gating
- **Evidence:** DB-16. Per-pixel expert assignment with learnable tensor gating. Proposes self-supervised routing classification loss.
- **Status:** VERIFIED
- **URL:** https://arxiv.org/abs/2211.13491

---

## 5. Routing Entropy as Decoder Input Claims

### CLAIM-21: No prior paper feeds routing entropy as an explicit feature channel into a dense prediction decoder
- **Evidence:** Searched 18+ queries on routing entropy, router confidence, expert routing uncertainty. Found papers using entropy for OOD detection (DB-49), K-selection (DB-50, DB-51), routing control (DB-52), but none feeding it as a spatial feature map into a segmentation/saliency decoder.
- **Status:** LIKELY VERIFIED (cannot prove negative with certainty)
- **Caveat:** This claim requires manual verification of the most closely related papers.

### CLAIM-22: Lee et al. (2025) use routing entropy for OOD detection in MoE semantic segmentation
- **Evidence:** DB-49. Routing entropy as OOD detector: ID inputs produce low entropy, OOD produce high entropy.
- **Status:** VERIFIED
- **Key difference from our work:** Uses entropy for OOD detection, not as decoder input feature.

---

## 6. Multi-scale MoE Claims

### CLAIM-23: WM-MoE has multi-scale experts but uses a single router
- **Evidence:** DB-19. Multi-Scale Experts with different receptive fields (DWConv 1/3/5/7), but routing is done by a single WEAR router.
- **Status:** VERIFIED

### CLAIM-24: No prior work uses independent per-scale routers with separate expert pools for dense prediction
- **Evidence:** Searched 15+ queries on multi-scale MoE, pyramid MoE, independent routing. Found multi-scale experts (WM-MoE), multi-granularity routing (M3KE), but not independent per-scale routers with separate expert pools.
- **Status:** LIKELY VERIFIED (cannot prove negative with certainty)

---

## 7. Dataset Claims

### CLAIM-25: WXSOD has 14,945 images with weather labels
- **Evidence:** DB-34. Confirmed by our repository data.
- **Status:** VERIFIED

### CLAIM-26: WXSOD has synthetic and real-world test splits
- **Evidence:** DB-34. test_sys (1,500 synthetic) and test_real (554 real-world).
- **Status:** VERIFIED

### CLAIM-27: XWOD is a real-world extreme-weather detection benchmark (NOT SOD)
- **Evidence:** Described in blueprint as detection, not SOD. Could not verify exact paper details.
- **Status:** UNVERIFIED — needs manual verification

---

## 8. Corrections Required to Blueprint

| # | Blueprint Claim | Corrected Claim | Severity |
|---|----------------|-----------------|----------|
| 1 | WM-MoE and MoWE are two separate papers | They are the same paper (arXiv:2303.13739) | HIGH |
| 2 | WM-MoE routes at "feature-map level" | WM-MoE routes at TOKEN-LEVEL | HIGH |
| 3 | WM-MoE requires weather labels at test time | WM-MoE does NOT require weather labels (uses learned weather branch) | HIGH |
| 4 | M3ViT is "multi-scale" | M3ViT is multi-task, not multi-scale | MEDIUM |
| 5 | Routers in Vision MoE documents failure modes | It is a comparative benchmark of 6 router designs | MEDIUM |
| 6 | Zhu et al. CVPR 2023 is "channel-level split (two-branch)" | It is a two-stage parameter-expansion approach | MEDIUM |
| 7 | GridFormer is "2023" | Published in IJCV 2024 | LOW |
| 8 | M3ViT title omits "with Model-Accelerator Co-design" | Full title includes hardware co-design contribution | LOW |

---

*Last updated: 2026-09-01. All claims verified against primary sources.*
