# NOVELTY_MATRIX.md — Comparative Analysis vs. Prior Work

Each row compares our Spatial-MoE SOD method against a specific prior paper across key dimensions.

---

## Key Dimensions

| Column | Description |
|--------|-------------|
| Paper | Prior work being compared |
| Task | What the paper targets |
| MoE | Whether it uses MoE |
| Routing granularity | Image / patch / token / feature-map / channel / modality / task |
| Routing signal | What drives routing decisions (weather labels, CLIP, contrastive, uncertainty, complexity, etc.) |
| Weather supervision | Whether weather labels are required at train or test time |
| Multi-scale | Whether it uses multi-scale features |
| Independent scale routing | Whether each scale has its own router and expert pool |
| Decoder use of routing confidence | Whether routing entropy/confidence is fed into the decoder |
| SOD | Whether it targets SOD |
| Adverse-weather SOD | Whether it targets adverse-weather SOD |
| Similarity | What is similar to our method |
| Difference | What is different from our method |
| Novelty implication | How this affects our novelty claims |

---

## COMPARISON TABLE

### Row 1: WM-MoE (Luo et al., 2023)
- **Task:** Image restoration (deraining, dehazing, desnowing)
- **MoE:** Yes
- **Routing granularity:** Token-level (per-patch via WEAR)
- **Routing signal:** Decoupled content + weather features via WGF-CL contrastive learning
- **Weather supervision:** No labels at test time (learned weather branch)
- **Multi-scale:** Yes (Multi-Scale Experts with DWConv 1/3/5/7)
- **Independent scale routing:** No (single router)
- **Decoder use of routing confidence:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Similarity:** Token-level routing, multi-scale experts, weather-agnostic at test time
- **Difference:** (1) Restoration, not SOD. (2) Uses dedicated weather branch (contrastive learning) — not fully label-free. (3) Single router across scales, not independent per-scale routers.
- **Novelty implication:** Our work is DISTINCT on task (SOD vs restoration), routing architecture (independent per-scale routers), and weather supervision (fully label-free, no weather branch).

### Row 2: MoFME (Zhang et al., AAAI 2024)
- **Task:** Multi-deweather
- **MoE:** Yes
- **Routing granularity:** Token-level
- **Routing signal:** Uncertainty-aware (MC dropout)
- **Weather supervision:** No
- **Multi-scale:** No
- **Independent scale routing:** No
- **Decoder use of routing confidence:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Similarity:** Token-level routing, no weather labels
- **Difference:** (1) Restoration, not SOD. (2) Uses MC dropout uncertainty, not routing entropy as decoder input. (3) No multi-scale routing.
- **Novelty implication:** Our work is DISTINCT on task and decoder integration of routing confidence.

### Row 3: Complexity Experts (Zamfir et al., CVPR 2025)
- **Task:** All-in-one image restoration
- **MoE:** Yes
- **Routing granularity:** Image-level (noisy top-1)
- **Routing signal:** Complexity-based (increasing receptive fields)
- **Weather supervision:** No
- **Multi-scale:** No (complexity-based, not spatial pyramid)
- **Independent scale routing:** No
- **Decoder use of routing confidence:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Similarity:** Implicit expert specialization, label-free routing
- **Difference:** (1) Restoration, not SOD. (2) Image-level routing, not spatial token-level. (3) Complexity-based, not content/weather-based.
- **Novelty implication:** Our work is DISTINCT on routing granularity and task.

### Row 4: NIFM (Chen et al., 2025)
- **Task:** Adverse-weather SOD
- **MoE:** No
- **Routing granularity:** N/A (no routing — uses one-hot weather-type encoding)
- **Routing signal:** Explicit weather-type one-hot vector
- **Weather supervision:** Yes (requires weather-type labels)
- **Multi-scale:** No (applied to encoder features)
- **Independent scale routing:** No
- **Decoder use of routing confidence:** No
- **SOD:** Yes
- **Adverse-weather SOD:** Yes
- **Similarity:** Same task (adverse-weather SOD), same benchmark (WXSOD)
- **Difference:** (1) Uses explicit weather-type labels, we are label-free. (2) No MoE routing — simple feature encoding. (3) No spatial routing — global weather-type vector.
- **Novelty implication:** Our work is DISTINCT on method (spatial MoE vs one-hot encoding) and weather supervision (label-free vs label-dependent).

### Row 5: WFANet (Chen et al., 2025)
- **Task:** Adverse-weather SOD
- **MoE:** No
- **Routing granularity:** N/A (two-branch: weather prediction + saliency)
- **Routing signal:** Image-level weather prediction branch
- **Weather supervision:** Yes (trained with weather labels)
- **Multi-scale:** No
- **Independent scale routing:** No
- **Decoder use of routing confidence:** No
- **SOD:** Yes
- **Adverse-weather SOD:** Yes
- **Similarity:** Same task, same benchmark
- **Difference:** (1) Two-branch at image level, not spatial token-level MoE. (2) Requires weather labels, we are label-free. (3) No expert specialization.
- **Novelty implication:** Our work is DISTINCT — fundamentally different architecture (spatial MoE vs two-branch).

### Row 6: V-MoE (Riquelme et al., NeurIPS 2021)
- **Task:** Image classification
- **MoE:** Yes
- **Routing granularity:** Token-level (top-k)
- **Routing signal:** Learned router MLP
- **Weather supervision:** No
- **Multi-scale:** No (single-scale ViT)
- **Independent scale routing:** No
- **Decoder use of routing confidence:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Similarity:** Foundational token-level top-k routing mechanism
- **Difference:** (1) Classification, not dense prediction. (2) No multi-scale routing. (3) No routing entropy in decoder.
- **Novelty implication:** Our work builds on V-MoE but is DISTINCT in task and architecture.

### Row 7: Soft MoE (Puigcerver et al., ICLR 2024)
- **Task:** Image classification
- **MoE:** Yes
- **Routing granularity:** Token-level (differentiable weighted combination)
- **Routing signal:** Learnable tensor
- **Weather supervision:** No
- **Multi-scale:** No
- **Independent scale routing:** No
- **Decoder use of routing confidence:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Similarity:** Alternative router design (soft vs hard)
- **Difference:** (1) Classification, not dense prediction. (2) Fully differentiable, no sparsity.
- **Novelty implication:** Our work is DISTINCT — we use hard top-k, not soft routing.

### Row 8: M3ViT (Liang et al., NeurIPS 2022)
- **Task:** Multi-task learning (classification + detection + segmentation)
- **MoE:** Yes
- **Routing granularity:** Token-level with task embedding
- **Routing signal:** Task-conditioned (task embedding concatenated to token)
- **Weather supervision:** No
- **Multi-scale:** No (multi-task, not multi-scale)
- **Independent scale routing:** No
- **Decoder use of routing confidence:** No
- **SOD:** No (includes segmentation but not SOD-specific)
- **Adverse-weather SOD:** No
- **Similarity:** Token-level routing for dense prediction
- **Difference:** (1) Multi-task, not multi-scale. (2) Task-conditioned, not content-conditioned. (3) No routing entropy in decoder.
- **Novelty implication:** Our work is DISTINCT — independent per-scale routing, not task-conditioned.

### Row 9: SegMoTE (Lu et al., CVPR 2026 Oral)
- **Task:** Medical image segmentation
- **MoE:** Yes
- **Routing granularity:** Token-level (per-modality expert tokens)
- **Routing signal:** Modality-aware
- **Weather supervision:** No
- **Multi-scale:** No
- **Independent scale routing:** No
- **Decoder use of routing confidence:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Similarity:** Token-level MoE for segmentation
- **Difference:** (1) Medical segmentation, not SOD. (2) Modality-aware, not weather/content-aware. (3) No multi-scale routing.
- **Novelty implication:** Our work is DISTINCT on task and routing signal.

### Row 10: Rule-Based Spatial MoE for Edge Detection (2026)
- **Task:** Edge detection
- **MoE:** Yes
- **Routing granularity:** Spatial (local feature statistics)
- **Routing signal:** Context vs boundary experts
- **Weather supervision:** No
- **Multi-scale:** No
- **Independent scale routing:** No
- **Decoder use of routing confidence:** No
- **SOD:** No (edge detection, related)
- **Adverse-weather SOD:** No
- **Similarity:** Spatial expert routing for dense prediction, interpretable gating
- **Difference:** (1) Edge detection, not SOD. (2) Only 2 experts (context vs boundary), not 8. (3) No multi-scale routing.
- **Novelty implication:** Our work is DISTINCT — more experts, multi-scale, SOD-specific.

### Row 11: M3KE (Qiao et al., 2026)
- **Task:** Nighttime dehazing
- **MoE:** Yes
- **Routing granularity:** Multi-granularity (image/patch/pixel)
- **Routing signal:** Frequency-aware router
- **Weather supervision:** No
- **Multi-scale:** Yes (multi-granularity)
- **Independent scale routing:** No (hierarchical, not independent)
- **Decoder use of routing confidence:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Similarity:** Multi-granularity routing, label-free
- **Difference:** (1) Dehazing, not SOD. (2) Hierarchical routing (image->patch->pixel), not independent per-scale. (3) No routing entropy in decoder.
- **Novelty implication:** Our work is DISTINCT — independent per-scale routing, not hierarchical.

---

## NOVELTY HYPOTHESIS ASSESSMENT

### Hypothesis A: Token-level MoE applied to SOD
- **Closest priors:** MMSOD (DB-01), Multi-Scale MoE KAN (DB-02), CMFNet (DB-03), CMoE (DB-04), M4-SAM (DB-05), PSOD (DB-06)
- **Exact overlap:** All these papers apply MoE to SOD
- **Exact technical difference:** They route at modality/task/scale level, NOT at spatial token level for single-RGB-modality SOD. WM-MoE does token-level routing but for image restoration, not SOD.
- **Prior work with essentially same idea:** No paper applies token-level spatial routing to single-RGB SOD
- **Distinction:** Task difference (SOD vs restoration) + routing granularity (token vs modality/task)
- **Classification:** **LIKELY DISTINCT**
- **Additional evidence needed:** Verify no 2026 preprint does token-level MoE for SOD

### Hypothesis B: Label-free weather/degradation routing
- **Closest priors:** WM-MoE (DB-19), MoFME (DB-20), MoE-WeatherNet (DB-24), LDR (DB-22), DA2Diff (DB-23), M2Restore (DB-28)
- **Exact overlap:** WM-MoE already routes without weather labels at test time (via contrastive learning weather branch). MoFME uses uncertainty-aware routing. CLIP-based methods use CLIP priors.
- **Exact technical difference:** WM-MoE uses a dedicated weather feature branch (contrastive learning). Our method has NO weather-specific branch at all — routing emerges purely from spatial content features.
- **Prior work with essentially same idea:** WM-MoE achieves label-free routing via a weather branch. Our claim of "fully label-free" is DISTINCT from WM-MoE's "weather-branch-assisted" approach.
- **Classification:** **POTENTIALLY DISTINCT — MORE SEARCH REQUIRED**
- **Additional evidence needed:** Verify whether WM-MoE's weather branch is truly "label-free" or whether contrastive learning supervision constitutes implicit weather labeling. Also verify whether any CLIP-based method achieves the same level of label-freedom.

### Hypothesis C: Independent MoE routing at multiple pyramid scales
- **Closest priors:** WM-MoE (single router, multi-scale experts), M3KE (hierarchical, not independent), MoE-SPNet (pixel-wise gating for multi-level features)
- **Exact overlap:** No paper uses independent per-scale routers with separate expert pools
- **Exact technical difference:** WM-MoE has one router for all scales. M3KE has hierarchical routing. Our system has 3 independent routers (one per scale) with separate expert pools.
- **Prior work with essentially same idea:** No
- **Classification:** **LIKELY DISTINCT**
- **Additional evidence needed:** Verify no 2026 preprint uses independent per-scale routing

### Hypothesis D: Routing entropy/confidence fed explicitly into the decoder
- **Closest priors:** Lee et al. (2025, OOD detection), GeMoE (2026, K-selection), Adaptive-K (2026, K-selection), UAR (2026, routing control)
- **Exact overlap:** These papers use routing entropy for OOD detection, K-selection, or routing control — NOT as an explicit feature channel in the decoder
- **Exact technical difference:** Our system concatenates per-token routing entropy as an extra channel at each scale before the final decoder conv. No prior work does this.
- **Prior work with essentially same idea:** No
- **Classification:** **LIKELY DISTINCT**
- **Additional evidence needed:** Verify no prior work feeds routing confidence back as a spatial feature map into a dense prediction decoder

### Hypothesis E: MoE specifically for adverse-weather SOD
- **Closest priors:** WXSOD/WFANet (DB-34), NIFM (DB-35), BMFJNet (DB-36), Dehaze-SOD (DB-37)
- **Exact overlap:** These papers target adverse-weather SOD
- **Exact technical difference:** None use MoE. NIFM uses explicit weather-type encoding. WFANet uses two-branch architecture.
- **Prior work with essentially same idea:** No paper combines MoE with SOD under adverse weather
- **Classification:** **LIKELY DISTINCT**
- **Additional evidence needed:** Verify no 2026 preprint combines MoE with adverse-weather SOD

---

## OVERALL NOVELTY ASSESSMENT

| Hypothesis | Classification | Confidence |
|------------|---------------|------------|
| A. Token-level MoE for SOD | LIKELY DISTINCT | High |
| B. Label-free weather routing | POTENTIALLY DISTINCT — MORE SEARCH REQUIRED | Medium |
| C. Independent per-scale routing | LIKELY DISTINCT | High |
| D. Routing entropy as decoder input | LIKELY DISTINCT | High |
| E. MoE for adverse-weather SOD | LIKELY DISTINCT | High |

**Strongest novelty claim:** D (routing entropy as decoder input) — no prior work found that does this.

**Most threatened claim:** B (label-free routing) — WM-MoE achieves label-free routing via a weather branch, which is conceptually similar to our approach.

**Recommended claim combination:** The strongest novelty is the COMBINATION of A+C+D+E: token-level MoE routing applied to SOD, with independent per-scale routers, feeding routing entropy into the decoder, all without any weather labels. The combination of these elements is novel even if individual elements have partial precedents.

---

*Last updated: 2026-09-01. Based on verified literature comparisons.*
