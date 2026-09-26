A strong formulation is **weather-conditioned sparse mixture-of-experts for salient object detection**, rather than simply adding more experts to a standard SOD model.

The closest benchmark I found is WXSOD: 14,945 RGB images with weather labels and pixel-wise saliency masks, split into synthetic and real-weather test sets. Its baseline uses a weather-prediction branch whose features are fused into the saliency branch, which supports making weather condition an explicit conditioning signal rather than an implicit nuisance [^1].

**Proposed architecture**

1. **Shared encoder:** a hierarchical CNN/transformer backbone producing multi-scale features $$F_1, F_2, F_3, F_4$$.
2. **Weather router:** global pooled features go to a lightweight classifier/gate that estimates a soft weather mixture $$g_w \in \mathbb{R}^K$$ for clear, rain, fog, snow, and mixed/unknown conditions. Use top-2 routing rather than hard argmax so ambiguous weather does not cause abrupt expert switching.
3. **Condition experts:** each expert specializes in a distinct degradation regime—e.g., low-contrast/fog, streak/occlusion/rain, snow/bright clutter, and a clean-scene expert. Apply experts mainly at intermediate and high-resolution feature levels; keeping the stem shared limits compute and prevents weather experts from learning unrelated low-level filters.
4. **Saliency-region router:** after an initial coarse saliency map, route spatial tokens or decoder queries to experts using local degradation statistics plus semantic features. This is preferable to image-only routing because one frame can contain locally different visibility and because SOD is a dense prediction problem.
5. **Decoder:** fuse the routed multi-scale features with cross-scale attention and predict saliency, boundary, and uncertainty maps.

This two-stage design is consistent with recent MoE work: dynamic routing must solve both expert assignment and expert specialization, otherwise routing can collapse onto the same subset of experts [^2]. A scene-level router followed by instance/query-level routing has also been proposed for detection because image- or patch-level routing is poorly aligned with object-centric reasoning [^3]; for SOD, the analogous second stage should operate on salient-region tokens or spatial windows.

**Training objective**

$$\mathcal{L}=\mathcal{L}_{\text{sal}}+\lambda_b\mathcal{L}_{\text{boundary}}+\lambda_w\mathcal{L}_{\text{weather}}+\lambda_d\mathcal{L}_{\text{diversity}}+\lambda_e\mathcal{L}_{\text{entropy}}$$

Use structure-aware saliency loss plus boundary loss for the mask; weather classification as auxiliary supervision; a diversity/orthogonality loss to prevent experts from becoming interchangeable; and a load-balancing term to prevent routing collapse. The gate should be mildly entropy-regularized early in training, then sharpened or temperature-annealed later. If weather labels are unavailable, use pseudo-labels from degradation statistics and consistency across augmented views.

**Most important ablations**

- Dense shared model vs. MoE with no routing supervision.
- Weather-only routing vs. spatial/region-level routing vs. hierarchical routing.
- Hard top-1 vs. soft top-2 routing.
- Experts at encoder levels only vs. encoder plus decoder.
- With and without diversity/load-balancing losses.
- Synthetic-weather training/testing vs. synthetic-to-real transfer.
- Separate experts for weather type vs. severity-conditioned continuous routing.

Report S-measure, mean F-measure, weighted F-measure, MAE, boundary F-score, parameter count, activated FLOPs, latency, and calibration/error by weather type. Do not rely only on synthetic corruption: WXSOD explicitly separates synthesized weather from real-world weather, so the real-weather split is the key generalization test [^1].

The main risk is **expert collapse or shortcut learning**: the router may identify weather correctly without improving saliency, while experts simply memorize dataset-specific appearance. The decisive experiment is therefore cross-weather generalization with held-out weather combinations and real-weather images—not an in-distribution gain on WXSOD alone. The literature currently supports the architecture as a principled design direction, but the retrieved evidence does not establish that MoE routing itself has already been validated for adverse-weather SOD; the SOD benchmark provides the task foundation, while the routing evidence comes mainly from broader vision and detection settings.

[^1]: Chen et al., 2025. WXSOD: A Benchmark for Robust Salient Object Detection in Adverse Weather Conditions. arXiv.org.

[^2]: Csaba et al., 2022. Diversified Dynamic Routing for Vision Tasks. ECCV Workshops.

[^3]: Vashkelis & Trukhina, 2026. HI-MoE: Hierarchical Instance-Conditioned Mixture-of-Experts for Object Detection. arXiv.org.