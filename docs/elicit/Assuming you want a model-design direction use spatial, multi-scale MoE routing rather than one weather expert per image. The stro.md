Assuming you want a model-design direction: use **spatial, multi-scale MoE routing** rather than one weather expert per image. The strongest template from the literature is a shared encoder plus a router that predicts per-token expert weights from **decoupled weather/degradation and content features**; weather experts handle haze, rain, snow, and mixed degradation, while multi-scale branches provide different receptive fields. WM-MoE explicitly assigns experts per image token and uses multi-scale experts for coupled weather conditions [^1]. Spatial-MoE provides the general rationale for fine-grained routing in spatial domains, along with a self-supervised routing loss and expert-error damping [^2]. For dense prediction, keep a semantic pathway alongside restoration: prior work cautions that restoration quality does not automatically translate into segmentation quality and instead uses semantic guidance to refine both tasks [^3].

**Recommended formulation**

1. **Shared backbone:** hierarchical CNN/ViT encoder with FPN-style features at 1/4, 1/8, and 1/16 resolution.
2. **Router input:** concatenate local feature, low-frequency illumination/contrast statistics, high-frequency artifact statistics, and a detached semantic feature. Predict top-$$k$$ experts per spatial token, with a soft mixture during training and sparse top-$$k$$ execution at inference.
3. **Experts:** haze/contrast-restoration, rain/streak removal, snow/occlusion handling, low-light/illumination correction, and a generic mixed-weather expert. Give each expert both local and global branches; do not force weather labels when degradations are spatially mixed.
4. **Aggregation:** normalize router weights locally, add a residual shared expert, and use uncertainty-aware fusion. A recent autonomous-driving MoE design separates routing from output aggregation and reweights experts using feature-statistical distance, which is a useful safeguard against brittle top-1 decisions [^4].
5. **Prediction head:** semantic segmentation, depth, detection, or other dense head consumes the routed features directly. Optionally add a lightweight restoration auxiliary head, but select checkpoints by downstream mIoU/depth error—not PSNR alone.

**Losses:** task loss + weather-consistency loss + router load-balancing loss + spatial smoothness loss on neighboring routing distributions + entropy penalty that encourages confident routing only where degradation evidence is strong. For unlabeled target-weather data, uncertainty-weighted multi-scale pseudo-labels are a reasonable adaptation component; FogAdapt reports this strategy for foggy segmentation [^5].

**Key ablations:** image-level versus token-level routing; top-1 versus top-2/soft routing; weather-only versus weather-plus-content router; shared versus weather-specific experts; with/without residual shared expert; restoration-first versus joint semantic training; and homogeneous versus spatially mixed weather. Evaluate per-weather and mixed-weather performance, boundary IoU, calibration/entropy by degradation severity, expert utilization, routing stability under crops, and latency/memory. The main risk is router shortcutting: it may route by scene type, camera, or dataset identity instead of degradation. Counter this with weather-balanced sampling, content-preserving degradation augmentation, router invariance tests, and held-out combinations of weather type and spatial layout.

[^1]: Luo et al., 2023. MoWE: Mixture of Weather Experts for Multiple Adverse Weather Removal. arXiv.org.

[^2]: Dryden & Hoefler, 2022. Spatial Mixture-of-Experts. Neural Information Processing Systems.

[^3]: Xia & Yang, 2019. Segment for Restoration, Restore for Segmentation. arXiv: Computer Vision and Pattern Recognition.

[^4]: Kou et al., 2025. Statistic-Augmented, Decoupled MoE Routing and Aggregating in Autonomous Driving. arXiv.org.

[^5]: Iqbal et al., 2022. FogAdapt: Self-supervised domain adaptation for semantic segmentation of foggy images. Neurocomputing.