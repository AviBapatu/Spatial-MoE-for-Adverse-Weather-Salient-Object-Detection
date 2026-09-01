A targeted search found a small but coherent cluster of papers linking salient object detection to dynamic selection; the strongest direct matches are mostly **RGB-D/multimodal SOD**, while “spatially adaptive experts” is often implemented under names such as adaptive fusion, feature selection, or scale-aware routing. The most directly relevant papers are:

- **MMSOD (2025)** — Uses a gating mechanism to adjust each expert’s contribution from modality-specific characteristics, with a multiscale fusion adapter. This is the clearest general mixture-of-experts formulation for multimodal SOD in the set I found. [^1]
- **HEHP (2025)** — Designed for underwater SOD. It separates RGB and depth representations, uses frequency-specialized experts for high- and low-frequency structure, and applies four-way fusion experts that dynamically select according to scale and orientation. This is the closest match to **spatially adaptive experts**. [^2]
- **CMFNet (2026)** — Combines MobileViT with a lightweight MoE module. Its router assigns expert processing units to multimodal features at different scales, with a balancing constraint intended to avoid poor expert utilization. The emphasis is efficiency and deployment rather than fine-grained spatial routing. [^3]
- **CMoE for modality-missing SOD (2026)** — Cascades two MoE stages: a missing-aware router reconstructs representations when RGB or depth is unavailable, then a multimodal MoE combines modality-specific experts. This is particularly relevant if the intended system must handle sensor dropout or incomplete inputs. [^4]
- **Dynamic Neural Network for RGB-D SOD (2026)** — Routes features using an estimate of task-relevant complementarity between RGB and depth, based on task-relevance scoring and mutual-information estimation. It reports roughly 1% average gains across several metrics and up to 1.8% in severe modality-degradation cases, although these are abstract-level claims and need full-text verification before treating them as definitive. [^5]
- **Feature Quality-Based Dynamic Feature Selection (2016)** — An early, non-MoE precursor: it measures feature quality and dynamically selects useful feature detectors per image. The method is conceptually important because it establishes image-conditional selection rather than fixed feature fusion; gains were strongest in cluttered images and scenes with multiple salient objects. [^6]

Two adjacent papers are useful for the routing design itself. **Learning Dynamic Routing for Semantic Segmentation (2020)** introduces differentiable soft conditional gates that select scale-transform paths according to each image’s scale distribution, with optional computation-budget constraints. [^7] **Diversified Dynamic Routing (2022)** addresses a central MoE failure mode: routers can collapse onto the same experts. It explicitly trains expert assignment and data partitioning to promote diversity, with evidence from semantic and instance segmentation rather than SOD. [^8]

Taken together, the field appears to be moving through three increasingly specific forms of adaptivity: **image-level feature selection** → **modality- and scale-aware routing** → **heterogeneous experts conditioned on spatial structure, uncertainty, frequency, or missing modalities**. For a new SOD contribution, the clearest whitespace is not simply “use MoE,” but a router that makes **pixel/region-level expert decisions**, explicitly separates scale from orientation or frequency specialization, and reports expert-load/diversity diagnostics alongside saliency metrics. The current search was weighted toward RGB-D SOD and recent papers; it may not capture every RGB-only paper using “dynamic convolution,” “conditional computation,” or “adaptive receptive fields” rather than MoE terminology.

[^1]: Cen et al., 2025. MMSOD: Mixture-of-experts with Adapter for Multimodal Salient Object Detection. IEEE International Joint Conference on Neural Network.

[^2]: Zha et al., 2025. Heterogeneous Experts and Hierarchical Perception for Underwater Salient Object Detection. IEEE Transactions on Image Processing.

[^3]: Wu et al., 2026. Cross-Modal Fusion With Mixture-of-Experts for Efficient RGB-D Salient Object Detection. IEEE transactions on multimedia.

[^4]: Wang et al., 2026. Taming Cascaded Mixture-of-Experts for Modality-missing Multi-modal Salient Object Detection. AAAI Conference on Artificial Intelligence.

[^5]: Li et al., 2026. Learning Modality Complementarity for RGB-D Salient Object Detection via Dynamic Neural Network. Electronics.

[^6]: Naqvi et al., 2016. Feature Quality-Based Dynamic Feature Selection for Improving Salient Object Detection. IEEE Transactions on Image Processing.

[^7]: Li et al., 2020. Learning Dynamic Routing for Semantic Segmentation. Computer Vision and Pattern Recognition.

[^8]: Csaba et al., 2022. Diversified Dynamic Routing for Vision Tasks. ECCV Workshops.