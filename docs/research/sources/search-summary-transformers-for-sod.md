A targeted search turns up a clear progression: transformers first supplied global context for SOD, then researchers began making the token budget input-adaptive. The direct SOD literature is still smaller than the general efficient-ViT literature, but it now contains several concrete routing designs.

**1. Transformer foundations for SOD**

- **VST — Visual Saliency Transformer (2021):** an early pure-transformer formulation for both RGB and RGB-D SOD. It treats image patches as tokens, fuses multiple token levels, and uses token-based saliency/boundary decoding with transformer-native upsampling. This established the basic argument for global token interactions in dense saliency prediction, but it is not primarily a conditional-computation paper. [^1]
- **VST++ (2023):** moves toward efficiency with **Select-Integrate Attention**: foreground regions remain fine-grained while background information is aggregated into a coarse token. Across RGB, RGB-D, and RGB-T settings, the authors report a 25% computational reduction without a significant performance compromise. This is closer to token aggregation/compression than hard routing. [^2]
- **Transformer Transforms SOD and COD (2021):** a broader transformer treatment spanning RGB, RGB-D, weakly supervised SOD, and camouflaged-object detection; useful for understanding why global structure and spatial supervision matter, but less directly relevant to conditional token execution. [^3]

**2. SOD-specific sparse or adaptive token methods**

- **SSTNet (2023):** uses sparse attention and selects the top-k relevant segments, with tokenized dilation to enlarge contextual interactions. Its RGB-D version adds cross-modal fusion, making it relevant when saliency depends on both appearance and depth. [^4]
- **Adaptive Spatial Tokenization Transformer for optical remote-sensing SOD (2023):** adaptively sparsifies tokens per input image while retaining global–local features, then uses a dense token-aggregation decoder to reconstruct the saliency map. The paper reports that its adaptive spatial tokenization module can halve the computational budget. This is one of the clearest examples of task-specific adaptive tokenization, although its evidence is concentrated in optical remote-sensing datasets. [^5]
- **TSVT (2024):** probably the most direct match to your phrase. Its dynamic sparse-token encoder adaptively selects and processes sparse tokens, while an asymmetric encoder–decoder and interactive RGB-depth fusion preserve multimodal information. It was evaluated on seven RGB-D SOD benchmarks; the authors report more than a twofold lower complexity than a variant without the dynamic sparse-token module. [^6]

**3. General conditional-computation papers that supply the routing toolkit**

- **DynamicViT (2021):** the canonical progressive token-pruning reference. Lightweight predictors estimate token importance at multiple layers, and attention masking makes pruning differentiable. Its headline result is pruning 66% of input tokens with 31–37% lower FLOPs, over 40% higher throughput, and at most a 0.5% accuracy drop across evaluated ViTs—but those headline results are primarily classification evidence, not SOD evidence. [^7]
- **Dynamic Spatial Sparsification (2022):** extends the idea beyond simply dropping tokens. Important locations take a slow path, while less informative locations use a cheaper fast path, preserving the spatial structure needed by dense tasks such as segmentation and detection. This is conceptually attractive for SOD because boundary and small-object regions should not disappear merely because they are locally low-confidence. [^8]
- **ToSA (2024):** predicts which tokens need the next attention layer; selected tokens receive attention, while the rest bypass that layer and are reassembled so the full spatial token set remains available for dense prediction. This is a strong architectural reference for “route some tokens, preserve all positions,” though the reported dense-prediction validation is monocular depth estimation rather than SOD. [^9]
- **BiFormer (2023):** uses **query-adaptive routed attention**, not global token dropping. It first filters candidate regions coarsely, then computes fine token-to-token attention only within the routed regions. This offers a useful alternative when every spatial position must retain a representation but not every position needs to attend everywhere. [^10]
- **Conditional computation in neural networks (2024):** a useful taxonomy covering token selection, mixture-of-experts, early exits, and submodule activation. It helps separate three design axes often conflated in this literature: which tokens are processed, which layers are executed, and which experts or submodules are activated. [^11]

**How the field currently fits together**

For SOD, the most defensible design space has four families:

1. **Hard token pruning:** remove low-importance tokens progressively, as in DynamicViT. Efficient, but risky for thin structures, boundaries, and small salient objects.
2. **Token aggregation:** merge background or similar tokens, as in VST++, reducing sequence length while retaining a coarse representation.
3. **Selective layer execution:** let important tokens enter expensive attention while other tokens bypass the layer, as in ToSA; this better preserves a complete spatial output.
4. **Routed attention:** keep tokens but restrict which key–value regions each query can access, as in BiFormer. This reduces pairwise attention without committing to irreversible token deletion.

The SOD-specific evidence favors **saliency-aware sparsification plus a spatially complete decoder**, rather than unconstrained pruning. TSVT and the remote-sensing adaptive-tokenization work show that task-specific routing can reduce compute while maintaining competitive saliency performance, but their evaluations are benchmark-specific and do not yet establish a universal routing rule. The main unresolved issue is boundary preservation: the tokens most expendable for image classification may be exactly the tokens needed to delineate a salient object. A strong next-generation SOD paper would therefore report not only aggregate F-measure or MAE, but also boundary metrics, small-object performance, token-retention maps, per-image compute, latency on real hardware, and failure cases where routing removes foreground or contour evidence.

This overview comes from a targeted search rather than a formal systematic review; the recent 2025–2026 literature may be incompletely captured.

[^1]: Liu et al., 2021. Visual Saliency Transformer. IEEE International Conference on Computer Vision.

[^2]: Liu et al., 2023. VST++: Efficient and Stronger Visual Saliency Transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence.

[^3]: Mao et al., 2021. Transformer Transforms Salient Object Detection and Camouflaged Object Detection. arXiv.org.

[^4]: Yang et al., 2023. SSTNet: Saliency sparse transformers network with tokenized dilation for salient object detection. IET Image Processing.

[^5]: Gao et al., 2023. Adaptive Spatial Tokenization Transformer for Salient Object Detection in Optical Remote Sensing Images. IEEE Transactions on Geoscience and Remote Sensing.

[^6]: Gao et al., 2024. TSVT: Token Sparsification Vision Transformer for robust RGB-D salient object detection. Pattern Recognition.

[^7]: Rao et al., 2021. DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. Neural Information Processing Systems.

[^8]: Rao et al., 2022. Dynamic Spatial Sparsification for Efficient Vision Transformers and Convolutional Neural Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence.

[^9]: Singh et al., 2024. ToSA: Token Selective Attention for Efficient Vision Transformers. arXiv.org.

[^10]: Zhu et al., 2023. BiFormer: Vision Transformer with Bi-Level Routing Attention. Computer Vision and Pattern Recognition.

[^11]: Scardapane et al., 2024. Conditional computation in neural networks: Principles and research trends. Intelligenza Artificiale.