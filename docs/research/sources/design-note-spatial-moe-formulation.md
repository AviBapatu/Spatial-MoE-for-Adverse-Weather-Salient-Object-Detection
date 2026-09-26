If you are sketching a model, the strongest formulation is a **spatial MoE for salient object detection**: tokenize the image into patches, predict a saliency/complexity score for each token, and route each token—rather than the whole image—to a small set of spatial experts. This is closely related to TSVT, which dynamically selects sparse RGB-D tokens and reports more than a two-fold complexity reduction versus its dense-token variant [^1], while CMFNet applies dynamic, balanced MoE routing to multi-scale RGB-D features [^2]. VST provides the decoder precedent: multi-level token fusion, token upsampling, and a separate boundary-aware task-token pathway are useful for recovering dense masks [^3].

A practical architecture would be:

1. **Backbone:** hierarchical ViT producing token maps at 1/4, 1/8, 1/16, and 1/32 resolution.
2. **Spatial router:** for token $$x_i$$, compute $$p_i = \operatorname{softmax}(W_r[x_i,\,s_i,\,b_i])$$, where $$s_i$$ is a saliency prior and $$b_i$$ is a boundary/uncertainty prior. Use top-1 or top-2 routing with capacity limits.
3. **Experts:** use complementary experts rather than identical FFNs: global-context, local-detail, boundary, and texture/low-contrast experts. A shared expert should process every token; routed experts provide the adaptive residual.
4. **Spatial consistency:** route neighboring tokens jointly or regularize neighboring routing distributions. Independent token choices can fluctuate and become unstable; attention-/similarity-aware routing was proposed specifically to reduce this problem [^4].
5. **Decoder:** scatter expert outputs back to their spatial locations, fuse multi-scale features, and use a dense boundary refinement head. Never discard uncertain boundary tokens solely because their saliency score is low.
6. **Loss:** $$L=L_{mask}+\lambda_bL_{boundary}+\lambda_uL_{uncertainty}+\lambda_{bal}L_{balance}+\lambda_sL_{spatial\text{-}smooth}$$. Use BCE or focal loss plus IoU/Dice for the mask, boundary loss for contours, and a mild load-balancing term. Avoid making the router optimize saliency alone; otherwise it may send obvious foreground tokens to expensive experts while neglecting hard background and contour regions.

The key novelty is not merely “MoE applied to SOD.” It is **saliency- and uncertainty-conditioned spatial routing with a boundary-preserving sparse decoder**. The most informative ablations would compare dense FFN, image-level MoE, independent pixel/token routing, neighborhood-aware routing, and routing with versus without the uncertainty branch. The literature search found direct precedents for sparse tokens and RGB-D MoE, but not a clearly established canonical method combining all of these elements specifically for RGB salient-object detection; that combination is therefore a plausible research contribution rather than an already standardized design.

[^1]: Gao et al., 2024. TSVT: Token Sparsification Vision Transformer for robust RGB-D salient object detection. Pattern Recognition.

[^2]: Wu et al., 2026. Cross-Modal Fusion With Mixture-of-Experts for Efficient RGB-D Salient Object Detection. IEEE transactions on multimedia.

[^3]: Liu et al., 2021. Visual Saliency Transformer. IEEE International Conference on Computer Vision.

[^4]: Nguyen et al., 2025. Improving Routing in Sparse Mixture of Experts with Graph of Tokens. arXiv.org.