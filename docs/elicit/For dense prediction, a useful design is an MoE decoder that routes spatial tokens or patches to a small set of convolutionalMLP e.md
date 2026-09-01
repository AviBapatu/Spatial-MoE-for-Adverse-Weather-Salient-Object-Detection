For **dense prediction**, a useful design is an MoE decoder that routes spatial tokens or patches to a small set of convolutional/MLP experts, then uses **routing entropy** to decide how much expert capacity to spend on each region.

Let the router produce logits $$z_{p,e}$$ for pixel/patch $$p$$ and expert $$e$$:

$$

a_{p,e}=\operatorname{softmax}_e(z_{p,e}/\tau),\qquad
H_p=-\sum_e a_{p,e}\log a_{p,e}.
$$

Use normalized entropy $$\hat H_p=H_p/\log E$$, where $$E$$ is the number of experts.

**Recommended decoder pattern**

1. **Shared decoder stem:** FPN/U-Net-style feature fusion from the encoder.
2. **Patch- or region-level router:** predict one routing distribution per local region rather than independently per pixel. Recent segmentation work specifically studies patch-wise sparse routing and reports improvements up to 3.9 mIoU, but also finds that performance is highly sensitive to architectural choices. [^1]
3. **Top-k experts:** use top-1 or top-2 experts for low-entropy regions; preserve a dense residual/shared path so uncertain or novel regions do not fail because the wrong expert was selected.
4. **Entropy-conditioned compute:** for high-entropy regions, either increase $$k$$, mix more experts softly, or fall back to the shared path. For low-entropy regions, use sparse top-1 routing.
5. **Entropy-aware loss:** combine task loss with load balancing and router stability:

$$
\mathcal L=\mathcal L_{task}+\lambda_{lb}\mathcal L_{load}+\lambda_{cons}\mathcal L_{cons}+\lambda_H\mathcal L_H.
$$

Avoid simply minimizing entropy: that can produce confident but incorrect routing and expert collapse. A better target is **low entropy when experts agree and high entropy near class boundaries, occlusion, domain shift, or genuinely ambiguous structure**.

**What to measure**

- pixel/region accuracy versus $$\hat H_p$$;
- calibration error and risk–coverage curves;
- boundary quality, small-object performance, and OOD detection;
- expert utilization, importance/load coefficient of variation, and routing stability under augmentations;
- FLOPs and latency as a function of the entropy threshold.

The closest direct evidence supports using gate entropy as a routing-uncertainty measure: one segmentation study evaluated predictive entropy, mutual information, expert variance, and gate entropy, finding better routing-uncertainty calibration with a simple gate than with more complex classwise gates. [^2] The practical interpretation is that entropy should be used as a **compute-allocation and abstention signal**, not treated as a proxy for semantic difficulty without calibration.

For multi-task dense prediction, a shared path plus task-specific low-rank experts is another strong pattern: it allows task-specific specialization while retaining explicit parameter sharing and controlling decoder cost. [^3]

[^1]: Pavlitska et al., 2026. Design and Behavior of Sparse Mixture-of-Experts Layers in CNN-based Semantic Segmentation. arXiv.org.

[^2]: Pavlitska et al., 2025. Extracting Uncertainty Estimates from Mixtures of Experts for Semantic Segmentation. 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW).

[^3]: Yang et al., 2024. Multi-Task Dense Prediction via Mixture of Low-Rank Experts. Computer Vision and Pattern Recognition.