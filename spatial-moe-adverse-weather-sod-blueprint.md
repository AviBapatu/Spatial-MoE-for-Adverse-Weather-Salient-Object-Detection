# Spatially Dynamic Mixture-of-Experts for Adverse-Weather Salient Object Detection
### Research Blueprint — Datasets, Architecture, Losses, Interpretability, Ablations

---

## 1. Benchmark Datasets & SOTA Literature (2021–2026)

### 1.1 Datasets

**Adverse-weather SOD (direct target task)**
- **WXSOD** (Chen et al., 2025) — the closest existing benchmark to your exact problem. 14,945 RGB images with pixel-wise saliency ground truth plus per-image weather labels; ships with both a *synthetic* test split (clean images + injected weather noise) and a *real-world* test split (genuine adverse-weather captures), which is exactly the generalization gap you want to claim. Their baseline, WFANet, is a two-branch (restore + segment) network — a natural point of comparison for your spatial-MoE design, since they route at the *image* level (dual branch) while you route at the *token* level.
- **RGBT-SOD sets** (VT5000, VT1000, VT821) — thermal-paired SOD datasets. Not weather-labeled per se, but thermal channels are robust to fog/low-light, so these are useful for a multimodal ablation or for arguing why you *don't* need a second sensor if spatial MoE recovers the same robustness from RGB alone.

**Adverse-weather scene understanding (transferable protocols, not SOD-labeled, but standard for building synthetic degradation pipelines and for domain-generalization eval)**
- **Foggy Cityscapes / Foggy Driving** — synthetic fog via depth-aware scattering model; standard for controlled-density fog studies.
- **RainCityscapes / Rain-KITTI / RainDrop** — rain streak and raindrop-on-lens degradations.
- **Snow100K / CSD (Comprehensive Snow Dataset)** — snow occlusion and veiling.
- **ExDark** — low-light/nighttime images with detection-style annotations, useful for low-light SOD transfer.
- **ACDC (Adverse Conditions Dataset with Correspondences)** — fog/rain/snow/night with pixel-level segmentation labels and clean-condition correspondences; excellent for building a controlled "same scene, N weather conditions" ablation the way BDD100K or Cityscapes-C cannot.
- **WeatherBench / WeatherStream** — used in the all-in-one restoration literature (see 1.2) for multi-degradation training; you can borrow their degradation-synthesis pipelines to weather-augment any clean SOD dataset (DUTS, HKU-IS, ECSSD, DUT-OMRON) rather than waiting for weather-labeled SOD masks.
- **XWOD** (2026) — real-world extreme-weather *detection* benchmark (rain/snow/fog/haze-sand-dust/flood/tornado/wildfire, 10K images). Not SOD, but a strong source of real degradation diversity beyond fog/rain/snow if you want to argue broader "heterogeneous adverse weather" coverage.

**Standard clean SOD benchmarks (needed for your "does robustness cost clean-condition accuracy" ablation)**
- DUTS (train/test), ECSSD, HKU-IS, DUT-OMRON, PASCAL-S, SOD.

### 1.2 Key Papers (organized by the two literatures you're fusing)

**A. Adverse-weather-aware dense prediction / restoration with expert-style routing**
1. **TransWeather** (Valanarasu et al., CVPR 2022) — transformer with learnable "weather-type" queries; single shared decoder handles multiple degradations without explicit routing. Good baseline for "implicit specialization without MoE."
2. **Learning Weather-General and Weather-Specific Features for Image Restoration under Multiple Adverse Weather Conditions** (Zhu et al., CVPR 2023) — explicitly splits features into a shared branch and a degradation-specific branch; conceptually the two-branch precursor to your token-level split, but done at the *channel* level, not the *spatial-token* level.
3. **WM-MoE: Weather-aware Multi-scale Mixture-of-Experts for Blind Adverse Weather Removal** (2023) — the most directly relevant prior work. Combines multi-scale features with MoE experts and weather-type awareness. Your novelty claim must explicitly differentiate: WM-MoE routes primarily on (multi-scale) *feature maps with weather-conditioned gating*, not on a fully spatial, per-token, weather-agnostic router that lets specialization emerge without any weather label at all.
4. **MoWE: Mixture of Weather Experts for Multiple Adverse Weather Removal** (2023) — closest "MoE" naming precedent; compare routing granularity (image/patch vs. your token level) in your related-work table.
5. **Efficient Deweathering Mixture-of-Experts with Uncertainty-aware Feature-wise Linear Modulation** (AAAI 2024) — introduces uncertainty-gated FiLM modulation per expert; useful reference for how to make your router's confidence signal do double duty as an *uncertainty estimate* (relevant to your interpretability section).
6. **Complexity Experts are Task-Discriminative Learners for Any Image Restoration** (Zamfir et al., CVPR 2025) — most recent SOTA; experts are specialized by *computational complexity* rather than task/weather label, letting difficulty-adaptive routing emerge implicitly — directly supports your "implicit expert specialization" design philosophy and is a strong related-work anchor for your discussion section.
7. **GridFormer** (2023) — residual dense grid transformer for multi-weather restoration; alternative multi-scale fusion baseline for your Section 2.2 comparison.

**B. Token-level / spatial Mixture-of-Experts in vision transformers**
8. **Scaling Vision with Sparse Mixture of Experts (V-MoE)** (Riquelme et al., NeurIPS 2021) — foundational token-level top-k routing for ViT; your router formulation in Section 2 is a direct architectural descendant.
9. **From Sparse to Soft Mixtures of Experts (Soft MoE)** (Puigcerver et al., ICLR 2024) — replaces hard top-k assignment with a fully differentiable weighted combination, removing load-balancing losses and token-dropping issues entirely; a strong alternative router design to benchmark against your top-k router in ablations.
10. **M³ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning** (Liang et al., NeurIPS 2022) and **Mod-Squad: Designing Mixtures of Experts as Modular Multi-task Learners** (Chen et al., CVPR 2023) — task-conditioned token routing for dense multi-task prediction; useful precedent for showing that spatial tokens naturally cluster by *task-relevant local structure* (their "task," your "weather artifact type").
11. **Routers in Vision Mixture of Experts: An Empirical Study** (Liu, Blondel, Riquelme, Puigcerver, TMLR 2024) — directly useful methodological reference for your router-design ablations (Section 5) and for citing known MoE failure modes (router collapse, expert underutilization) that motivate your auxiliary losses (Section 3).

### 1.3 Positioning your contribution
Your related-work table should make three axes explicit, since every reviewer will ask this immediately:
| Axis | WM-MoE / MoWE | Zhu et al. CVPR'23 | Your method |
|---|---|---|---|
| Routing granularity | Feature-map / coarse-patch | Channel (branch-level) | **Per-token, per-scale** |
| Task | Restoration (pixel regression) | Restoration | **SOD (pixel classification with topology constraints)** |
| Conditioning signal | Weather-type label/embedding | Implicit (2-branch) | **Fully implicit — no weather label at train or test time** |
| Multi-scale routing | Single fused scale | Single scale | **Independent routers at 1/4, 1/8, 1/16, fused after** |

This is your paper's central novelty claim: nobody has combined (a) fully label-free spatial/token routing, (b) at multiple pyramid scales independently, (c) for the *segmentation-topology-sensitive* SOD task rather than pixel-regression restoration.

---

## 2. Technical Architecture Blueprint

### 2.1 Shared Backbone

Input image $I \in \mathbb{R}^{H_0 \times W_0 \times 3}$. A hierarchical backbone (Swin-B or PVTv2-B4 recommended over plain ResNet, since window/pyramid transformers already produce natural token grids at each stage) produces multi-scale feature maps:

$$F_s = \text{Backbone}_s(I) \in \mathbb{R}^{H_s \times W_s \times C_s}, \quad s \in \{4, 8, 16\}$$

where $H_s = H_0/s$, $W_s = W_0/s$. Each $F_s$ is flattened into a token sequence $X_s \in \mathbb{R}^{N_s \times C_s}$, $N_s = H_s W_s$, and projected to a common working dimension $d$ via a 1×1 conv / linear layer: $\tilde{X}_s = X_s W_s^{proj}$, $\tilde X_s \in \mathbb R^{N_s\times d}$.

### 2.2 Spatial / Token-Level Router (per scale $s$)

For each token $x_i \in \tilde X_s$ (index $i = 1 \dots N_s$), the router is a lightweight 2-layer MLP with a local-context-aware twist: rather than routing from the token's own embedding alone (which can be a weak, aliased signal at low resolution), the router also consumes a small local neighborhood via a depthwise 3×3 conv, so the routing decision uses local texture statistics (haze gradient smoothness, streak orientation energy, specular contrast) not just the pointwise feature:

$$g_i = W_2 \, \sigma\!\big(W_1 \, [\,x_i \,\|\, \text{DWConv}_{3\times3}(X_s)_i\,] + b_1\big) + b_2 \in \mathbb{R}^{N}$$

where $N$ is the number of experts, $\sigma$ is GELU, and $\|$ is concatenation. Router logits are converted to a sparse gate via noisy top-$k$ (Shazeer-style) to preserve load-balancing gradients:

$$g_i' = g_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \text{softplus}(W_{noise} x_i)^2)$$

$$\text{Top-}k(g_i') = \{ j : g_i'^{(j)} \text{ among the } k \text{ largest of } g_i' \}$$

$$p_i^{(j)} = \begin{cases} \dfrac{\exp(g_i'^{(j)})}{\sum_{j' \in \text{Top-}k(g_i')} \exp(g_i'^{(j')})} & j \in \text{Top-}k(g_i') \\ 0 & \text{otherwise} \end{cases}$$

Use $k=2$ (standard sweet spot balancing specialization vs. redundancy/robustness — if one expert is "wrong" for an ambiguous token, the second acts as a soft fallback, which matters a lot when weather artifacts are locally ambiguous, e.g. fog vs. low-contrast shadow).

### 2.3 Implicit Expert Pool

$N$ lightweight experts $E_1 \dots E_N$ per scale (do **not** share expert weights across scales — a fog-smoothing expert at 1/16 resolution and one at 1/4 resolution need different receptive-field priors). Each expert is a small residual block, e.g. a depthwise-separable conv + pointwise MLP (inverted bottleneck, MobileNetV2-style) to keep FLOPs low despite $N$-way replication:

$$E_j(x) = x + W_{out}^{(j)} \, \text{GELU}\big(\text{DWConv}^{(j)}(W_{in}^{(j)} x)\big)$$

The token's expert output is the gate-weighted sum over its active experts:

$$y_i = \sum_{j=1}^{N} p_i^{(j)} \, E_j(x_i)$$

Recommended $N=6$–$8$ experts per scale for an SOD-sized model (larger than needed if you only have ~3–4 canonical weather artifact types, but overprovisioning lets specialization *emerge* rather than be hand-assigned — critical for your "implicit" claim, and gives headroom for experts to split further, e.g. separate rain-streak-orientation experts for near-vertical vs. wind-slanted streaks).

### 2.4 Multi-Scale Pyramid Integration

Each scale routes **independently** (separate router weights $\theta_s^{router}$, separate expert pools $\{E_j^{(s)}\}$) — this is the core architectural choice to defend in ablations, because a shared router across scales would force one routing decision to serve incompatible receptive fields (a token that looks "hazy" at 1/16 may correspond to sharp rain streaks once you zoom to 1/4). After per-scale MoE processing, you get $Y_4, Y_8, Y_{16}$. Top-down fusion via feature-pyramid upsampling + cross-attention alignment:

$$\hat Y_{16} = Y_{16}$$
$$\hat Y_8 = Y_8 + \text{CrossAttn}(Q=Y_8,\, K=V=\text{Upsample}_{2\times}(\hat Y_{16}))$$
$$\hat Y_4 = Y_4 + \text{CrossAttn}(Q=Y_4,\, K=V=\text{Upsample}_{2\times}(\hat Y_8))$$

Cross-attention (rather than naive add/concat) matters here because expert outputs at different scales are *not* spatially or semantically aligned in the same way that plain FPN features are — coarse-scale experts encode global weather context (e.g., "whole scene is foggy") that needs to selectively modulate, not just add to, fine-scale boundary-expert output.

### 2.5 Weighted Fusion Module + SOD Decoder

Final decoder is a standard progressive-upsampling SOD head (BASNet/U²-Net-style) operating on $[\hat Y_4, \hat Y_8, \hat Y_{16}]$, with one addition specific to your architecture: concatenate the **per-token router entropy** (Section 4) as an extra channel at each scale before the final conv — this gives the decoder an explicit "how confident/ambiguous was routing here" signal, which empirically helps at object boundaries where degradation is most severe:

$$S = \text{DecoderHead}\big([\hat Y_4 \,\|\, H_4,\; \hat Y_8 \,\|\, H_8,\; \hat Y_{16} \,\|\, H_{16}]\big) \in [0,1]^{H_0 \times W_0}$$

where $H_s$ is the per-token routing-entropy map at scale $s$ (defined in Section 4.1), upsampled to match.

---

## 3. Loss Function Formulation

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda_{iou}\mathcal{L}_{\text{IoU}} + \lambda_{edge}\mathcal{L}_{\text{edge}} + \lambda_{lb}\mathcal{L}_{\text{load-balance}} + \lambda_{imp}\mathcal{L}_{\text{importance}}$$

### 3.1 Saliency supervision

**BCE** (per-pixel, standard):
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{HW}\sum_{u,v}\big[G_{uv}\log S_{uv} + (1-G_{uv})\log(1-S_{uv})\big]$$

**IoU loss** (region-level, complements BCE's pixel-independence weakness):
$$\mathcal{L}_{\text{IoU}} = 1 - \frac{\sum_{u,v} S_{uv} G_{uv}}{\sum_{u,v}\big(S_{uv} + G_{uv} - S_{uv}G_{uv}\big)}$$

**Edge/boundary loss** — critical for this paper specifically, since weather degradation (fog blur, rain-streak occlusion) attacks boundaries first; use a Sobel/Laplacian-derived edge map $G_{edge}$ from the ground truth and weighted BCE against the predicted saliency's own gradient map $S_{edge} = |\nabla S|$:
$$\mathcal{L}_{\text{edge}} = -\frac{1}{HW}\sum_{u,v}\big[G^{edge}_{uv}\log S^{edge}_{uv} + (1-G^{edge}_{uv})\log(1-S^{edge}_{uv})\big]$$
(Alternatively use a dedicated boundary-IoU or dice term — either is standard in BASNet/PraNet-family losses; pick one and justify via ablation rather than stacking both.)

### 3.2 MoE auxiliary losses (per scale $s$, averaged over all $s$)

**Load-balancing loss** (Shazeer/Switch-Transformer form) — prevents a small subset of experts from absorbing all tokens (mode collapse), which is the single most common MoE failure mode and would directly undermine your "implicit specialization" claim if a couple of experts end up dead:

$$\mathcal{L}_{\text{load-balance}} = N \sum_{j=1}^{N} f_j \cdot P_j$$

where $f_j = \frac{1}{N_s}\sum_i \mathbb{1}[j \in \text{Top-}k(g_i')]$ is the *fraction of tokens routed to expert $j$* (hard count, non-differentiable), and $P_j = \frac{1}{N_s}\sum_i p_i^{(j)}$ is the *mean routing probability mass on expert $j$* (soft, differentiable) — this product form lets gradients flow through $P_j$ while still penalizing the true hard-count imbalance $f_j$ measures.

**Importance loss** (coefficient-of-variation form, encourages the *distribution* of gate weights across experts, not just token counts, to stay balanced — catches the subtler failure where all tokens get routed roughly evenly by count but one expert always receives near-1.0 gate weight while others get scraps):
$$\text{Importance}_j = \sum_i p_i^{(j)}, \qquad \mathcal{L}_{\text{importance}} = \left(\frac{\text{std}(\text{Importance})}{\text{mean}(\text{Importance})}\right)^2$$

Typical weights: $\lambda_{iou}=1$, $\lambda_{edge}=1$, $\lambda_{lb}=0.01$, $\lambda_{imp}=0.01$ (MoE aux losses should stay small relative to task loss — over-weighting them forces artificially uniform routing and directly fights the "let experts specialize" goal; tune via the load-balance-weight ablation in Section 5).

---

## 4. Expert Interpretability & Visualization Strategy

### 4.1 Routing entropy maps
Per-token routing entropy $H_i = -\sum_j p_i^{(j)}\log p_i^{(j)}$, reshaped back to $H_s \times W_s$ and upsampled — low entropy = confident single-expert assignment (likely a "clean" or unambiguous region), high entropy = ambiguous/contested region (often exactly the pixels near saliency boundaries or overlapping degradation types, e.g. foggy-and-backlit). Plotting entropy maps overlaid on input images across a weather-diversity grid (clean / fog / rain / snow / low-light / glare) is your headline qualitative figure.

### 4.2 Per-expert token-assignment heatmaps
For a fixed expert $j$, render $p^{(j)}$ (before top-k masking, or the hard assignment mask) as a heatmap over the image for a batch of validation images spanning weather types. Aggregate over the test set: for each expert, compute the empirical distribution of assigned tokens' *local image statistics* (local contrast, gradient magnitude, colorfulness/saturation drop, dark-channel prior value as a haze proxy) to produce a quantitative "expert profile" — e.g. "Expert 3: mean dark-channel value 0.71 (high, i.e., hazy), mean gradient magnitude low → fog-smoothing specialist" — turning a qualitative heatmap claim into a defensible statistic.

### 4.3 t-SNE / UMAP cluster analysis
Collect token embeddings $x_i$ (pre-router) across a stratified sample of validation tokens labeled by (a) which expert they were routed to and (b) the coarse weather category of their source image (even though the model never sees this label). Project via t-SNE/UMAP and color by expert assignment vs. by weather category — the key figure is showing that expert clusters *correlate* with weather-artifact clusters without being *identical* to weather category clusters (this is the evidence for "spatial/local specialization beyond global weather label," which is your core architectural claim over WM-MoE-style weather-conditioned routing).

### 4.4 Ablated single-expert forward passes
Force-route 100% of tokens to a single expert $j$ (bypass the router) and run the full pipeline; visualize the resulting saliency map degradation per weather type. An expert specialized for high-frequency boundary recovery should produce reasonable saliency on clean/low-fog images but collapse specifically on snow/rain-heavy inputs, and vice versa for a fog-specialist expert — this "expert knockout" study is the causal complement to the correlational t-SNE evidence.

### 4.5 Routing consistency across weather severity
For synthetic weather augmentation (where you control severity continuously, e.g. fog optical depth $\beta \in [0, 1.5]$), plot expert-assignment fraction $f_j$ as a function of severity for a fixed scene — a good specialization story shows smooth, monotonic hand-off from "clean-region experts" to "degradation-specific experts" as severity increases, rather than abrupt or noisy switching (noisy switching indicates the router is unstable / needs more load-balance regularization).

---

## 5. Ablation Study Plan

1. **Spatial/token routing vs. global (image-level) routing** — replace the per-token router with a single global router producing one expert-mixture weight for the whole image (closest to how WM-MoE/MoWE condition on weather type). This is your single most important ablation: it directly tests whether *spatial* granularity is necessary or whether image-level routing captures most of the benefit. Report both overall SOD metrics (mean absolute error / MAE, $F_\beta$, $S_\alpha$, $E_\phi$) and a **boundary-region-only** metric split, since you expect the spatial router's advantage to concentrate at partially-degraded boundaries rather than uniformly-degraded interiors.

2. **Multi-scale independent routing vs. single-scale routing** — collapse to routing only at one scale (try 1/8 alone, both as an ablation and to identify which single scale is the strongest fallback) vs. sharing one router's decision across all three scales (upsample/downsample the same gate map) vs. your full independent-per-scale design. Tests whether the "route independently per receptive field" claim in Section 2.4 actually earns its complexity.

3. **MoE vs. standard dense (non-MoE) backbone of matched FLOPs/parameters** — the essential "does MoE even help" baseline. Build a dense model with equivalent compute (either a wider shared block or a small ensemble with fixed equal-weight averaging instead of learned routing) and compare. If your MoE model doesn't beat this at matched compute, the routing mechanism itself is not earning its complexity, which reviewers will check first.

4. **Top-k sparsity and expert count sweep** — vary $k \in \{1,2,3\}$ and $N \in \{4,6,8,12\}$ per scale; report both accuracy and the load-balance/importance loss curves during training, to characterize the specialization-vs-redundancy trade-off and justify your chosen $(N,k)$ rather than asserting it.

5. **Auxiliary-loss ablation** — remove load-balancing loss only, remove importance loss only, remove both, and (optionally) compare hard top-k routing against Soft MoE's fully differentiable weighting (item 9 in Section 1.2) as an alternative that sidesteps load-balancing losses entirely. This substantiates the specific claims in Section 3.2 about what each auxiliary loss prevents, and pre-empts the "why not just use Soft MoE" reviewer question.

6. *(Stretch, if compute allows)* **Domain-generalization cross-check** — train on synthetic weather augmentation only (fog/rain/snow synthesized onto DUTS), evaluate zero-shot on WXSOD's *real* test split. This is the strongest possible evidence for the paper's actual claim (generalization to *heterogeneous, unseen-at-training* real weather) versus merely fitting to a fixed synthetic distribution, and it's the ablation most likely to separate this paper from the restoration-MoE literature it's built on.

---

## Summary Table: Where Your Novelty Actually Lives

| Component | Precedent exists? | Your delta |
|---|---|---|
| Token-level MoE routing | Yes (V-MoE, Soft MoE) | Applied to dense SOD, not classification |
| Weather-aware MoE | Yes (WM-MoE, MoWE) | Routing is **label-free and spatial**, not weather-conditioned/global |
| Multi-scale MoE | Partial (M³ViT is multi-task, not multi-scale-independent) | **Independently-routed** pyramid scales with cross-attention fusion |
| SOD boundary loss + MoE | No direct precedent found | Router-entropy-as-decoder-input is a genuinely novel wiring choice worth protecting as a contribution |

This table itself is a strong candidate for your paper's Figure 1 / Table 1.
