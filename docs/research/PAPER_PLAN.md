# Paper Writing Plan

## Paper Title
Spatially Dynamic Mixture-of-Experts for Adverse-Weather Salient Object Detection

## Novelty Claims (from blueprint)

Three-axis positioning against prior work:

| Axis | WM-MoE / MoWE | Zhu et al. CVPR'23 | This Method |
|------|---------------|---------------------|-------------|
| Routing granularity | Feature-map / coarse-patch | Channel (branch-level) | **Per-token, per-scale** |
| Task | Restoration (pixel regression) | Restoration | **SOD (pixel classification with topology constraints)** |
| Conditioning signal | Weather-type label/embedding | Implicit (2-branch) | **Fully implicit — no weather label at train or test time** |
| Multi-scale routing | Single fused scale | Single scale | **Independent routers at 1/4, 1/8, 1/16, fused after** |

**Central claim:** Nobody has combined (a) fully label-free spatial/token routing, (b) at multiple pyramid scales independently, (c) for the segmentation-topology-sensitive SOD task.

## Suggested Paper Structure

### 1. Introduction
- SOD degradation under adverse weather
- Limitations of weather-specific dual-branch networks
- Contribution: token-level spatial MoE without weather labels

### 2. Related Work

#### 2.1 Adverse-Weather SOD
- TransWeather (CVPR 2022): transformer with weather-type queries
- Zhu et al. (CVPR 2023): shared + degradation-specific branches
- WM-MoE (2023): weather-aware MoE for restoration
- MoWE (2023): Mixture of Weather Experts
- Complexity Experts (CVPR 2025): complexity-based routing

#### 2.2 Token-level MoE in Vision
- V-MoE (NeurIPS 2021): foundational token-level top-k routing
- Soft MoE (ICLR 2024): differentiable weighted combination
- M³ViT (NeurIPS 2022): task-conditioned routing
- Routers in Vision MoE (TMLR 2024): router design study

### 3. Method

#### 3.1 Shared Backbone
- PVTv2-B4 → multi-scale features F_4, F_8, F_16

#### 3.2 Spatial Token-Level Router
- DWConv3x3 + MLP with local context
- Noisy top-k routing (k=2)

#### 3.3 Implicit Expert Pool
- TokenWiseMLP experts per scale (not shared across scales)
- Gate-weighted sum over active experts

#### 3.4 Multi-Scale Pyramid Integration
- Independent routers per scale
- Top-down cross-attention fusion

#### 3.5 Weighted Fusion + SOD Decoder
- Entropy channel fusion
- Progressive upsampling with refinement blocks
- Dual heads: saliency + boundary

#### 3.6 Loss Function
- BCE + IoU + load-balancing + importance + deep supervision

### 4. Expert Interpretability

#### 4.1 Routing Entropy Maps
- Per-token entropy → heatmap overlay
- Low entropy = confident assignment, high = ambiguous

#### 4.2 Per-Expert Token-Assignment Heatmaps
- Empirical expert profiles (local contrast, gradient magnitude, dark-channel prior)

#### 4.3 t-SNE / UMAP Cluster Analysis
- Token embeddings colored by expert assignment vs weather category
- Key: expert clusters correlate with weather but are not identical

#### 4.4 Ablated Single-Expert Forward Passes
- Force-route 100% to one expert, visualize degradation

#### 4.5 Routing Consistency Across Severity
- Plot expert assignment fraction vs severity for synthetic weather

### 5. Experiments

#### 5.1 Setup
- Dataset: WXSOD (train_sys/test_sys/test_real)
- Metrics: MAE, S-measure, E-measure, F-measure
- Implementation: PyTorch, DDP 2×GPU, AMP FP16

#### 5.2 Comparison with State-of-the-Art
- Compare on test_sys and test_real
- Weather-wise breakdown

#### 5.3 Ablation Studies

**A1. Spatial routing vs. global routing** (most important)
- Per-token router vs single global router
- Report overall + boundary-only metrics

**A2. Multi-scale independent vs. single-scale**
- Independent per scale vs shared routing
- Try routing at 1/8 alone as fallback

**A3. MoE vs. dense baseline at matched FLOPs**
- Does MoE earn its complexity?

**A4. Top-k and expert count sweep**
- k ∈ {1,2,3}, N ∈ {4,6,8,12}
- Report accuracy + load-balance curves

**A5. Auxiliary loss ablation**
- Remove load-balancing only
- Remove importance only
- Remove both
- Compare against Soft MoE alternative

#### 5.4 Qualitative Results
- Entropy maps across weather types
- Expert assignment heatmaps
- Failure cases

### 6. Conclusion
- Label-free spatial routing for SOD
- Multi-scale independent experts
- Strong generalization to real adverse weather

## Figures to Produce

1. **Architecture diagram** — End-to-end pipeline
2. **Router mechanism detail** — DWConv + MLP + noisy top-k
3. **Entropy map grid** — Clean / fog / rain / snow / low-light
4. **Expert assignment heatmaps** — Per-expert token distribution
5. **t-SNE visualization** — Expert clusters vs weather categories
6. **Ablation bar charts** — Spatial vs global, scale variants
7. **Weather-wise performance** — Bar chart per weather category
8. **Qualitative comparison** — Saliency maps across methods

## Existing Results for Paper

From `evaluation/best_new_1/`:

**Synthetic test (test_sys, 1500 images):**
- Global: MAE=0.0192, S=0.9139, F_max=0.9015
- Best: clean (MAE=0.0142)
- Worst: rainafog (MAE=0.0237)

**Real-world test (test_real, 554 images):**
- Global: MAE=0.0168, S=0.9151, F_max=0.8936
- Best: snow (MAE=0.0094)
- Worst: light (MAE=0.0243)

## Open Questions for Paper

1. What is the exact parameter count? (Run verification)
2. How does boundary F1 compare to SOTA? (Currently 0.36-0.55)
3. Why is real-world performance better than synthetic? (Investigate)
4. What do individual experts specialize in? (Run diagnostics)
5. Does the entropy channel actually help? (Ablation needed)
