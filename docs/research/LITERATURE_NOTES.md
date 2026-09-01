# Literature Notes

## Source
All notes from `spatial-moe-adverse-weather-sod-blueprint.md:26-52`

## Adverse-Weather-Aware Dense Prediction

### 1. TransWeather (Valanarasu et al., CVPR 2022)
- Transformer with learnable "weather-type" queries
- Single shared decoder handles multiple degradations
- **No explicit routing** — implicit specialization without MoE
- Good baseline for "what if we don't route at all"

### 2. Zhu et al. (CVPR 2023)
- "Learning Weather-General and Weather-Specific Features"
- Splits features into shared branch + degradation-specific branch
- **Channel-level split** (not spatial/token level)
- Two-branch precursor to token-level split
- Done at channel level, not spatial-token level

### 3. WM-MoE (2023)
- "Weather-aware Multi-scale Mixture-of-Experts for Blind Adverse Weather Removal"
- **Most directly relevant prior work**
- Combines multi-scale features with MoE experts + weather-conditioned gating
- Routes on feature maps with weather-conditioned gating
- **NOT per-token, weather-agnostic routing**

### 4. MoWE (2023)
- "Mixture of Weather Experts for Multiple Adverse Weather Removal"
- Closest "MoE" naming precedent
- **Routing granularity:** image/patch level (not token level)

### 5. Efficient Deweathering MoE (AAAI 2024)
- Uncertainty-gated FiLM modulation per expert
- Router's confidence signal as uncertainty estimate
- Relevant to interpretability section

### 6. Complexity Experts (Zamfir et al., CVPR 2025)
- Experts specialized by computational complexity
- **Difficulty-adaptive routing emerges implicitly**
- Most recent SOTA
- Supports "implicit expert specialization" design philosophy

### 7. GridFormer (2023)
- Residual dense grid transformer
- Multi-weather restoration
- Alternative multi-scale fusion baseline

## Token-level / Spatial MoE in Vision Transformers

### 8. V-MoE (Riquelme et al., NeurIPS 2021)
- Foundational token-level top-k routing for ViT
- **Direct architectural descendant** of this router formulation
- Reference for sparse routing mechanism

### 9. Soft MoE (Puigcerver et al., ICLR 2024)
- Replaces hard top-k with fully differentiable weighted combination
- Removes load-balancing losses and token-dropping
- **Strong alternative router design** for ablation comparison

### 10. M³ViT (Liang et al., NeurIPS 2022)
- Task-conditioned token routing for dense multi-task prediction
- Shows spatial tokens cluster by task-relevant local structure
- **Task = "weather artifact type" in this work**

### 11. Mod-Squad (Chen et al., CVPR 2023)
- Mixture of Experts as modular multi-task learners
- Precedent for spatial token clustering by task-relevant structure

### 12. Routers in Vision MoE (Liu et al., TMLR 2024)
- Empirical study of router designs
- **Useful for router-design ablations**
- Documents known MoE failure modes: router collapse, expert underutilization
- Motivates auxiliary losses (load-balancing, importance)

## Key Differentiators (from blueprint)

| Component | Precedent | Delta in This Work |
|-----------|-----------|-------------------|
| Token-level MoE routing | V-MoE, Soft MoE | Applied to dense SOD, not classification |
| Weather-aware MoE | WM-MoE, MoWE | **Label-free and spatial**, not weather-conditioned/global |
| Multi-scale MoE | M³ViT (multi-task) | **Independently-routed** pyramid scales with cross-attention |
| SOD boundary loss + MoE | No direct precedent | Router-entropy-as-decoder-input is novel |

## Related Benchmarks (from blueprint)

### Adverse-Weather SOD
- **WXSOD** (Chen et al., 2025): 14,945 images, weather labels, synthetic + real test splits
- **RGBT-SOD** (VT5000, VT1000, VT821): thermal-paired SOD

### Adverse-Weather Scene Understanding
- **Foggy Cityscapes / Foggy Driving**: synthetic fog via depth-aware scattering
- **RainCityscapes / Rain-KITTI / RainDrop**: rain streak and raindrop
- **Snow100K / CSD**: snow occlusion and veiling
- **ExDark**: low-light/nighttime
- **ACDC**: fog/rain/snow/night with pixel-level labels + clean correspondences
- **WeatherBench / WeatherStream**: multi-degradation training
- **XWOD** (2026): real-world extreme-weather detection (10K images)

### Clean SOD Benchmarks
- DUTS (train/test), ECSSD, HKU-IS, DUT-OMRON, PASCAL-S, SOD
- Needed for "does robustness cost clean-condition accuracy" ablation

## Citation Notes

For the paper, cite:
1. WM-MoE and MoWE as closest MoE weather restoration work
2. V-MoE as foundational token-level routing
3. Soft MoE as alternative router design
4. Routers in Vision MoE study for failure mode discussion
5. Complexity Experts (CVPR 2025) as most recent SOTA
6. WXSOD as the primary benchmark
7. ACDC for controlled weather ablation potential
