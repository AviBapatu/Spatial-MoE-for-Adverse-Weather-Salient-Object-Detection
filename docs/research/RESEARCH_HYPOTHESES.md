# RESEARCH_HYPOTHESES.md — Proposed Ideas, Novelty Claims, and Future Experiments

This file contains ONLY hypotheses, proposed architecture ideas, novelty claims, and
future experiment ideas. Nothing here is a verified fact unless explicitly marked.

Verified facts belong in `RESEARCH_TRUTH.md`.

---

## 1. Central Novelty Claim (Proposed, Not Yet Verified)

**Claim:** Nobody has combined:
(a) fully label-free spatial/token routing (no weather label at train or test time),
(b) at multiple pyramid scales independently (separate routers at 1/4, 1/8, 1/16),
(c) for the segmentation-topology-sensitive SOD task (rather than pixel-regression restoration).

**Status:** PROPOSED ONLY. Requires literature verification (see `LITERATURE_VERIFICATION_QUEUE.md`).

**Key prior work to differentiate from:**
- WM-MoE (2023): weather-conditioned gating, feature-map routing, restoration task
- MoWE (2023): image/patch-level routing, weather-type conditioned
- Zhu et al. (CVPR 2023): channel-level (branch-level) routing, restoration task
- V-MoE (NeurIPS 2021): token-level routing but for classification, not dense SOD
- Soft MoE (ICLR 2024): differentiable weighting, not hard top-k

---

## 2. Proposed Architecture Ideas (from blueprint)

### 2.1 Token-Level Spatial Routing for SOD

**Hypothesis:** Routing at the spatial token level (rather than image-level or feature-map-level) is necessary for SOD because weather artifacts affect different spatial regions differently within a single image.

**Status:** IMPLEMENTED. The code routes each spatial token independently via top-k selection.

**What would validate this:** The image-level routing ablation (replacing per-token router with a single global router producing one expert-mixture weight for the whole image). This ablation code exists in `src/ablations.py:134-138` but has not been run.

### 2.2 Independent Per-Scale Routers

**Hypothesis:** A token that looks "hazy" at 1/16 may correspond to sharp rain streaks at 1/4. Independent routers per scale allow different routing decisions at different receptive fields.

**Status:** IMPLEMENTED. Three separate `SpatialMoELayer` instances, one per scale.

**What would validate this:** Collapse to routing only at one scale, or share one router's decision across all three scales. This ablation is proposed but NOT implemented in code.

### 2.3 Router Entropy as Decoder Input

**Hypothesis:** The per-token routing entropy gives the decoder an explicit "how confident/ambiguous was routing here" signal, which helps at object boundaries where degradation is most severe.

**Status:** IMPLEMENTED. Entropy is fused via `EntropyFusionBlock` (learned additive fusion with scale parameter initialized to 0.1).

**What would validate this:** Ablation removing entropy fusion from the decoder. This ablation is NOT implemented.

### 2.4 Implicit Expert Specialization

**Hypothesis:** With enough experts (N=8) and label-free routing, experts will implicitly specialize by weather artifact type without any weather label supervision.

**Status:** HYPOTHESIS ONLY. No interpretability analysis has been run.

**What would validate this:**
- t-SNE/UMAP clustering of pre-router tokens colored by expert assignment vs weather category
- Per-expert token-assignment heatmaps aggregated over test set
- Quantitative "expert profiles" (mean local statistics of assigned tokens)
- Forced single-expert forward passes (the counterfactual expert ablation code exists in `src/train_ddp.py:624-654`)

### 2.5 Router-Entropy-Boundary Correlation

**Hypothesis:** Routing entropy should be highest near saliency boundaries where degradation is most severe, providing the decoder with useful uncertainty information.

**Status:** HYPOTHESIS ONLY. No analysis done.

**What would validate this:** Plot entropy maps overlaid on input images across weather types. Compute correlation between entropy values and boundary regions.

---

## 3. Proposed Novelty Claims (All Require Literature Verification)

| Claim | Status | What must be verified |
|-------|--------|----------------------|
| "First token-level MoE for SOD" | PROPOSED | Check if any prior work applies token-level MoE to segmentation tasks |
| "First label-free weather routing" | PROPOSED | Check if WM-MoE/MoWE/etc. use weather labels at train time |
| "First independent multi-scale MoE routing" | PROPOSED | Check if M³ViT or others do independent per-scale routing |
| "Router entropy as decoder input is novel" | PROPOSED | Check if any prior work uses routing confidence as a decoder feature |
| "MoE for adverse-weather SOD is novel" | PROPOSED | Check if any prior work applies MoE specifically to weather-robust SOD |

None of these claims should appear in the paper without external verification.

---

## 4. Proposed Interpretability Analysis (Not Yet Run)

### 4.1 Routing Entropy Maps
Plot per-token routing entropy as heatmaps overlaid on input images across weather types. Low entropy = confident assignment, high entropy = ambiguous/contested region.

### 4.2 Per-Expert Token-Assignment Heatmaps
For each expert, render the hard assignment mask or soft gate weights as a heatmap. Aggregate over test set to compute quantitative expert profiles (local contrast, gradient magnitude, colorfulness, dark-channel prior).

### 4.3 t-SNE/UMAP Cluster Analysis
Collect pre-router token embeddings, label by (a) expert assignment and (b) weather category. Project and color to show expert clusters correlate with weather-artifact clusters without being identical.

### 4.4 Forced Single-Expert Forward Passes
Use the `force_expert_id` parameter (`src/moe_layer.py:69-94`) to route 100% of tokens to a single expert. Visualize resulting saliency maps per weather type. An expert specialized for fog should collapse on rain-heavy inputs.

### 4.5 Routing Consistency Across Weather Severity
For synthetic weather augmentation with continuous severity, plot expert-assignment fraction as a function of severity. Good specialization shows smooth monotonic hand-off.

---

## 5. Proposed Experiments (from blueprint, NOT yet run)

### Experiment 1: Image-Level vs Token-Level Routing
- **Hypothesis:** Spatial granularity is necessary for SOD
- **Variable:** Router type (per-token vs single global router)
- **Fixed:** Everything else (backbone, expert count, losses, training schedule)
- **Metric:** MAE, S_measure, boundary_F1, plus boundary-region-only metric split
- **Confounders:** Different router parameter counts; global router may need different hyperparameters
- **Implementation status:** Config generator exists (`src/ablations.py:134-138`), not run
- **Feasibility:** Feasible. The code supports `router_variant: "global"` config but the MoE layer always uses DWConv3x3 regardless of this config field — this would need a code change.

### Experiment 2: Multi-Scale Independent vs Shared Routing
- **Hypothesis:** Independent routing per receptive field is needed
- **Variable:** Whether routing is independent per scale or shared across scales
- **Fixed:** Everything else
- **Metric:** MAE, S_measure
- **Confounders:** Shared routing would need architectural changes
- **Implementation status:** NOT implemented (would require model code changes)
- **Feasibility:** Requires new model variant

### Experiment 3: MoE vs Matched-FLOPS Dense Baseline
- **Hypothesis:** MoE earns its complexity over a dense model of equivalent compute
- **Variable:** MoE type (sparse vs dense vs none)
- **Fixed:** Everything else
- **Metric:** MAE, S_measure, FLOPs
- **Confounders:** Matching FLOPs precisely is non-trivial; dense model may need different training
- **Implementation status:** Config generator exists (`src/ablations.py:70-91`) with `moe_type` field. BUT the MoE layer code does not check this field — it always runs sparse routing. Would need code changes.
- **Feasibility:** Partially feasible (requires code changes to support dense/no-MoE modes)

### Experiment 4: Top-k and Expert Count Sweep
- **Hypothesis:** k=2, N=8 is the sweet spot
- **Variable:** k ∈ {1,2,3}, N ∈ {4,6,8}
- **Fixed:** Everything else
- **Metric:** MAE, load-balance loss curve, importance loss curve
- **Confounders:** Different parameter counts across N values
- **Implementation status:** Config generator exists (`src/ablations.py:20-68`)
- **Feasibility:** Feasible

### Experiment 5: Auxiliary Loss Ablation
- **Hypothesis:** Each auxiliary loss prevents a specific failure mode
- **Variable:** Which auxiliary losses are active
- **Fixed:** Everything else
- **Metric:** MAE, expert utilization, load-balance curves
- **Confounders:** Removing aux losses may require re-tuning other hyperparameters
- **Implementation status:** Config generator exists (`src/ablations.py:93-117`)
- **Feasibility:** Feasible

### Experiment 6: Domain-Generalization Cross-Check
- **Hypothesis:** Training on synthetic weather augmentation enables zero-shot generalization to real weather
- **Variable:** Training data (synthetic only vs WXSOD)
- **Fixed:** Model architecture, evaluation protocol
- **Metric:** MAE on test_real
- **Confounders:** Different training distributions, different convergence
- **Implementation status:** NOT implemented
- **Feasibility:** Requires new training pipeline (synthetic weather augmentation onto clean SOD datasets)

---

## 6. Future Experiment Ideas (Not in Blueprint)

### 6.1 Clean SOD Benchmark Evaluation
Evaluate the trained model on standard clean SOD benchmarks (DUTS-test, ECSSD, HKU-IS, DUT-OMRON) to assess whether MoE routing hurts clean-condition performance.

### 6.2 Entropy Ablation
Remove the EntropyFusionBlock entirely to test whether entropy as decoder input actually helps.

### 6.3 Router Gradient Analysis
Analyze whether the router learns to attend to weather-relevant features (e.g., haze gradients, rain streak orientation) by visualizing router attention patterns.

### 6.4 Expert Pruning
After training, identify and prune unused experts to create a more efficient model.

### 6.5 Few-Shot Weather Adaptation
Fine-tune the router (freeze experts) on a small number of labeled weather images to test whether the expert pool generalizes.

---

*This file contains only hypotheses and proposals. Verified facts belong in RESEARCH_TRUTH.md.*
