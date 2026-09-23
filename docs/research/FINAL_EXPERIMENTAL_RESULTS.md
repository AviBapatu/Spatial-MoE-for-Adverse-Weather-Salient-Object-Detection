# FINAL EXPERIMENTAL RESULTS — Complete Audit

**Audit date:** 2026-09-02
**Source of truth:** `results/legacy/legacy_8expert/evaluation_results/`, `evaluation/`, source code, configs
**Scope:** All completed experiments as of audit date

---

## 1. Experimental Inventory

### 1.1 Checkpoints

| Checkpoint | Location | Model Hash | Notes |
|------------|----------|------------|-------|
| `best.pth` | `/kaggle/working/WXSOD_Checkpoints/best.pth` | `2cd252ad3a3581e1c65961b828857d70` | Kaggle training output |
| `best_new_1.pth` | `checkpoints/best_new_1.pth` | `2cd252ad3a3581e1c65961b828857d70` | Local copy — **same weights** |

**Critical finding:** Both checkpoints have identical model hashes. They are the same trained model with different filenames.

### 1.2 Configuration

| Item | Value | Source |
|------|-------|--------|
| Config file | `experiments/baseline_v1.json` | — |
| Backbone | PVTv2-B4 | `src/backbone.py:11` |
| Working dim | 256 | `experiments/baseline_v1.json:14` |
| Experts per scale | 8 | `experiments/baseline_v1.json:14` |
| Top-k | 2 | `experiments/baseline_v1.json:15` |
| Router | DWConv3x3 + MLP (2-layer) | `src/moe_layer.py:49-57` |
| MoE type | Sparse (true sparse dispatch) | `src/moe_layer.py:119-141` |
| Decoder | Cross-attention fusion | `src/decoder.py:232-233` |
| Window size | 7 | `experiments/baseline_v1.json:18` |
| Deep supervision | OFF (weight=0.4 but config=false) | `experiments/baseline_v1.json:20` |
| SSIM loss | OFF (weight=0.0) | `experiments/baseline_v1.json:27` |
| Boundary loss | OFF (weight=0.0) | `experiments/baseline_v1.json:28` |
| Load balance weight | 0.01 | `experiments/baseline_v1.json:29` |
| Importance weight | 0.01 | `experiments/baseline_v1.json:30` |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 | `experiments/baseline_v1.json:36-39` |
| Scheduler | WarmupCosine (warmup=1%) | `experiments/baseline_v1.json:42` |
| Epochs | 50 | `experiments/baseline_v1.json:44` |
| Effective batch | 32 (1 per GPU × 2 GPUs × 16 accum) | `experiments/baseline_v1.json:48-50` |
| AMP | FP16 | `experiments/baseline_v1.json:43` |
| Image size | 384×384 | `src/dataset.py` |

### 1.3 Complete File Inventory

```
results/legacy/legacy_8expert/evaluation_results/
├── best/
│   ├── test_real/none/
│   │   ├── metrics.json                          ← boundary_MAE=0.4941, boundary_F1=0.0659
│   │   ├── summary.txt
│   │   ├── evaluation_manifest.json
│   │   └── 554 PNGs (predictions)
│   └── test_sys/none/
│       ├── metrics.json                          ← boundary_MAE=0.5494, boundary_F1=0.1494
│       ├── summary.txt
│       ├── evaluation_manifest.json
│       └── 1500 PNGs (predictions)
├── force_expert_out/
│   └── 554 PNGs (forced-expert predictions, test_real)
├── metrics_test_real_20260901_173227.json        ← boundary_MAE=0.6334, boundary_F1=0.3601
├── metrics_test_sys_20260901_172914.json         ← boundary_MAE=0.4817, boundary_F1=0.5472
├── summary_test_real_20260901_173227.txt
├── summary_test_sys_20260901_172914.txt
├── evaluation_manifest_test_real_20260901_173227.json
├── evaluation_manifest_test_sys_20260901_172914.json
├── force_expert_ablation.json
├── entropy_comparison.json
├── compute_cost.json
└── qualitative_grid.png

results/legacy/legacy_8expert/evaluation/best_new_1/
├── test_real/none/
│   ├── metrics.json                              ← boundary_MAE=0.6334, boundary_F1=0.3601
│   ├── summary.txt
│   ├── evaluation_manifest.json
│   └── 554 PNGs (predictions)
└── test_sys/none/
    ├── metrics.json                              ← boundary_MAE=0.4817, boundary_F1=0.5472
    ├── summary.txt
    ├── evaluation_manifest.json
    └── 1500 PNGs (predictions)
```

### 1.4 Experiment Registry

| ID | Checkpoint | Dataset | Split | N | MAE | S_measure | Boundary Source | Status |
|----|-----------|---------|-------|---|-----|-----------|----------------|--------|
| E1 | best.pth | test_real | none | 554 | 0.0168 | 0.9151 | `results/legacy/legacy_8expert/evaluation_results/best/` (kernel=5, ellipse) | COMPLETED |
| E2 | best.pth | test_sys | none | 1500 | 0.0192 | 0.9139 | `results/legacy/legacy_8expert/evaluation_results/best/` (kernel=5, ellipse) | COMPLETED |
| E3 | best.pth | test_real | none | 554 | 0.0168 | 0.9151 | timestamped run (kernel=5, ellipse) | COMPLETED |
| E4 | best.pth | test_sys | none | 1500 | 0.0192 | 0.9139 | timestamped run (kernel=5, ellipse) | COMPLETED |
| E5 | best_new_1.pth | test_real | none | 554 | 0.0168 | 0.9151 | `results/legacy/legacy_8expert/evaluation/best_new_1/` (kernel=5, ellipse) | COMPLETED |
| E6 | best_new_1.pth | test_sys | none | 1500 | 0.0192 | 0.9139 | `results/legacy/legacy_8expert/evaluation/best_new_1/` (kernel=5, ellipse) | COMPLETED |
| A1 | best.pth (forced expert 0, scale 4) | test_real | none | 554 | 0.0170 | 0.9139 | force_expert_ablation.json | COMPLETED |

**Note:** E1/E2 and E3/E4 and E5/E6 all use the **same model** (hash `2cd252ad3a3581e1c65961b828857d70`). The core SOD metrics (MAE, S_measure, E_*, F_*) are **identical** across all six. Only boundary metrics differ between evaluation runs.

---

## 2. Canonical Baseline Results

The canonical baseline is the **SpatialMoESODNet** (PVTv2-B4 backbone, 8 experts per scale, top-k=2, sparse routing, cross-attention decoder) trained on WXSOD train_sys and evaluated on test_sys and test_real.

### 2.1 Synthetic Test Set (test_sys, 1,500 images)

**Source:** `results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/metrics.json` (or equivalently `results/legacy/legacy_8expert/evaluation_results/metrics_test_sys_20260901_172914.json`)

| Metric | Value |
|--------|-------|
| MAE | 0.0192 |
| S_measure | 0.9139 |
| E_adaptive | 0.9591 |
| E_mean | 0.9585 |
| E_max | 0.9633 |
| F_adaptive | 0.8822 |
| F_mean | 0.8886 |
| F_max | 0.9015 |
| boundary_MAE | 0.4817 |
| boundary_F1 | 0.5472 |

### 2.2 Real-World Test Set (test_real, 554 images)

**Source:** `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json` (or equivalently `results/legacy/legacy_8expert/evaluation_results/metrics_test_real_20260901_173227.json`)

| Metric | Value |
|--------|-------|
| MAE | 0.0168 |
| S_measure | 0.9151 |
| E_adaptive | 0.9530 |
| E_mean | 0.9551 |
| E_max | 0.9608 |
| F_adaptive | 0.8631 |
| F_mean | 0.8747 |
| F_max | 0.8936 |
| boundary_MAE | 0.6334 |
| boundary_F1 | 0.3601 |

---

## 3. Master Results Table

Since only one model configuration was trained and evaluated, the "master table" is:

| Experiment | Variant | Checkpoint | Dataset | Split | N | MAE | S_measure | F_mean | F_max | E_adaptive | Boundary MAE | Boundary F1 | Notes |
|------------|---------|------------|---------|-------|---|-----|-----------|--------|-------|------------|-------------|-------------|-------|
| Baseline | Full MoE | best.pth | test_sys | none | 1500 | 0.0192 | 0.9139 | 0.8886 | 0.9015 | 0.9591 | 0.4817 | 0.5472 | Canonical |
| Baseline | Full MoE | best.pth | test_real | none | 554 | 0.0168 | 0.9151 | 0.8747 | 0.8936 | 0.9530 | 0.6334 | 0.3601 | Canonical |
| Forced Expert | Expert 0, scale 4 only | best.pth | test_real | none | 554 | 0.0170 | 0.9139 | 0.8713 | 0.8928 | 0.9510 | 0.6349 | 0.3558 | Single expert forced |

---

## 4. Ablation Analysis

### 4.1 Forced-Expert Ablation (A1)

**What changed:** All tokens at scale 4 forced to expert 0 (router bypassed).
**What stayed fixed:** Experts at scales 8 and 16 still routed normally. Same checkpoint, same dataset.
**Dataset:** test_real (554 images)

| Metric | Normal Routing | Forced Expert 0 (scale 4) | Delta | Relative |
|--------|---------------|---------------------------|-------|----------|
| MAE | 0.0168 | 0.0170 | +0.0002 | +1.19% |
| S_measure | 0.9151 | 0.9139 | -0.0012 | -0.13% |
| E_adaptive | 0.9530 | 0.9510 | -0.0020 | -0.21% |
| F_mean | 0.8747 | 0.8713 | -0.0034 | -0.39% |
| F_max | 0.8936 | 0.8928 | -0.0008 | -0.09% |
| boundary_MAE | 0.6334 | 0.6349 | +0.0015 | +0.24% |
| boundary_F1 | 0.3601 | 0.3558 | -0.0043 | -1.20% |

**Assessment:** PARTIALLY CONTROLLED. This test only forces one scale (4) to one expert. It does NOT test:
- What happens when ALL scales are forced
- Whether different experts produce different outputs
- Whether the degradation is consistent across weather types
- Whether the remaining routing at scales 8/16 compensates

**Weather-wise forced vs normal:**

| Weather | Normal MAE | Forced MAE | Delta | Normal S | Forced S | Delta |
|---------|-----------|-----------|-------|----------|----------|-------|
| dark | 0.0191 | 0.0192 | +0.0001 | 0.9072 | 0.9066 | -0.0006 |
| fog | 0.0148 | 0.0150 | +0.0002 | 0.9179 | 0.9167 | -0.0012 |
| light | 0.0243 | 0.0246 | +0.0003 | 0.8897 | 0.8869 | -0.0028 |
| rain | 0.0165 | 0.0166 | +0.0001 | 0.9157 | 0.9146 | -0.0011 |
| snow | 0.0094 | 0.0094 | +0.0000 | 0.9478 | 0.9472 | -0.0006 |

Degradation is tiny across all weather types. The effect is largest on "light" (low-light), smallest on "snow".

### 4.2 Planned But NOT Completed Ablations

The following ablations are defined in `src/ablations.py` and `docs/research/ABLATIONS.md` but have **zero completed runs**:

| Ablation | Description | Status |
|----------|-------------|--------|
| Spatial vs global routing | Per-token router vs single global router | NOT RUN |
| Multi-scale independent vs shared | Independent per-scale vs shared routing | NOT RUN |
| MoE vs dense baseline at matched FLOPs | Does MoE earn its complexity? | NOT RUN |
| Top-k sweep (k∈{1,2,3}) | Effect of sparsity level | NOT RUN |
| Expert count sweep (N∈{4,6,8,12}) | Effect of expert count | NOT RUN |
| Loss ablation (SSIM, boundary) | Effect of loss components | NOT RUN |
| Router variant ablation (R1/R2/R3) | Token-only vs local vs global router | NOT RUN |
| No-MoE baseline | Shared block without expert routing | NOT RUN |
| Dense MoE baseline | All experts evaluated, no sparsity | NOT RUN |

**None of the config fields `moe_type` or `router_variant` are consumed by the forward path.** The code always runs sparse dispatch regardless of these config values (verified at `src/moe_layer.py:68-94` and `src/moe_layer.py:119-141`).

---

## 5. Weather-Wise Results

### 5.1 test_sys (synthetic, 9 weather categories)

| Weather | N | MAE | S_measure | F_mean | F_max | E_adaptive |
|---------|---|-----|-----------|--------|-------|------------|
| clean | 167 | 0.0142 | 0.9329 | 0.9101 | 0.9228 | 0.9690 |
| fog | 166 | 0.0166 | 0.9308 | 0.9128 | 0.9255 | 0.9696 |
| snow | 172 | 0.0189 | 0.9192 | 0.8940 | 0.9068 | 0.9647 |
| rain | 179 | 0.0200 | 0.9173 | 0.8909 | 0.9030 | 0.9542 |
| dark | 156 | 0.0199 | 0.9049 | 0.8725 | 0.8836 | 0.9539 |
| light | 166 | 0.0215 | 0.9091 | 0.8903 | 0.9046 | 0.9623 |
| rainafog | 172 | 0.0237 | 0.8902 | 0.8547 | 0.8691 | 0.9384 |
| rainasnow | 166 | 0.0194 | 0.9152 | 0.8916 | 0.9032 | 0.9637 |
| snowafog | 156 | 0.0189 | 0.9051 | 0.8800 | 0.8950 | 0.9567 |

**Best:** clean (MAE=0.0142). **Worst:** rainafog (MAE=0.0237).

### 5.2 test_real (real-world, 5 weather categories)

| Weather | N | MAE | S_measure | F_mean | F_max | E_adaptive |
|---------|---|-----|-----------|--------|-------|------------|
| snow | 90 | 0.0094 | 0.9478 | 0.9268 | 0.9421 | 0.9825 |
| fog | 126 | 0.0148 | 0.9179 | 0.8673 | 0.8914 | 0.9560 |
| rain | 120 | 0.0165 | 0.9157 | 0.8813 | 0.8935 | 0.9645 |
| dark | 125 | 0.0191 | 0.9072 | 0.8696 | 0.8885 | 0.9477 |
| light | 93 | 0.0243 | 0.8897 | 0.8323 | 0.8629 | 0.9129 |

**Best:** snow (MAE=0.0094). **Worst:** light (MAE=0.0243).

### 5.3 Weather Robustness Analysis

| Condition | test_sys MAE | test_real MAE | Notes |
|-----------|-------------|---------------|-------|
| Snow | 0.0189 | 0.0094 | Real-world snow easier than synthetic |
| Fog | 0.0166 | 0.0148 | Real-world fog slightly easier |
| Rain | 0.0200 | 0.0165 | Real-world rain slightly easier |
| Dark | 0.0199 | 0.0191 | Nearly equivalent |
| Light | 0.0215 | 0.0243 | Real-world low-light harder |
| Compound (rainafog) | 0.0237 | N/A | Worst condition on synthetic |

---

## 6. Synthetic vs Real Analysis

### 6.1 Global Comparison

| Split | N | MAE | S_measure | F_mean | E_adaptive |
|-------|---|-----|-----------|--------|------------|
| test_sys | 1500 | 0.0192 | 0.9139 | 0.8886 | 0.9591 |
| test_real | 554 | 0.0168 | 0.9151 | 0.8747 | 0.9530 |

Real-world MAE is **lower** (better) than synthetic: 0.0168 vs 0.0192. This is counterintuitive and may reflect:
- Synthetic test set contains harder compound weather conditions (rainafog, rainasnow, snowafog) not present in real
- Real-world weather may be less severe than synthetically generated weather
- Different image characteristics between synthetic and real captures

### 6.2 Domain Generalization Assessment

**This is NOT a domain generalization experiment.** The model was trained on `train_sys` (synthetic images). Both `test_sys` and `test_real` are test splits. The model was never trained on real-world weather images.

However, the comparison between test_sys and test_real performance on matching weather categories (snow, fog, rain, dark, light) provides **indirect evidence** of synthetic-to-real transfer. The model performs comparably or better on real-world weather, suggesting the synthetic training data captures weather-relevant features effectively.

**Cannot claim:** "The model generalizes from synthetic to real" as a rigorous domain generalization claim, because:
1. No controlled domain-shift experiment was designed
2. No clean-source vs weather-target evaluation exists
3. The test_real images may share distributional properties with train_sys

---

## 7. SOTA / Baseline Comparison

**NO DIRECT SOTA COMPARISON AVAILABLE.**

The `results/legacy/legacy_8expert/evaluation_results/` directory contains no comparisons against:
- WFANet
- NIFM
- Any other WXSOD method
- Any clean-domain SOD method

No benchmark numbers from other papers have been recorded in the evaluation files.

**Cannot claim:** "Our method outperforms SOTA" or "Our method achieves competitive performance" without running direct comparisons under identical evaluation protocol.

---

## 8. Interpretability Results

### 8.1 Routing Entropy

**Source:** `results/legacy/legacy_8expert/evaluation_results/entropy_comparison.json`

| Split | Scale 4 (1/4) | Scale 8 (1/8) | Scale 16 (1/16) |
|-------|---------------|---------------|-----------------|
| Synthetic | 0.6910 | 0.6931 | 0.6800 |
| Real | 0.6912 | 0.6931 | 0.6822 |

Mean routing entropy across all tokens and all images. Normalized by log(2) (assuming K=2, maximum entropy = 1.0).

**Observations:**
- Entropy is very similar between synthetic and real across all scales
- Scale 16 has slightly lower entropy (more confident routing) than scales 4 and 8
- All scales have entropy around 0.68-0.69, indicating moderate routing uncertainty (not collapsing to one expert, not uniformly random)

### 8.2 Expert Assignment / Specialization

**NO EVIDENCE AVAILABLE.**

The following analyses were planned but not completed:
- Per-expert hard assignment counts
- Per-expert soft gate mass distributions
- Weather | expert conditional distributions
- KL/JS divergence from weather prior
- Per-expert assignment heatmaps
- t-SNE/UMAP of expert embeddings

The `RoutingTracker` and `WeatherAnalyzer` classes exist in `src/diagnostics.py` but are **disabled in production training** (`train_ddp.py:560` has `if False and ...`).

**Cannot claim:** "Experts specialize by weather/artifact type" without actual per-expert assignment statistics.

### 8.3 Forced-Expert Visual Analysis

The `force_expert_out/` directory contains 554 prediction PNGs from forcing all tokens to expert 0 at scale 4. However, no quantitative comparison of these predictions vs normal predictions per-image exists. Only the aggregate metrics in `force_expert_ablation.json` are available.

### 8.4 Entropy-Boundary Correlation

**NO EVIDENCE AVAILABLE.** No analysis correlating routing entropy with prediction difficulty or boundary accuracy has been performed.

---

## 9. Computational Cost

**Source:** `results/legacy/legacy_8expert/evaluation_results/compute_cost.json`

| Metric | Value |
|--------|-------|
| Parameters | 66.27M |
| MACs | 278.2G |
| FPS | 3.83 |

**Notes:**
- FPS was likely measured on a specific GPU (unspecified — probably Kaggle T4 or similar)
- No latency measurements at different batch sizes
- No GPU memory measurements
- No throughput analysis
- No comparison to non-MoE baseline compute cost

**Cannot claim:** "The method is computationally efficient" without:
1. Comparison to a non-MoE baseline at matched parameter count
2. Latency measurements on standardized hardware
3. Memory usage analysis

---

## 10. Result Consistency Audit

### 10.1 Duplicate Results

| Issue | Details | Severity |
|-------|---------|----------|
| Two checkpoints, same model | `best.pth` and `best_new_1.pth` have identical model hash | LOW — same model, different filenames |
| Three evaluation runs of same model | `results/legacy/legacy_8expert/evaluation/best_new_1/`, `results/legacy/legacy_8expert/evaluation_results/best/`, timestamped files | LOW — core metrics match |
| No prediction images in timestamped runs | Only metrics/summary saved, not per-image PNGs | LOW — metrics are the canonical output |

### 10.2 Boundary Metric Discrepancy

**CRITICAL ISSUE:** Two different boundary metric values exist for the same model on the same datasets.

| Source | test_real boundary_MAE | test_real boundary_F1 | test_sys boundary_MAE | test_sys boundary_F1 |
|--------|----------------------|---------------------|---------------------|---------------------|
| `results/legacy/legacy_8expert/evaluation_results/best/` | 0.4941 | 0.0659 | 0.5494 | 0.1494 |
| `results/legacy/legacy_8expert/evaluation/best_new_1/` + timestamped | 0.6334 | 0.3601 | 0.4817 | 0.5472 |

**Root cause:** The `results/legacy/legacy_8expert/evaluation_results/best/` subdirectory appears to have been produced by an earlier evaluation run with a different boundary metric implementation. The timestamped files and `results/legacy/legacy_8expert/evaluation/best_new_1/` are consistent with each other.

**Resolution:** The `results/legacy/legacy_8expert/evaluation/best_new_1/` and timestamped results should be treated as canonical, since they are internally consistent and use the same boundary computation as the current `src/metrics.py`.

### 10.3 Core Metric Consistency

All six evaluation runs of the same model produce **identical** core SOD metrics:
- MAE: 0.0192 (test_sys), 0.0168 (test_real) — exact match across all runs
- S_measure: 0.9139 (test_sys), 0.9151 (test_real) — exact match
- E_adaptive: 0.9591 (test_sys), 0.9530 (test_real) — exact match
- F_mean: 0.8886 (test_sys), 0.8747 (test_real) — exact match

This confirms the evaluation pipeline is deterministic and reproducible.

### 10.4 Sample Count Consistency

| Split | Expected | Actual | Status |
|-------|----------|--------|--------|
| test_sys | 1500 | 1500 | ✓ |
| test_real | 554 | 554 | ✓ |

Weather category counts are consistent across all evaluation runs for each split.

---

## 11. Supported Claims

Based on the completed experiments, the following claims are supported:

### 11.1 Strongly Supported

1. **The model achieves MAE = 0.0192 on synthetic test (1,500 images) and MAE = 0.0168 on real-world test (554 images).**
   - Evidence: `results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/metrics.json`, `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json`

2. **The model performs across 5 real-world weather categories and 9 synthetic weather categories.**
   - Evidence: Weather-wise breakdowns in all metrics.json files

3. **Snow is the easiest weather condition; low-light is the hardest.**
   - Evidence: test_real weather-wise metrics (snow MAE=0.0094, light MAE=0.0243)

4. **Compound weather conditions (rainafog) are the most challenging on synthetic data.**
   - Evidence: test_sys weather-wise metrics (rainafog MAE=0.0237, worst among 9 categories)

5. **Forcing all tokens at one scale to a single expert causes only marginal degradation (ΔMAE = +0.0002 on test_real).**
   - Evidence: `results/legacy/legacy_8expert/evaluation_results/force_expert_ablation.json`

6. **The model has 66.27M parameters, 278.2G MACs, and runs at 3.83 FPS (hardware unspecified).**
   - Evidence: `results/legacy/legacy_8expert/evaluation_results/compute_cost.json`

### 11.2 Partially Supported

7. **Routing entropy is similar between synthetic and real data.**
   - Evidence: `results/legacy/legacy_8expert/evaluation_results/entropy_comparison.json` (scale-wise means)
   - Limitation: Only mean entropy reported; no per-token or per-weather analysis

8. **The model handles synthetic-to-real transfer without severe degradation.**
   - Evidence: Real-world MAE (0.0168) is comparable to or better than synthetic MAE (0.0192)
   - Limitation: Not a controlled domain generalization experiment

### 11.3 NOT Supported (cannot be claimed)

9. **"Token-level spatial routing is useful for SOD"** — No ablation comparing routed vs non-routed model exists.
10. **"No weather-specific component is useful/sufficient"** — No weather-label-conditioned baseline exists for comparison.
11. **"Independent per-scale routing is useful"** — No ablation comparing independent vs shared routing exists.
12. **"Routing entropy in the decoder is useful"** — No ablation with/without entropy fusion exists.
13. **"MoE is useful for adverse-weather SOD"** — No non-MoE baseline exists for comparison.
14. **"Experts specialize by weather/artifact type"** — No per-expert assignment analysis exists.
15. **"The model generalizes from synthetic to real"** — No controlled domain-shift experiment.
16. **"Our method outperforms SOTA"** — No comparison to other methods.
17. **"The model is computationally efficient"** — No comparison to baselines or standardized measurements.

---

## 12. Remaining Limitations

1. **No ablation studies completed.** The ablation system (`src/ablations.py`) exists but zero runs have been executed. No `registry.csv` exists.

2. **No SOTA comparison.** No other WXSOD methods have been evaluated under the same protocol.

3. **No interpretability analysis.** The `RoutingTracker` and `WeatherAnalyzer` diagnostics are disabled in training. No per-expert statistics exist.

4. **No non-MoE baseline.** Cannot attribute performance gains to MoE architecture specifically.

5. **No controlled domain generalization test.** The synthetic→real comparison is observational, not experimental.

6. **Boundary metric ambiguity.** Two different boundary metric implementations produce different values. The canonical values are from `results/legacy/legacy_8expert/evaluation/best_new_1/`.

7. **Hardware context for compute measurements is incomplete.** FPS of 3.83 is reported without specifying GPU model, batch size, or resolution.

8. **Config fields `moe_type` and `router_variant` are not consumed by code.** The ablation system defines these variants but the forward path ignores them.

9. **Single training run.** No multiple seeds or repeated training to assess variance.

10. **No validation curves or training logs.** Cannot assess convergence, overfitting, or training stability.
