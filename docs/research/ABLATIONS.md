# Ablation Studies

## Ablation System (`src/ablations.py`)

### Overview

The ablation system generates experiment config variants from a baseline, designed to be run through `train_ddp.py` with the experiment registry tracking results.

**Current status:** `experiments/registry.csv` exists (four rows written 2026-09-06, one of
them `COMPLETED`), but no cell of these generated matrices has been run to completion and
evaluated — `results/` contains no run produced by this system.

## Architecture Matrix (`ablations.py:20-68`)

Sweep over backbone, expert count, and top-k:

| Variant | Backbone | Experts | Top-K | Batch/GPU | Notes |
|---------|----------|---------|-------|-----------|-------|
| A | pvt_v2_b2 | 8 | 2 | 2 | Smaller backbone — **BROKEN, see below** |
| B | pvt_v2_b4 | 8 | 2 | 1 | Baseline equivalent |
| C | pvt_v2_b4 | 6 | 2 | 1 | Fewer experts |
| D | pvt_v2_b4 | 8 | 1 | 1 | Single expert per token |
| E | pvt_v2_b4 | 8 | 3 | 1 | More expert redundancy |

> **This matrix cannot currently test the backbone axis.** `config.model.backbone` is
> never passed to the model — `src/model.py` hard-codes PVTv2-B4 — so arm A and arm B
> would train *identical* architectures and differ only in batch shape and seed. Do not
> run or report a backbone comparison until that field is wired through
> (`build_model` -> `SpatialMoESODNet` -> `MultiScaleBackbone`).

**Batch equivalence adjustment** (`ablations.py:10-18`):
```python
def adjust_for_effective_batch(config, target_effective):
    world_size_assumed = 2  # Kaggle T4
    batch_gpu = config.opt.batch_per_gpu
    if target_effective % (batch_gpu * world_size_assumed) == 0:
        config.opt.grad_accum_steps = target_effective // (batch_gpu * world_size_assumed)
        config.batch_equivalence = "MATCHED"
    else:
        config.batch_equivalence = "NON_MATCHED"
```

## MoE Ladder (`ablations.py:70-91`)

Compare against non-MoE baselines:

| Variant | MoE Type | Description |
|---------|----------|-------------|
| none | `"none"` | Shared block (no expert routing, expert parameters removed) |
| dense | `"dense"` | One shared expert applied to every token |
| sparse | `"sparse"` | Sparse MoE (top-k routing) |

**Status:** `moe_type` *is* consumed — it selects the arm in `SpatialMoESODNet.__init__`
(`"none"` passes features through, `"dense"` applies one shared expert, `"sparse"` routes
top-k experts). However **none of these arms has been run**: `results/` contains zero
runs with `M_DENSE` or `M_NONE`, so the mixture-vs-dense comparison does not yet exist.

## Loss Matrix (`ablations.py:93-117`)

| Variant | BCE | IoU | SSIM | Boundary | Description |
|---------|-----|-----|------|----------|-------------|
| L1 | ✓ | ✓ | ✗ | ✗ | Minimal loss |
| L2 | ✓ | ✓ | ✓ | ✗ | + Structural similarity |
| L3 | ✓ | ✓ | ✓ | ✓ | + Edge supervision |

## Router Matrix (`ablations.py:119-140`)

| Variant | Router Type | Description |
|---------|-------------|-------------|
| R1 | token_only | Token embedding only |
| R2 | token_local | Token + local DWConv context |
| R3 | global | Image-level router |

**Note:** The `router_variant` config field exists but the router implementation (`moe_layer.py`) always uses the token+local context approach (equivalent to R2). The config field is not consumed by the router code.

## Blueprint Ablation Plan (`spatial-moe-adverse-weather-sod-blueprint.md:165-177`)

### Planned Ablations (from research blueprint)

1. **Spatial routing vs. global routing** — Most important ablation. Replace per-token router with single global router.

2. **Multi-scale independent vs. single-scale routing** — Collapse to routing at one scale vs. sharing routing across scales.

3. **MoE vs. dense baseline at matched FLOPs** — Essential "does MoE help" comparison.

4. **Top-k sparsity and expert count sweep** — k ∈ {1,2,3}, N ∈ {4,6,8,12}

5. **Auxiliary loss ablation** — Remove load-balancing only, remove importance only, remove both.

6. **Domain-generalization cross-check** — Train on synthetic, evaluate zero-shot on real WXSOD.

## Counterfactual Expert Ablation

### Implementation (`src/moe_layer.py`, `SpatialMoELayer.forward`)

The model supports forcing all tokens to a single expert via `ablation_cfg`:

```python
# In model.forward():
if ablation_cfg.get('scale') == 4:
    force_4 = ablation_cfg.get('expert_id')

# In SpatialMoELayer.forward():
if force_expert_id is not None:
    expert = self.experts[force_expert_id]
    flat_x = x_tokens.reshape(B * N_tokens, C)
    expert_out = expert(flat_x)
    # Set routing probs to 1.0 for forced expert
```

### Usage in Training (not wired to a CLI flag)

> **Not currently reachable from training.** The forward pass accepts an `ablation_cfg`
> (`scale` + `expert_id`, `random_routing`, `disable_entropy`), but no command-line parser
> sets `args.expert_ablation`, so a training-time forced-expert run cannot be launched today.
> The hook is used by `src/evaluate.py`, which is how the proxy-ablation table in `RESULTS.md`
> was produced.

```python
# In model.forward(): the ablation_cfg the evaluator passes
out_4 = self.moe_4(res_4, force_expert_id=force_4, random_routing=random_routing)
```

### Diagnostic Outputs (`src/diagnostics/`)

**RoutingTracker** (`src/diagnostics/`):
- Hard assignment counts per expert
- Soft gate mass per expert
- Mean entropy over the full E-way gate distribution (normalized by `ln E`)
- Dead expert detection (hard fraction < 0.01)

**WeatherAnalyzer** (`src/diagnostics/`):
- P(weather | expert) with Laplace smoothing
- KL divergence from global weather prior
- JS divergence

**Visualization** (`src/diagnostics/`):
- Per-expert hard assignment heatmaps
- Per-expert soft gate heatmaps
- Entropy heatmaps

**Status:** diagnostics *do* run in training — they execute at epoch boundaries and
write per-epoch routing statistics, which is where the expert-usage and weather-enrichment
numbers in `RESULTS.md` come from.

## DDP Ablation Tests (`tests/test_moe_ddp.py`)

### Test 1: Two-Rank Expert Activation
- Rank 0 routes to experts [0,1,3], Rank 1 to [1,2,5]
- Verifies gradient synchronization works with divergent routing

### Test 2: DDP Gradient Consistency
- Compares DDP gradient averaging vs single-process gradient
- Tolerance: max diff < 1e-3 (AMP FP16)

### Test 3: no_sync Equivalence
- Verifies N-1 microsteps of `no_sync()` + 1 sync step equals N fully synced steps
- Tolerance: max diff < 1e-5

### Test 4: DistributedSampler Resume
- Verifies sampler determinism when resuming mid-epoch

### Test 5: Exact Resume Reproducibility
- Run A: continuous 10 steps
- Run B: 5 steps → checkpoint → resume → 5 steps
- Compares final parameters, scaler state, loss (tolerance < 1e-5)
