# Ablation Studies

## Ablation System (`src/ablations.py`)

### Overview

The ablation system generates experiment config variants from a baseline, designed to be run through `train_ddp.py` with the experiment registry tracking results.

**Current status:** Ablation matrices are defined but no `registry.csv` exists, indicating no formal ablation runs have been completed through this system.

## Architecture Matrix (`ablations.py:20-68`)

Sweep over backbone, expert count, and top-k:

| Variant | Backbone | Experts | Top-K | Batch/GPU | Notes |
|---------|----------|---------|-------|-----------|-------|
| A | pvt_v2_b2 | 8 | 2 | 2 | Smaller backbone |
| B | pvt_v2_b4 | 8 | 2 | 1 | Baseline equivalent |
| C | pvt_v2_b4 | 6 | 2 | 1 | Fewer experts |
| D | pvt_v2_b4 | 8 | 1 | 1 | Single expert per token |
| E | pvt_v2_b4 | 8 | 3 | 1 | More expert redundancy |

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
| none | `"none"` | Shared block (no expert routing) |
| dense | `"dense"` | Dense MoE (all experts evaluated) |
| sparse | `"sparse"` | Sparse MoE (top-k routing) |

**Note:** The `moe_type` config field exists but `SpatialMoELayer` always runs sparse dispatch. The "none" and "dense" variants are not currently implemented in the forward path — the layer ignores this config field.

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

### Implementation (`moe_layer.py:68-94`)

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

### Usage in Training (`train_ddp.py:623-654`)

```python
# Passed via --expert_ablation "scale,expert_id"
expert_ablation = getattr(args, 'expert_ablation', None)
if expert_ablation is not None:
    scale, exp_id = map(int, expert_ablation.split(','))
    ablation_cfg = {'scale': scale, 'expert_id': exp_id}
    out, _ = model.module(v_images, ablation_cfg=ablation_cfg)
```

### Diagnostic Outputs (`src/diagnostics.py`)

**RoutingTracker** (`diagnostics.py:12-78`):
- Hard assignment counts per expert
- Soft gate mass per expert
- Mean entropy (normalized by log(K))
- Dead expert detection (hard fraction < 0.01)

**WeatherAnalyzer** (`diagnostics.py:80-177`):
- P(weather | expert) with Laplace smoothing
- KL divergence from global weather prior
- JS divergence

**Visualization** (`diagnostics.py:179-270`):
- Per-expert hard assignment heatmaps
- Per-expert soft gate heatmaps
- Entropy heatmaps

**Note:** Diagnostics are disabled in production training (`train_ddp.py:560`):
```python
if False and config.diag.routing_diagnostic_epochs > 0 and ...:
    # SKIPPED
```

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
