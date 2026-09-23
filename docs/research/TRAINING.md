# Training Pipeline

> **Note on citations.** Line-number citations below predate the current source
> layout (the decoder is now the `src/decoder/` package, the training loop lives under
> `src/training/`, the evaluation artifacts are under `results/`). Treat module and
> function names as authoritative and `RESEARCH_TRUTH.md` as the verified reference;
> numbers live in `RESULTS.md`, which is generated from the result files.

## Entry Points

| Script | Use Case | GPUs |
|--------|----------|------|
| `src/train_ddp.py` | Production DDP training | 2+ (NCCL) |
| `train.py` | Legacy single-GPU (has bug) | 1 |

**Known bug in `train.py:61`:** `get_dataloaders()` returns 4 values but the call unpacks 3. Use `train_ddp.py` instead.

## DDP Training (`src/train_ddp.py`)

### Launch
```bash
torchrun --nproc_per_node=2 -m src.train_ddp --config experiments/baseline_v1.json
```

### Initialization Sequence (`train_ddp.py:136-268`)

1. `dist.init_process_group("nccl")` — NCCL backend required
2. Load `ExperimentConfig` from JSON
3. Validate dataset root exists
4. Require `world_size >= 2` for production (not dry_run)
5. `setup_experiment_run()` — creates run directory, saves config, registers in CSV
6. Broadcast `run_dir` from rank 0 to all ranks
7. Deterministic seeding: `seed + rank` for each rank
8. `get_dataloaders(distributed=True)` — returns train/val/test loaders

### Model Construction (`train_ddp.py:264-297`)

```python
model = SpatialMoESODNet(
    use_deep_supervision=config.model.deep_supervision,
    num_experts=config.model.num_experts,
    window_size=config.model.window_size
).to(device)
```

**Architecture consistency check:** MD5 hash of sorted parameter names compared across all ranks via `all_gather`. Mismatch → RuntimeError.

**BatchNorm check:** Static inspection logs any BatchNorm layers (none expected in this architecture — uses GroupNorm/LayerNorm).

**DDP wrap:** `nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)`

`find_unused_parameters=True` is required because sparse routing means some experts receive zero tokens per step, making their parameters "unused" in that forward pass.

### Optimization (`train_ddp.py:300-310`)

**Parameter groups** (`src/optimization.py:6-72`):
- `backbone_decay` / `backbone_no_decay`: LR = `backbone_lr` (1e-4)
- `moe_decay` / `moe_no_decay`: LR = `new_module_lr` (1e-4)
- `decoder_decay` / `decoder_no_decay`: LR = `new_module_lr` (1e-4)
- `heads_decay` / `heads_no_decay`: LR = `new_module_lr` (1e-4)

No-decay groups: biases, normalization params, relative_position_bias_table.

**Scheduler:** `WarmupCosineScheduler` (`optimization.py:84-102`)
- Warmup: `warmup_ratio * total_steps` (default 1%)
- Cosine decay to `min_lr_ratio=0.01` of base LR

**AMP:** FP16 via `torch.amp.autocast` + `GradScaler`

**Gradient clipping:** `max_norm=1.0` via `nn.utils.clip_grad_norm_`

### Training Loop (`train_ddp.py:374-655`)

```
for epoch in range(start_epoch, config.train.epochs):
    1. model.train()
    2. set_epoch(epoch) on sampler and dataset
    3. Freeze backbone epoch 0, unfreeze epoch 1
    4. For each batch:
       a. Forward with autocast
       b. Compute loss
       c. Scale loss, backward
       d. On final microstep: unscale → clip → step → update scaler
       e. Checkpoint every N steps
    5. Validation on rank 0 only
    6. Save best model if val MAE improved
```

**Gradient accumulation:**
```python
is_final_microstep = (batch_idx + 1) % config.opt.grad_accum_steps == 0
sync_context = model.no_sync() if not is_final_microstep else nullcontext()
```
Non-final microsteps skip DDP gradient synchronization via `no_sync()`.

**Backbone freezing:**
```python
if epoch == 0 and not args.resume:
    freeze_backbone(model.module)  # epoch 0: warmup
elif epoch == 1:
    unfreeze_backbone(model.module)  # epoch 1+: full training
```

### Checkpointing (`train_ddp.py:446-497`)

**Format version:** `CHECKPOINT_FORMAT_VERSION = 1`

**Atomic write pattern:**
```python
torch.save(state, tmp_path)           # Write to .tmp
_ = torch.load(tmp_path)              # Validate readable
os.replace(tmp_path, final_path)      # Atomic rename
```

**Checkpoint contents:**
```python
{
    'checkpoint_format_version': 1,
    'epoch': int,
    'batch_in_epoch': int,
    'global_step': int,
    'best_metric': float,
    'model_state_dict': dict,
    'engine_state_dict': {
        'optimizer': ...,
        'scheduler': ...,
        'scaler': ...,
        'global_step': int
    },
    'rng_states': {
        'python': ...,
        'numpy': ...,
        'torch_cpu': ...,
        'torch_cuda': ...
    },
    'config': dict,
    'config_hash': str,
    'world_size': int,
    'timestamp': float
}
```

**Resume validation (`train_ddp.py:342-349`):**
1. `checkpoint_format_version` must match
2. `world_size` must match current run
3. `config_hash` must match current config

**Best model:** Saved when `val_mae < best_mae` with all same fields.

### HuggingFace Sync (`src/hf_sync/`)

**Async pusher** (`hf_sync.py:437-491`): Background thread with FIFO queue, never blocks training. Enqueued after checkpoint write.

**Repo layout** (HF dataset repo):
```
code/spatial_moe_sod_code.zip
code/project_manifest.json
checkpoints/latest.pth
checkpoints/best.pth
checkpoints/checkpoint_manifest.json
```

### Validation (`train_ddp.py:529-655`)

- Runs on rank 0 only
- Uses `SODMetrics` (MAE, S, E, F via py_sod_metrics)
- Geometry reversal: unpad + resize to original resolution
- Saves best model when MAE improves

### Signal Handling (`train_ddp.py:146-153`)

SIGINT and SIGTERM handled gracefully:
```python
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)
```
Sets `global_stop_flag[0] = True`, training completes current step then exits cleanly.

## Loss Computation (`src/loss.py:62-219`)

### Saliency Losses
- **BCE:** `nn.BCEWithLogitsLoss()` — per-pixel binary cross-entropy
- **IoU:** `1 - (intersection + eps) / (union + eps)` computed per-image, averaged over batch
- **SSIM:** Custom implementation (`loss.py:9-57`), window_size=11, Gaussian window
- **Boundary:** Gradient magnitude of predictions vs GT boundary, `SmoothL1Loss`

### MoE Auxiliary Losses (`loss.py:109-152`)

**Load balancing loss** (Shazeer-style):
```python
f_j = bincount(flat_indices) / total_assignments  # Hard fraction (detached)
P_j = scatter_add(flat_gates) / (B * N_tokens)    # Soft mass (differentiable)
l_lb = E * sum(f_j * P_j)                          # Product form
```

**Importance loss:**
```python
mean_imp = P_j_sum.mean()
std_imp = P_j_sum.std(unbiased=False)
l_imp = (std_imp / (mean_imp + 1e-6))**2           # Coefficient of variation squared
```

**Z-loss (optional):**
```python
l_z = mean(logsumexp(clean_logits, dim=1)**2)
```

### Deep Supervision (`loss.py:183-195`)

BCE loss on each auxiliary head, target resized via nearest-neighbor interpolation:
```python
l_deep_supervision = (l_ds_16 + l_ds_8 + l_ds_4) / 3.0
```

### Total Loss
```python
L_total = L_bce + λ_iou*L_iou + λ_ssim*L_ssim + λ_boundary*L_boundary
        + λ_lb*L_lb + λ_importance*L_imp + λ_z*L_z
        + λ_aux_boundary*L_aux_boundary + λ_deep_supervision*L_deep_supervision
```

**Default weights (baseline_v1.json):**
| Weight | Value |
|--------|-------|
| λ_bce | 1.0 |
| λ_iou | 1.0 |
| λ_ssim | 0.0 |
| λ_boundary | 0.0 |
| λ_lb | 0.01 |
| λ_importance | 0.01 |
| λ_z | 0.0 |
| λ_aux_boundary | 0.0 |
| λ_deep_supervision | 0.4 |

## Smoke Test Suite (`src/smoke_test.py`)

### Modes
- `smoke1gpu`: Single-GPU forward + loss + gradient + optimizer checks
- `ddp`: 2-GPU DDP validation
- `memory`: Memory calibration (150 steps, check < 15GB)
- `resume_a`: Write checkpoint
- `resume_b`: Load checkpoint, verify behavioral resume

### Checks Performed
1. **Sparsity:** Every token dispatched exactly K times, no dense expert execution
2. **Loss:** All loss terms finite, perfect prediction loss < random prediction loss
3. **Gradients:** At least one parameter has non-zero gradient, deep supervision heads get gradients
4. **Optimizer:** Frozen params unchanged, trainable params updated
5. **Resume:** Parameter/loss/scaler state matches after interrupted+resumed run
