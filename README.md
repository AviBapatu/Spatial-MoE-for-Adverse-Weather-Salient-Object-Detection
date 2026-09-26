# Spatial-MoE for Adverse-Weather Salient Object Detection

Spatially Dynamic Mixture-of-Experts (MoE) for robust Salient Object Detection (SOD) under fog, rain, snow, and low-light conditions. Individual spatial tokens are routed to specialized expert networks without any weather label at train or test time.

## Repository Layout

```
src/
  model.py                # Top-level SpatialMoESODNet
  backbone.py             # PVTv2-B4 multi-scale feature extractor
  moe_layer.py            # Router + expert pool + sparse dispatch
  decoder/                # Cross-attention fusion + prediction heads
  loss.py                 # BCE + IoU + load-balance + importance + deep supervision
  metrics.py              # MAE / F-beta / S-score evaluation
  dataset.py              # WXSOD dataset loading + augmentation
  config.py               # ExperimentConfig dataclass
  experiment.py           # Run-directory + registry management
  optimization.py         # WarmupCosine + backbone freeze/unfreeze
  log.py                  # Rank-aware logging (centralized)
  training/
    cli.py                # Argument parsing + main loop
    setup.py              # Model/optimizer/dataloader construction + resume
    loop.py               # Per-epoch validation + diagnostics
    epoch.py              # Per-epoch training batches
    checkpoint.py         # Save/load/resume + HF enqueue
    distributed.py        # Process-group lifecycle + monitored barrier
  diagnostics/            # Routing stats, expert similarity, weather analysis
  evaluate.py             # Batch evaluation + prediction export
  hf_sync/                # HF Hub sync (code + checkpoint push/pull)
experiments/              # Config JSONs (baseline_v1.json, smoke_test_config.json, ...)
tests/                    # Unit + integration tests
docs/research/            # Persistent research knowledge base
data/WXSDO_data/          # WXSOD dataset (train_sys, test_sys, test_real splits)
checkpoints/              # Model weight files (.pth)
results/                  # Evaluation results, routing statistics, legacy artifacts
```

## Environment Setup

Requires Python 3.12+ and a CUDA-capable GPU.

```bash
# Install dependencies (pinned versions)
uv sync

# Or with pip
pip install -r requirements.txt
```

The project uses [uv](https://docs.astral.sh/uv/) as the package manager. All dependency versions are pinned in `pyproject.toml`.

## Running Training

### Direct training (2+ GPUs via DDP)

```bash
# Standard training
uv run torchrun --nproc_per_node=2 -m src.train_ddp --config experiments/baseline_v1.json

# Resume from latest checkpoint
uv run torchrun --nproc_per_node=2 -m src.train_ddp --config experiments/baseline_v1.json --resume latest

# Resume from specific checkpoint
uv run torchrun --nproc_per_node=2 -m src.train_ddp --config experiments/baseline_v1.json --resume checkpoints/best.pth

# Preflight smoke test (5 optimizer steps, no HF upload)
uv run torchrun --nproc_per_node=2 -m src.train_ddp --config experiments/baseline_v1.json --preflight

# Dry run (validates config, no training)
uv run torchrun --nproc_per_node=2 -m src.train_ddp --config experiments/baseline_v1.json --dry_run
```

### Notebook training (Kaggle)

The notebooks pull the code zip from Hugging Face and verify its hash, so repackage with
`python package_project.py` after changing anything under `src/` or `experiments/`.

Open one of the generated notebooks: `moe-of-sod__accountA__v_e8_repro_best.ipynb` or
`moe-of-sod__accountB__v_e8_repro_gatedense.ipynb`, and the `moe-of-sod__v_*.ipynb`
variants behind them. They are generated from the `generate_notebook_*.py` scripts, which
are the editable source. Each notebook uses `RUN_MODE` to control behaviour:

| `RUN_MODE` | Description |
|------------|-------------|
| `TRAIN`    | Train from scratch. Resumes from HF checkpoint only if checkpoint is present AND mode is `RESUME`. |
| `RESUME`   | Resume training from the last HF checkpoint. |
| `VALIDATE` | Run validation only (no training). |
| `EVALUATE` | Run evaluation on test splits. |

## CLI Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | Experiment config JSON (default: `experiments/baseline_v1.json`) |
| `--resume [latest\|PATH]` | Resume from a checkpoint |
| `--preflight` | Smoke test: 5 optimizer steps, validates loss/DDP/checkpoint |
| `--dry_run` | Validate config only, exit immediately |
| `--overwrite` | Bypass "training already complete" safety check |
| `--max_optimizer_steps N` | Stop after N optimizer steps |
| `--max_epochs N` | Cap maximum training epochs |
| `--smoke_test` | Legacy flag; inert (run `uv run python -m src.smoke_test` instead) |
| `--calibration` | Memory calibration mode |

## Experiment Configs

Configs live in `experiments/`. `baseline_v1.json` is the preset template; the configs
behind published runs are the `v_e8_*` and `v_e4_*` files.

Each config derives an **experiment ID** from its own fields — backbone, expert count,
top-k, effective batch, router and loss variants, MoE type, plus an optional `variant`.
That ID names the output directory and every artifact on the Hub, so **a different
recipe needs a different `variant`**, or two runs will overwrite each other's
checkpoints. `docs/research/EXPERIMENTS.md` has the rule and the current run set.
```bash
# Generate ablation configs
uv run python -m src.ablations

# List all registered experiments
cat experiments/registry.csv
```

## Evaluation

```bash
uv run python -m src.evaluate --checkpoint checkpoints/best.pth --dataset both
```

Results go under `results/<experiment_id>/eval_results/`, with a dataset and TTA subfolder.
Flip-average TTA is on by default for new runs and keeps its results separate.

## Testing

```bash
# Full test suite
uv run pytest tests/ -v

# Smoke tests (pre-flight)
uv run python -m src.smoke_test --mode smoke1gpu --data_root data/WXSDO_data --result_file /tmp/r.json
uv run torchrun --nproc_per_node=2 -m src.smoke_test --mode ddp --data_root data/WXSDO_data --result_file /tmp/r.json
```

## Linting

```bash
uv run ruff check src/
```

## DDP Safety Rules

This codebase enforces specific rules for multi-GPU training safety:

1. **Every DataLoader must be rank-aware.** `DistributedSampler` is required on train, validation, and test loaders. An earlier version had a bug where val/test loaders silently lacked samplers, causing rank-0 idling.

2. **`dist.init_process_group` always uses explicit `timeout=timedelta(...)`.** The NCCL default of 600s is too long; we use 45 minutes as a hard ceiling.

3. **Rank-0-only blocks must be short, or followed by `dist.monitored_barrier()`.** If rank 0 gets stuck doing I/O while other ranks wait at a collective, the hang is silent until an NCCL timeout. `monitored_barrier` catches this immediately with a clear error message.

4. **Checkpoint I/O never synchronously blocks the hot path.** `torch.save` / `torch.load` verification happens off the critical path (background HF-upload thread or separate step).

5. **Every expert module must be touched by every rank on every step.** DDP's gradient sync will desync across ranks with different expert-usage patterns otherwise. The sparse-dispatch pattern in `moe_layer.py` handles this correctly.

## Troubleshooting

### NCCL timeout / hang during training

This is almost always caused by a rank-0-only block (logging, checkpoint save, HF upload) taking too long while other ranks wait at a collective. The fix: ensure any rank-0-only work is either very fast (< a few seconds) or immediately followed by `dist.monitored_barrier(timeout_minutes=N)` (see `src/training/distributed.py`).

The `init_process_group` uses a 45-minute timeout ceiling as a safety net, but the real fix is making rank-0-only blocks provably short.

### CUDA OOM

The current configs run `batch_per_gpu 4` with `grad_accum_steps 4` on two T4s (effective
batch 32). Reduce `batch_per_gpu` first if a config does not fit.

### Config hash mismatch on resume

This means the model architecture or critical hyperparameters changed between training sessions. Use `--overwrite` to start fresh, or resume from the exact same config.

## Documentation

Two documents are canonical: `RESEARCH_TRUTH.md` (what may be claimed, and what is
forbidden) and `docs/research/RESULTS.md` (every measured number, each traceable to a
result file). `docs/research/INDEX.md` maps the rest.

## License

Internal research project.
