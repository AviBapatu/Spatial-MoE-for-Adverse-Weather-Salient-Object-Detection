# Training

Verified against `src/training/`, `src/optimization.py` and `src/config.py`.

## Entrypoint

```bash
torchrun --nproc_per_node=2 -m src.train_ddp --config experiments/<config>.json
torchrun --nproc_per_node=2 -m src.train_ddp --config <cfg> --resume latest
torchrun --nproc_per_node=2 -m src.train_ddp --config <cfg> --preflight --max_optimizer_steps 100000
```

`src/train_ddp.py` is a thin entry point; the loop, setup, checkpointing and DDP helpers
live under `src/training/`. Two ranks are required except under `--preflight` or
`--dry_run`.

## Configuration

`ExperimentConfig` (`src/config.py`) is the single source of truth for hyperparameters.
Defaults in constructors are not authoritative — `SpatialMoESODNet`, for instance, defaults
to `num_experts=6` while every config passes its own value. `build_model` asserts that the
built model matches the config (expert count, top-k, gate mode, MoE type) so a field that
is validated but never forwarded fails at construction instead of training a different
model in silence.

## Experiment identity

The ID is `EXP_{backbone}_E{experts}_K{topk}_S{effective_global_batch}_{router}_{loss}_{moe}`
with `_{variant}` appended when set. It is a **pure function of the config**, derived on
every rank by `resolve_workspace` before the rank-0-only setup runs — so both ranks agree on
the output directory. `run_id` carries a timestamp and a uuid and genuinely comes from rank
0 over a broadcast.

An ID identifies a run's checkpoints locally and on the Hub, so two different recipes must
never share one. That is what `variant` is for.

## Distributed invariants

- `find_unused_parameters=True`. Which modules participate in the forward pass depends on
  the config (disabling router noise leaves `RouterNoise.noise_linear` unused), so the flag
  is not optional. The cost is one extra autograd traversal per step.
- Every expert is called on every rank every step, even when it receives no tokens, or DDP
  gradient reduction desyncs across ranks with different usage.
- `dist.init_process_group` is always given an explicit device and timeout; never rely on
  the default NCCL timeout as the safety margin.
- Work gated on `rank == 0` (checkpointing, logging, Hub pushes) is kept short, and
  validation and diagnostics are computed on both ranks so no rank idles into a timeout.

## Optimisation

| Element | Value |
|---|---|
| Optimizer | AdamW, parameter groups split by module family and by decay / no-decay |
| Learning rate | `backbone_lr` and `new_module_lr`, both 1e-4 in the current configs |
| Weight decay | 1e-4 on decay groups |
| Schedule | linear warmup then cosine decay to 1% of peak (`WarmupCosineScheduler`) |
| Warmup | `warmup_ratio` of total steps — 0.04 in the current configs |
| Gradient clipping | max norm 1.0, applied before the optimizer step |
| Precision | AMP fp16 with a `GradScaler`; the scaler adapts on overflow |
| Gradient accumulation | `grad_accum_steps`, with `no_sync` on non-final microsteps |

Total steps are `len(train_loader) // grad_accum_steps * epochs`, so warmup and the cosine
span the whole run.

## Backbone freezing

`freeze_backbone` / `unfreeze_backbone` in `src/optimization.py` are deliberate no-ops. The
freeze is implemented by nulling the backbone's gradients after backward, for
`train.freeze_backbone_epochs` epochs (1 in the current configs). The pretrained backbone
therefore starts receiving 1e-4 updates at the second epoch — the transition to watch if
you suspect the learning rate.

## Checkpointing

- `latest.pth` is written every `train.checkpoint_every_n_steps` optimizer steps (200), and
  **only** when a step falls on that boundary. A very short run writes nothing.
- `best.pth` is selected by validation MAE.
- Checkpoints carry the config hash, and resuming validates it. The hash deliberately
  excludes `experiment_id`, `run_id`, `batch_equivalence`, `variant`, `data.dataset_root`,
  `train.num_workers` and `train.checkpoint_every_n_steps`.
- **The hash does include `train.epochs`**, and `--max_epochs` is applied before hashing, so
  resuming with a different epoch budget is refused by design — the budget changes the LR
  schedule.
- Managed-workspace guards refuse to start a config whose checkpoints already exist unless
  `--overwrite` or `--resume` is given. The training notebooks therefore pass `--overwrite`
  only when their `ALLOW_OVERWRITE` flag is set, so an accidental re-run stops instead of
  destroying finished work.

## Pre-flight run (`--preflight`)

Redirects outputs to a separate root, disables Hub pushes, permits a single rank, and
defaults `--max_optimizer_steps` to 5. The notebook's pre-run check uses it to train one
real epoch over 100 images, exercising the epoch-boundary paths (validation, routing
diagnostics, checkpoint write) that a normal run would only reach hours in. It does **not**
test resume: a resume cannot be exercised through a config whose hash covers `train.epochs`.
That round trip belongs to `src.smoke_test --mode resume_a / resume_b`.

## Logging

Progress lines report step, batch, loss and — since the current code — the distinct learning
rates among the optimizer's parameter groups and the AMP scaler's scale. A scaler that keeps
halving means the learning rate is too high for fp16. Logging is written so that a failure
to read a value degrades to `?` rather than raising: a log line must never kill a run.

## Kaggle workflow

The notebook pulls the code zip and manifests from the Hub, verifies the archive hash,
unpacks into the working directory, and runs `torchrun` from there. Checkpoints and
diagnostics are pushed back under their experiment ID, which is what makes a wiped session
recoverable: the resume path pulls `{experiment_id}_latest.pth` back down.

Evaluate one config per session. The evaluation output paths are fixed within a session, so
a second config evaluated without clearing them would upload the first config's results
under its own label.
