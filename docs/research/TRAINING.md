# Training

> **Note on citations.** Line-number citations below predate the current source
> layout (the decoder is now the `src/decoder/` package, the training loop lives under
> `src/training/`, the evaluation artifacts are under `results/`). Treat module and
> function names as authoritative and `RESEARCH_TRUTH.md` as the verified reference;
> numbers live in `RESULTS.md`, which is generated from the result files.

Verified against `src/`. Entrypoints and flags below are the ones the Kaggle notebooks
actually call. Where something is a design decision worth defending in a paper, it says why.

## Entry points

| Command | Purpose |
|---|---|
| `torchrun --nproc_per_node=2 -m src.train_ddp --config <cfg>` | training (2 ranks, NCCL) |
| `python -m src.evaluate --checkpoint <ckpt> --dataset both` | evaluation (single process) |
| `torchrun --nproc_per_node=2 -m src.smoke_test --mode <mode>` | pre-flight checks |
| `python package_project.py` | build + upload the code zip the notebooks unpack |

`train.py` at the repository root is the legacy single-GPU entrypoint and is not maintained;
use `src.train_ddp`.

The trainer accepts `--resume <path|latest>`, `--preflight`, `--max_epochs`,
`--max_optimizer_steps`, `--overwrite` and `--dry-run`.

## Startup — `src/training/setup.py::init_training`

1. Load and validate the config (`ExperimentConfig.load` + `validate`, which rejects
   `top_k > num_experts`, an unknown `moe_type`, and similar).
2. Confirm `config.data.dataset_root` exists.
3. Require `world_size >= 2` unless `--preflight` or `--dry-run`.
4. Resolve the workspace: the preflight root in preflight mode, otherwise the checkpoint
   root, each with the experiment ID beneath it.
5. Build rank-aware dataloaders (a `DistributedSampler` per rank — every loader reachable
   from the entrypoint, validation included).
6. Build the model, then verify it against the config (see *Invariants*).
7. Build optimizer, scheduler and criterion, then apply `--resume`.

### Experiment ID and run directory

`config.experiment_id` is a pure function of the config —
`EXP_{backbone}_E{experts}_K{topk}_S{effective_batch}_{router}_{loss}_{moe}[_{variant}]` —
so every rank derives the same value locally, and the two ranks always agree on
`base_dir`. Anything that changes the recipe must get a distinct `variant`, or two runs
share one ID and overwrite each other's checkpoints on the Hub.

`config.run_id` carries a timestamp and a uuid, so only rank 0 can produce it; it is
broadcast to the other ranks.

## Optimization

**Optimizer:** AdamW over eight parameter groups — backbone / moe / decoder / heads, each
split into decay and no-decay (`src/optimization.py::get_parameter_groups`). Biases,
normalization parameters and relative position bias tables go in the no-decay groups.
`backbone_lr` and `new_module_lr` are separate config fields; the reproduction configs set
both to 1e-4.

**Scheduler:** `WarmupCosineScheduler` — linear warmup for `warmup_ratio` of the run, then
cosine decay to `min_lr_ratio = 0.01` of peak. `min_lr_ratio` is a constructor default, not
a config field. Total steps are `len(train_loader) // grad_accum_steps * epochs`, so with the
reproduction config (1292 batches, accum 4, 8 epochs) that is ~2,584 steps and ~103 warmup
steps. Scheduler state is saved in the checkpoint, so a resume continues the same curve.

**Precision:** fp16 autocast with a `GradScaler`. On an overflow the step is skipped and the
scale is halved; the epoch counts overflows and logs only the first. A scale that keeps
halving indicates the LR is too high for fp16.

**Clipping:** `max_norm = 1.0`.

**Effective batch:** `batch_per_gpu x world_size x grad_accum_steps`. Non-final micro-steps
run under `model.no_sync()` so gradients are synchronised once per optimizer step.

**Backbone freeze:** `train.freeze_backbone_epochs` is implemented by **nulling the backbone
gradients after backward**, not by setting `requires_grad`. The backbone still runs forward
and DDP still synchronises its parameters; it simply receives no update. `freeze_backbone()`
and `unfreeze_backbone()` in `src/optimization.py` are deliberate no-ops and should not be
relied on. The reproduction configs freeze the backbone for epoch 1, which matters when
reading the loss curve: the backbone first receives updates at epoch 2.

## Loss — `src/loss.py`

Total loss is a weighted sum of per-pixel BCE, IoU, SSIM and boundary terms, plus the MoE
auxiliary terms, plus optional deep supervision on the auxiliary heads.

MoE auxiliaries:

- **Load balance:** the product of the hard assignment fraction and the soft gate mass,
  summed over experts and scaled by the expert count. The hard fraction is detached, so this
  term shapes the router only through the soft mass.
- **Importance:** squared coefficient of variation of the per-expert gate mass, which pushes
  towards equal total gate mass.
- **Z-loss (optional):** mean squared log-sum-exp of the routing logits.
- **Entropy confidence (optional, `entropy_confidence_weight`):** a penalty on routing
  entropy, i.e. a pressure towards confident routing.

The load-balance optimum is exactly uniform usage, which matters when interpreting the
routing statistics in `RESULTS.md`: uniform usage is what this term is designed to produce,
not evidence of specialisation.

**Weights in the current reproduction recipe** (`experiments/v_e8_repro_best.json` and its
siblings): bce 1.0, iou 1.0, ssim 1.0, boundary 1.0, importance 0.05, z 0.001,
aux_boundary 0.5, deep_supervision 0.4, load-balance 0.08 / 0.15 / 0.1 per scale, entropy
confidence 0.0. The older shipped preset `experiments/baseline_v1.json` uses different
weights — read the file rather than assuming.

## Per-epoch loop — `src/training/epoch.py`, `src/training/loop.py`

```
for epoch in range(start_epoch, config.train.epochs):
    model.train()
    sampler.set_epoch(epoch)                 # and the dataset's augmentation seed
    for each micro-batch:
        forward under autocast, compute loss, scale, backward
        on the final micro-step: unscale -> clip -> step -> update scaler
    validation, then routing diagnostics, then checkpointing
```

Training prints a progress line every 50 batches, now including the learning rate and the
current AMP scale:

```
[Train] Ep 1 | 50/1292 | Loss 3.1020 | Step 11 | lr 1.00e-04 | amp scale 65536
```

The reported Loss is the last micro-batch of that optimizer step multiplied by the
accumulation factor, not an average over the 50 batches — expect it to be noisy and read the
trend.

## Checkpointing — `src/training/checkpoint.py`

- `latest.pth` is written every `train.checkpoint_every_n_steps` optimizer steps (200 by
  default) and **only on those steps** — the writer returns early otherwise. A run shorter
  than the cadence writes no `latest.pth` at all, which is why the pre-run check sets the
  cadence to 1 for its 100-image epoch.
- `best.pth` is written when validation MAE improves; checkpoint selection is by validation
  MAE.
- Writes are atomic (write to a temporary path, verify by reloading, then rename).
- Contents: format version, epoch, batch-in-epoch, global step, best metric, model state,
  engine state (optimizer + scheduler + scaler + step), RNG states, the config, the config
  hash, world size, timestamp.
- An async pusher uploads to the Hub on a background thread so training never blocks on the
  network. Artifacts are named `{experiment_id}_latest.pth` and `{experiment_id}_best.pth`
  in `checkpoints/` — the experiment ID is in the remote name, not the local filename. The
  preflight disables pushing entirely.

### Resume

`--resume latest` resolves to `{base_dir}/latest.pth`, where `base_dir` is this run's own
directory in both modes (preflight and production). The checkpoint is validated on format
version, world size and **config hash**.

The config hash covers the recipe but excludes the volatile fields: `experiment_id`,
`run_id`, `batch_equivalence`, `variant`, `data.dataset_root`, `data.split_manifest_path`,
`train.num_workers`, `train.checkpoint_every_n_steps`.

**Consequence worth knowing:** `train.epochs` is *not* excluded, and it is set from
`--max_epochs` before hashing. Resuming with a different epoch budget is therefore refused —
correctly, since the schedule length changes the LR curve, but it means a resume must use the
same epoch budget as the run it continues.

## Distributed invariants

These are not style preferences; each one has caused a failure here.

- **`find_unused_parameters=True` on the DDP wrap.** Which modules participate in the forward
  depends on the config — with router noise disabled, `RouterNoise.noise_linear` is never
  called — so "every expert is always called" is not sufficient to guarantee an
  unused-parameter-free graph. `False` aborts with *"Expected to have finished reduction in
  the prior iteration"* and unused-parameter indices pointing at the router noise layers.
- **Every expert is called on every step, on every rank**, even with zero routed tokens.
  Conditional expert calls would desynchronise gradient reduction across ranks that route
  differently.
- **Explicit `device_id` and an NCCL timeout** at process-group creation. Relying on the
  default timeout turns a stall into a 30-minute wait with an opaque error.
- **`base_dir` must be identical on every rank.** It is derived from `experiment_id`, which
  each rank now computes locally (see above) rather than relying on the broadcast.
- **All ranks participate in the checkpoint path**, not just rank 0: `apply_resume` loads the
  checkpoint on every rank. A rank-0-only ID made rank 1 look in the wrong directory and fail.
- **Rank-0-only work stays short.** Checkpoint writing, manifest updates and Hub pushes are
  rank-0-gated; expensive work guarded by `if rank == 0` would leave other ranks idle.

## Pre-run check

The notebooks run `## 11_preflight_checks` before training, gated by `RUN_PREFLIGHT`. It has
two stages:

1. **Architecture probe (CPU):** rebuild the model from the config and assert that the
   expert count, top-k, gate mode and mixture type actually reached it.
2. **One real epoch on 100 images:** training steps, epoch-end validation, routing
   diagnostics and a checkpoint write, with `checkpoint_every_n_steps=1` so the checkpoint
   exists. This is the part that earns its keep — those epoch-boundary paths are where an
   unattended run dies four hours in.

Artifacts go to the preflight root, never the production checkpoint directory, and no Hub
push happens. There is deliberately **no resume stage**: the config hash covers
`train.epochs`, so a resume cannot be tested through a different epoch budget. The round trip
has its own tool — `src.smoke_test --mode resume_a` then `resume_b`.

## Evaluation

`python -m src.evaluate --checkpoint <ckpt> --dataset {test_sys,test_real,both}
--tta {none,hflip} --data_dir <root> --out_dir <dir>`

- The architecture is rebuilt from the **checkpoint's own config**, so evaluation cannot
  silently run a different model: a mismatch fails the state-dict load.
- Predictions are produced in the 384x384 padded frame and then reverse-geometried (crop the
  pad, resize) into **original image coordinates** before scoring. Metrics are therefore
  computed at native resolution.
- `--tta hflip` averages the forward pass with the horizontally flipped forward pass. The
  model is trained with `HorizontalFlip(p=0.5)` after the geometry step and the pad is
  centre + reflect, so the flipped input is in-distribution. Output lands in a `<tta>/`
  subdirectory next to the plain results, so the two never overwrite each other.
- **Every model in a comparison table must use the same TTA setting**, and the setting must
  be stated wherever the numbers are reported.
