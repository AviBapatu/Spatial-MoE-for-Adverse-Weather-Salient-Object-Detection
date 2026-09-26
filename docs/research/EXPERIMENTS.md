# Experiments

## Configuration

Every run is defined by one JSON file under `experiments/`, loaded into
`ExperimentConfig` (`src/config.py`). The config is the single source of truth: the model is
asserted to match it at construction, and a config hash is written into every checkpoint so
a resume can prove it is the same experiment (checkpoints carry `get_config_hash`, an MD5;
the canonical SHA-256 is a separate value written to the run directory — see `TRAINING.md`).

Fields that carry weight: `data` (root, `max_samples`), `model` (backbone — see the caveat
below — `num_experts`, `top_k`, `gate_mode`, `moe_type`, `moe_16_mode`, router noise),
`loss` (weights per term, `load_balance_weights`), `opt` (learning rates, batch per GPU,
accumulation, AMP, clipping), `train` (epochs, seed, workers, checkpoint cadence), `eval`,
`diag`, and `variant`.

The canonical hash excludes fields that cannot change the mathematics: `experiment_id`,
`run_id`, `batch_equivalence`, `variant`, `data.dataset_root`, `train.num_workers`,
`train.checkpoint_every_n_steps`. It **includes** `train.epochs`.

**The preset is not the recipe.** `experiments/baseline_v1.json` and `src/presets/baseline.json`
are templates: they carry `ssim_weight: 0.0`, `boundary_weight: 0.0` and
`importance_weight: 0.01`. The configs behind actual runs are the `v_e8_*` and `v_e4_*` files,
which use `ssim_weight: 1.0`, `boundary_weight: 1.0` and `importance_weight: 0.05`. A loss
table quoted from the preset will not match a trained run.

## Experiment IDs

```
EXP_{backbone}_E{experts}_K{topk}_S{effective_global_batch}_{router}_{loss}_{moe}[_{variant}]
```

The ID names every artifact for a run: the output directory, the Hub checkpoint names, the
diagnostics and the uploaded analysis folder. It is a pure function of the config and is
derived on every rank.

**Rule: a different recipe needs a different `variant`.** Six configs previously derived one
identical ID, which would have made any two of them overwrite each other's checkpoints.
All 28 configs in `experiments/` now derive unique IDs, verified with
`src.experiment.generate_experiment_id`.

**Do not confuse `S` with the seed.** `S` is the effective global batch
(`batch_per_gpu x grad_accum_steps x world_size`). Seed repeats are distinguished by
`variant` (`REPRO_SEED43`, `REPRO_SEED44`), not by `S`.

## The current experiment set

Two arms isolate one variable each from the historical best recipe (8 experts, k=2,
renormalized gate, router noise on, load-balance weights 0.08/0.15/0.1, effective batch 32,
8 epochs, seed 42).

| Arm | Config | ID | Question |
|---|---|---|---|
| A | `v_e8_repro_best.json` | `..._E8_K2_S32_..._REPRO` | reproduce the historical best under current code |
| B | `v_e8_repro_gatedense.json` | `..._REPRODENSE` | does the dense gate (gradient to every expert) change behaviour or accuracy? |
| seed 43 | `v_e8_repro_best_seed43.json` | `..._REPRO_SEED43` | variance estimate |
| seed 44 | `v_e8_repro_best_seed44.json` | `..._REPRO_SEED44` | variance estimate |
| E4 | `v_e4_repro_renorm.json` | `EXP_B4_E4_K2_S32_...` | does 4 experts reach 8, all else fixed? |

Later single-variable arms at the same E4/E8 S32 recipe add: a 14-epoch budget
(`v_e8_repro_best_14ep.json`, `REPRO_E14`), routing restricted to the finest scale
(`v_e8_repro_scale1.json`, `SCALE1`, `moe_type: sparse_fine`), the mixture-vs-shared-MLP
control (`v_e4_repro_nonectrl.json`, `E4_NONECTRL`), a dense-gate control at E4
(`v_e4_repro_densectrl.json`, `E4_DENSECTRL`), a zero-load-balance arm
(`v_e4_repro_nolb.json`, `NOLB`), the expert-count and top-k axes
(`v_e2_k2_s32.json`, `v_e4_k1_s32.json`) and an entropy-confidence-weight arm
(`v_4expert_entropyconf.json`, `ENTROPYCONF`, which sets
`loss.entropy_confidence_weight: 0.2`).

Three seeds of A give the noise floor; without it, no difference between arms is
interpretable. B tests the mechanism behind dead experts: the renormalized gate gives
gradient only to selected experts, the dense gate gives it to all.

## Historical runs

| Run | Recipe | Status |
|---|---|---|
| E2 K1 GATEDENSE | dense gate, noise off, load-balance 0.016/0.03/0.02, effective batch 40 | trained and evaluated |
| E4 K2 GATEDENSE | same recipe, 4 experts | trained and evaluated |
| E8 legacy | renormalized, noise on, effective batch 32 | the 0.0168 model; evaluated |
| E8 `ablation_full_moe`, `ablation_moe16_dense` | earlier ablation arms | trained, partial diagnostics |

The E2/E4 GATEDENSE runs differ from the E8 runs in five respects at once, so they cannot
support an expert-count comparison. The E4 arm above exists because of that.

Numbers for every evaluated run: `RESULTS.md`.

## Protocols

- **Pre-run check.** The training notebooks run `--preflight` first: an architecture probe
  plus one 100-image epoch covering training, validation, routing diagnostics and a
  checkpoint write. It is not a resume test.
- **Overwrite protection.** Notebooks pass `--overwrite` only when `ALLOW_OVERWRITE` is set.
  With it unset, pointing a notebook at an already-trained config stops with the trainer's
  error rather than destroying the previous run.
- **Evaluation.** One config per session; the evaluation output paths are fixed within a
  session, and the notebook refuses to proceed if they already hold another config's
  results. Flip-average TTA (`EVAL_TTA = "hflip"`) is the default for new runs, and hflip
  results are written under their own subfolder so they cannot overwrite the plain ones.
  Every model in a comparison must use the same TTA setting.
- **Artifacts.** Checkpoints are pushed as `{experiment_id}_best.pth` and
  `{experiment_id}_latest.pth`, which is what makes a wiped Kaggle session recoverable.

## What has not been done

| Gap | Why it matters |
|---|---|
| Mixture-vs-dense control (`v_e4_repro_nonectrl.json`, `v_e4_repro_densectrl.json`, and the older `v_4expert_nonecontrol.json`, `v_4expert_densecontrol.json`) | no `M_DENSE`/`M_NONE` run has been evaluated under `results/`; without it the mixture's contribution is unmeasured |
| External baselines | no comparison to prior methods exists |
| Seed repeats beyond arm A | every other number is a single run |
| Backbone comparison | `config.model.backbone` does not reach the model, so backbone arms train identical architectures |
| `router_variant` comparison | the field affects only the experiment ID |
