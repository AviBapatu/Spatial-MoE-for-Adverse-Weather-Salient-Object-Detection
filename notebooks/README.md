# Notebooks

Every run is driven by a notebook. Each one is **generated** — edit the generator, never the
`.ipynb` — so that the cells stay identical across arms and cannot drift apart.

```
notebooks/
  train/       one notebook per training run; each trains, then evaluates itself
  evaluate/    one notebook per evaluation-only run; pulls its checkpoint from the Hub
  generators/  the source of truth for every notebook above
```

## train/

| notebook | experiment id | what it adds |
|---|---|---|
| `moe-of-sod__accountA__v_e8_repro_best.ipynb` | `..._E8_K2_S32_..._REPRO` | the reference arm |
| `moe-of-sod__accountB__v_e8_repro_gatedense.ipynb` | `..._E8_K2_S32_..._REPRODENSE` | dense gate |
| `moe-of-sod__v_e8_repro_best_seed43.ipynb` | `..._REPRO_SEED43` | seed repeat |
| `moe-of-sod__v_e8_repro_best_seed44.ipynb` | `..._REPRO_SEED44` | seed repeat |
| `moe-of-sod__v_e8_repro_best_14ep.ipynb` | `..._REPRO_E14` | **the flagship** — same recipe and seed, 14 epochs |
| `moe-of-sod__v_e2_k2_s32.ipynb` | `..._E2_K2_S32_...` | expert count, lower point |
| `moe-of-sod__v_e4_k1_s32.ipynb` | `..._E4_K1_S32_...` | top-k, lower point |
| `moe-of-sod__v_e4_repro_nolb.ipynb` | `..._E4_K2_S32_..._NOLB` | load-balance weight set to zero |
| `moe-of-sod__v_e4_repro_renorm.ipynb` | `..._E4_K2_S32_...` | the S32 baseline (never run) |

## evaluate/

| notebook | experiment id | what it adds |
|---|---|---|
| `moe-of-sod__v_e8_repro_scale1.ipynb` | `..._E8_K2_S32_..._SCALE1` | routing at 1/4 only |
| `moe-of-sod__v_e4_repro_densectrl.ipynb` | `..._E4_K2_S32_..._DENSE_E4_DENSECTRL` | one shared expert per scale |
| `moe-of-sod__v_e4_repro_nonectrl.ipynb` | `..._E4_K2_S32_..._NONE_E4_NONECTRL` | **no MoE at all** — the control behind the paper's main claim |
| `moe-of-sod__v_e2_loadbal.ipynb` | `..._E2_K1_S40_..._SPARSE` | S40 load-balance arm, not attributable |
| `moe-of-sod__v_e4_loadbal.ipynb` | `..._E4_K2_S40_..._SPARSE` | S40 load-balance arm, not attributable |
| `moe-of-sod__v_e2_gatedense.ipynb` | `..._E2_K1_S40_..._GATEDENSE` | S40 gate-mode arm |
| `moe-of-sod__v_e4_gatedense.ipynb` | `..._E4_K2_S40_..._GATEDENSE` | S40 gate-mode arm |
| `moe-of-sod__legacy_eval.ipynb` | *(architecture in the checkpoint)* | certifies the reported model's 0.0168 / 0.0192 |

## Regenerating

From the repository root:

```bash
python notebooks/generators/generate_notebook_variants.py    # everything except the two account notebooks and the legacy evaluation
python notebooks/generators/generate_notebook_accountA.py
python notebooks/generators/generate_notebook_accountB.py
python notebooks/generators/generate_notebook_legacy_eval.py
```

Regenerating with no source changes must leave the working tree clean — if `git status` shows
a modified notebook, the generator and the committed notebook had drifted apart.
`tests/test_notebook_generators.py` checks the same thing.

## Running one on Kaggle

Each notebook downloads the code zip and its checkpoint from Hugging Face, so it is
self-contained. Set `RUN_MODE` at the top of the first cell:

- `TRAIN` — train from scratch, then evaluate. Refuses to start if the run's checkpoints
  already exist, so a finished run cannot be silently overwritten; set `ALLOW_OVERWRITE = True`
  to redo one on purpose.
- `EVALUATE` — pull this run's checkpoint and score it, no training.
- `RESUME` — continue an interrupted run from its latest checkpoint.
