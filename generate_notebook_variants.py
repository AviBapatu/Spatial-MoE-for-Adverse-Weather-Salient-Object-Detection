"""Generate the seed-repeat and E4 notebooks from the same cell source.

Each notebook differs from the one it is copied from in one or two lines: the
ACTIVE_CONFIG_PATH assignment, and RUN_MODE when the notebook is evaluation-only.
Rather than duplicating the whole generator per variant, accountA's builder is
parameterized on (notebook, config path, run mode), so the arm's cells stay in one
place and cannot drift.

Usage:
    python generate_notebook_variants.py
"""
from generate_notebook_accountA import create_notebook

VARIANTS = [
    ("moe-of-sod__v_e8_repro_best_seed43.ipynb", "experiments/v_e8_repro_best_seed43.json", "TRAIN"),
    ("moe-of-sod__v_e8_repro_best_seed44.ipynb", "experiments/v_e8_repro_best_seed44.json", "TRAIN"),
    # The flagship: the same recipe and seed as REPRO, trained for 14 epochs instead of 8.
    # The benchmark's "best" model was a 14-epoch run, and on the real split the budget is
    # the difference between mid-table and second place.  Nothing else changes, so the pair
    # is a clean single-variable comparison of training budget.
    ("moe-of-sod__v_e8_repro_best_14ep.ipynb", "experiments/v_e8_repro_best_14ep.json", "TRAIN"),
    # Multi-scale-routing ablation: routed experts at 1/4 only, pass-through at 1/8
    # and 1/16.  Without it the three-scale routing claim has no arm behind it.
    ("moe-of-sod__v_e8_repro_scale1.ipynb", "experiments/v_e8_repro_scale1.json", "TRAIN"),
    # Clean S32 arms for the two axes that existed only in the S40 generation, whose
    # arms differ in batch size, loss weights AND router noise, so they cannot isolate
    # expert count or top-k. Cloned from the S32 baseline so only the axis changes.
    ("moe-of-sod__v_e2_k2_s32.ipynb", "experiments/v_e2_k2_s32.json", "TRAIN"),
    ("moe-of-sod__v_e4_k1_s32.ipynb", "experiments/v_e4_k1_s32.json", "TRAIN"),
    # Load-balance ablation: the loss term that forces uniform expert usage is removed
    # entirely.  If experts still do not specialise with no balancing pressure at all,
    # the cause is the saliency objective, not the regulariser - which is the mechanistic
    # question this project's routing diagnostics raise.
    ("moe-of-sod__v_e4_repro_nolb.ipynb", "experiments/v_e4_repro_nolb.json", "TRAIN"),
    ("moe-of-sod__v_e4_repro_renorm.ipynb", "experiments/v_e4_repro_renorm.json", "TRAIN"),
    ("moe-of-sod__v_e4_repro_densectrl.ipynb", "experiments/v_e4_repro_densectrl.json", "TRAIN"),
    ("moe-of-sod__v_e4_repro_nonectrl.ipynb", "experiments/v_e4_repro_nonectrl.json", "TRAIN"),
    # Evaluation-only.  These runs are already trained and their checkpoints are on the
    # Hub; only the metrics are missing.  The two load-balance arms were never evaluated,
    # and the two GATEDENSE arms need re-scoring under hflip so every row in the table
    # shares one test-time protocol.
    ("moe-of-sod__v_e2_loadbal.ipynb", "experiments/v_2expert_loadbal.json", "EVALUATE"),
    ("moe-of-sod__v_e4_loadbal.ipynb", "experiments/v_4expert_loadbal.json", "EVALUATE"),
    ("moe-of-sod__v_e2_gatedense.ipynb", "experiments/v_2expert_gatedense.json", "EVALUATE"),
    ("moe-of-sod__v_e4_gatedense.ipynb", "experiments/v_4expert_gatedense.json", "EVALUATE"),
]


def main() -> None:
    """Write one notebook per variant."""
    for notebook, config, mode in VARIANTS:
        create_notebook(notebook, config, mode)


if __name__ == "__main__":
    main()
