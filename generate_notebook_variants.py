"""Generate the seed-repeat and E4 notebooks from the same cell source.

Each notebook differs from the one it is copied from in exactly one cell: the
ACTIVE_CONFIG_PATH assignment. Rather than duplicating the whole generator per
variant, accountA's builder is parameterized on (notebook, config path), so the
arm's cells stay in one place and cannot drift.

Usage:
    python generate_notebook_variants.py
"""
from generate_notebook_accountA import create_notebook

VARIANTS = [
    ("moe-of-sod__v_e8_repro_best_seed43.ipynb", "experiments/v_e8_repro_best_seed43.json"),
    ("moe-of-sod__v_e8_repro_best_seed44.ipynb", "experiments/v_e8_repro_best_seed44.json"),
    ("moe-of-sod__v_e4_repro_renorm.ipynb", "experiments/v_e4_repro_renorm.json"),
]


def main() -> None:
    """Write one notebook per variant."""
    for notebook, config in VARIANTS:
        create_notebook(notebook, config)


if __name__ == "__main__":
    main()
