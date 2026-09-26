"""Drift guard for the Kaggle notebooks and their generators.

``generate_notebook_accountA.py`` / ``_accountB.py`` are the editable source of
truth for the two account notebooks, and ``generate_notebook_variants.py`` builds
the seed-repeat and E4 notebooks from the same cell source. Regenerating in a temp
directory must reproduce every committed notebook cell-for-cell, so a cell edited
in one place but not another fails here instead of silently diverging — which is
how these notebooks previously went stale.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

GENERATORS = [
    ("notebooks/generators/generate_notebook_accountA.py",
     ["notebooks/train/moe-of-sod__accountA__v_e8_repro_best.ipynb"]),
    ("notebooks/generators/generate_notebook_accountB.py",
     ["notebooks/train/moe-of-sod__accountB__v_e8_repro_gatedense.ipynb"]),
    ("notebooks/generators/generate_notebook_variants.py", [
        "notebooks/train/moe-of-sod__v_e8_repro_best_seed43.ipynb",
        "notebooks/train/moe-of-sod__v_e8_repro_best_seed44.ipynb",
        "notebooks/train/moe-of-sod__v_e8_repro_best_14ep.ipynb",
        "notebooks/train/moe-of-sod__v_e2_k2_s32.ipynb",
        "notebooks/train/moe-of-sod__v_e4_k1_s32.ipynb",
        "notebooks/train/moe-of-sod__v_e4_repro_nolb.ipynb",
        "notebooks/train/moe-of-sod__v_e4_repro_renorm.ipynb",
        "notebooks/evaluate/moe-of-sod__v_e8_repro_scale1.ipynb",
        "notebooks/evaluate/moe-of-sod__v_e4_repro_densectrl.ipynb",
        "notebooks/evaluate/moe-of-sod__v_e4_repro_nonectrl.ipynb",
        "notebooks/evaluate/moe-of-sod__v_e2_loadbal.ipynb",
        "notebooks/evaluate/moe-of-sod__v_e4_loadbal.ipynb",
        "notebooks/evaluate/moe-of-sod__v_e2_gatedense.ipynb",
        "notebooks/evaluate/moe-of-sod__v_e4_gatedense.ipynb",
    ]),
    ("notebooks/generators/generate_notebook_legacy_eval.py",
     ["notebooks/evaluate/moe-of-sod__legacy_eval.ipynb"]),
]


def _cell_content(cell: dict) -> str:
    """Return a cell's source as one string, whatever form the JSON stores it in."""
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


@pytest.mark.parametrize(("script", "notebooks"), GENERATORS)
def test_generator_reproduces_committed_notebook(script: str, notebooks: list,
                                                 tmp_path: Path) -> None:
    """Every notebook a generator writes must match its committed copy cell-for-cell."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    for notebook in notebooks:
        built = json.loads((REPO_ROOT / notebook).read_text())
        committed = json.loads((REPO_ROOT / notebook).read_text())

        assert [(c["cell_type"], _cell_content(c)) for c in built["cells"]] == [
            (c["cell_type"], _cell_content(c)) for c in committed["cells"]
        ], f"{notebook} has drifted from {script} — re-run {script} to rebuild it"
