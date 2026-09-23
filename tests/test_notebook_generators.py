"""Drift guard for the account notebooks and their generators.

``generate_notebook_accountA.py`` / ``_accountB.py`` are the editable source of
truth for the two Kaggle notebooks. Regenerating a notebook in a temp directory
must reproduce the committed notebook's cell content exactly, so a cell edited
in one place but not the other fails here instead of silently diverging — which
is how these notebooks previously went stale.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

GENERATORS = [
    ("generate_notebook_accountA.py", "moe-of-sod__2expert_accountA.ipynb"),
    ("generate_notebook_accountB.py", "moe-of-sod__4expert_accountB.ipynb"),
]


def _cell_content(cell: dict) -> str:
    """Return a cell's source as one string, whatever form the JSON stores it in."""
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


@pytest.mark.parametrize(("script", "notebook"), GENERATORS)
def test_generator_reproduces_committed_notebook(script: str, notebook: str, tmp_path: Path) -> None:
    """The generator must rebuild the committed notebook cell-for-cell."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    built = json.loads((tmp_path / notebook).read_text())
    committed = json.loads((REPO_ROOT / notebook).read_text())

    assert [(c["cell_type"], _cell_content(c)) for c in built["cells"]] == [
        (c["cell_type"], _cell_content(c)) for c in committed["cells"]
    ], f"{notebook} has drifted from {script} — re-run {script} to rebuild it"
