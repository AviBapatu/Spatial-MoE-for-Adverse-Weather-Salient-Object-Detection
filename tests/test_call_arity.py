"""Static guard: a call must pass a number of arguments the function accepts.

A signature in this repo was changed while its call site was not. The mistake
survived the test suite (nothing imported that path) and only surfaced on Kaggle,
at process start, under two ranks -- after a packaging round trip. Python's
compiler cannot catch it, so it is caught here by parsing the source instead of
running it.

Only direct ``Name(...)`` calls are compared, and only against functions defined
exactly once *in the same file*: the same name is defined in several modules here,
so a repo-wide scan compares call sites against the wrong definition and fires on
correct code. The comparison accounts for parameters with defaults and for
arguments passed by keyword -- getting either of those wrong does the same.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"


def _signature_shape(node: ast.FunctionDef | ast.AsyncFunctionDef):
    """Return (required, positional_capacity, accepts_anything_extra)."""
    args = node.args
    positional = args.posonlyargs + args.args  # keyword-only params cannot be positional
    total = len(positional)
    required = total - len(args.defaults)
    accepts_extra = bool(args.vararg) or bool(args.kwarg)
    return required, total, accepts_extra


def _functions_in_file(tree: ast.AST) -> dict:
    """Map name -> signature shape, only for names defined once in this file."""
    found: dict = {}
    repeated: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in found:
                repeated.add(node.name)
            found[node.name] = _signature_shape(node)
    for name in repeated:
        found.pop(name, None)
    return found


def _find_arity_problems(root: Path) -> list:
    """Return human-readable arity problems for every .py file under *root*."""
    problems = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        functions = _functions_in_file(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in functions:
                continue
            if any(isinstance(a, ast.Starred) for a in node.args):
                continue  # *splat: unknowable statically
            if any(k.arg is None for k in node.keywords):
                continue  # **kwargs splat
            required, total, accepts_extra = functions[node.func.id]
            if accepts_extra:
                continue
            supplied = len(node.args) + sum(1 for k in node.keywords if k.arg is not None)
            if supplied < required or supplied > total:
                problems.append(
                    f"{path.name}:{node.lineno} calls {node.func.id}() with {supplied} "
                    f"arguments, but it accepts {required}..{total}"
                )
    return problems


def test_every_direct_call_passes_a_number_of_arguments_the_function_accepts() -> None:
    problems = _find_arity_problems(SRC)
    assert not problems, "argument-count mismatch(es):\n  " + "\n  ".join(problems)


def test_the_checker_flags_a_call_that_misses_a_required_argument(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("def f(a, b):\n    return a, b\n\n\nf(1)\n")
    problems = _find_arity_problems(tmp_path)
    assert len(problems) == 1 and "f()" in problems[0], problems


def test_the_checker_flags_a_call_with_too_many_arguments(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("def f(a):\n    return a\n\n\nf(1, 2)\n")
    assert len(_find_arity_problems(tmp_path)) == 1


def test_the_checker_allows_defaults_and_keyword_arguments(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "def f(a, b=1):\n    return a, b\n\n\n"
        "def g(x, y, z):\n    return x, y, z\n\n\n"
        "f(1)\nf(1, 2)\nf(a=1)\ng(x=1, y=2, z=3)\n"
    )
    assert _find_arity_problems(tmp_path) == []


def test_the_checker_skips_splats_and_shadowed_names(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "def f(a, b):\n    return a, b\n\n\n"
        "args = [1]\nf(*args)\nf(1, **{'b': 2})\n"
        "def g(a, b):\n    return a, b\n\n\n"
        "def g(a):\n    return a\n"
    )
    assert _find_arity_problems(tmp_path) == []


def test_the_checker_flags_the_real_regression_it_was_written_for(tmp_path: Path) -> None:
    """The exact bug: setup.py dropped a parameter from apply_resume's signature
    but the call site kept passing it -- only visible at process start on Kaggle."""
    source = (SRC / "training" / "setup.py").read_text()
    broken = source.replace(
        "        args, config, model, engine, base_dir, device, cfg_hash,\n",
        "        args, config, model, engine, base_dir, device, cfg_hash, project_root,\n",
        1,
    )
    assert broken != source, "the call site moved -- update this test"
    (tmp_path / "setup.py").write_text(broken)
    problems = _find_arity_problems(tmp_path)
    assert any("apply_resume" in p for p in problems), problems
