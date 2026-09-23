"""The progress line must report the LR and AMP scale -- and must never raise.

That line runs inside the training step: if formatting could throw, a healthy run
would die over a log message.  The second test pins that property directly.
"""
from types import SimpleNamespace

from src.training.epoch import _optimizer_health


def _engine(lrs, scale):
    return SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"lr": lr} for lr in lrs]),
        scaler=SimpleNamespace(get_scale=lambda: scale),
    )


def test_reports_distinct_learning_rates_only():
    # eight real groups collapse to the distinct values
    health = _optimizer_health(_engine([1e-4] * 6 + [2e-5] * 2, 65536.0))
    assert health == "lr 2.00e-05,1.00e-04 | amp scale 65536"


def test_reports_a_single_learning_rate():
    assert _optimizer_health(_engine([1e-4] * 8, 1024.0)) == \
        "lr 1.00e-04 | amp scale 1024"


def test_survives_a_degraded_engine_instead_of_raising():
    broken = SimpleNamespace(optimizer=object(), scaler=object())
    assert _optimizer_health(broken) == "lr ? | amp scale ?"


def test_survives_a_missing_scaler():
    engine = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]))
    assert _optimizer_health(engine) == "lr 1.00e-04 | amp scale ?"
