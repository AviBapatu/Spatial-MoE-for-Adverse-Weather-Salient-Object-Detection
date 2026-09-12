import numpy as np
import pytest
import py_sod_metrics

from src.metrics import (
    SODMetrics,
    BoundaryMetrics,
    compute_fbeta,
    compute_fmax,
    compute_mae,
)


def test_compute_mae_hand_computed():
    pred = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    gt = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    # pred -> [0, 1, 1, 0]; gt -> [0, 1, 0, 0]; |diff| = [0, 0, 1, 0]
    assert compute_mae(pred, gt) == 0.25


def test_compute_mae_matches_py_sod_metrics():
    rng = np.random.default_rng(0)
    pred = rng.random((64, 64)).astype(np.float32)
    gt = (rng.random((64, 64)) > 0.5).astype(np.float32)

    p_u8 = (pred * 255).astype(np.uint8)
    g_u8 = (gt * 255).astype(np.uint8)

    raw = py_sod_metrics.MAE()
    raw.step(p_u8, g_u8)
    expected = raw.get_results()["mae"]

    assert np.isclose(compute_mae(p_u8, g_u8), expected, atol=1e-12)
    # Float [0,1] input differs only by uint8 quantization.
    assert np.isclose(compute_mae(pred, gt), expected, atol=0.005)


def test_constant_prediction_mae():
    # Constant prediction: no min-max rescaling is applied (matches py_sod).
    pred = np.full((8, 8), 128, dtype=np.uint8)
    gt = (np.arange(64).reshape(8, 8) % 2).astype(np.uint8) * 255
    assert compute_mae(pred, gt) > 0.0


def test_compute_fbeta_hand_computed():
    pred = np.array([[0, 0], [1, 1]], dtype=np.float32)  # {0, 1}
    gt = np.zeros((2, 2), dtype=np.float32)
    gt[:, 1] = 1.0

    f1 = compute_fbeta(pred, gt, beta=1.0, threshold=0.5)
    # tp=1, fp=1, fn=1 -> P=0.5, R=0.5 -> F1=0.5
    assert np.isclose(f1, 0.5)

    f2 = compute_fbeta(pred, gt, beta=2.0, threshold=0.5)
    # P=0.5, R=0.5 -> F2 = (1+4)*0.25/(4*0.5+0.5) = 1.25/2.5 = 0.5
    assert np.isclose(f2, 0.5)


def test_compute_fmax_perfect_prediction_is_one():
    gt = np.zeros((16, 16), dtype=np.float32)
    gt[4:12, 4:12] = 1.0
    pred = gt.copy()
    assert compute_fmax(pred, gt) == 1.0


def test_compute_fmax_at_least_max_f1():
    rng = np.random.default_rng(1)
    pred = rng.random((32, 32)).astype(np.float32)
    gt = (rng.random((32, 32)) > 0.5).astype(np.float32)
    best = compute_fmax(pred, gt)
    at_half = compute_fbeta(pred, gt, threshold=0.5)
    assert best >= at_half


def test_sod_metrics_aggregation_matches_per_image_mae():
    rng = np.random.default_rng(3)
    preds = [rng.random((16, 16)).astype(np.float32) for _ in range(3)]
    gts = [(rng.random((16, 16)) > 0.5).astype(np.float32) for _ in range(3)]

    metrics = SODMetrics()
    for p, g in zip(preds, gts):
        metrics.step(p, g)

    results = metrics.get_results()
    # SODMetrics.step feeds uint8-quantized arrays to compute_mae; mirror that.
    manual = np.mean(
        [
            compute_mae((p * 255).astype(np.uint8), (g * 255).astype(np.uint8))
            for p, g in zip(preds, gts)
        ]
    )
    assert np.isclose(results["MAE"], manual, atol=1e-12)

    for key in ["S_measure", "E_adaptive", "E_mean", "E_max", "F_adaptive", "F_mean", "F_max"]:
        assert np.isfinite(results[key])


def test_boundary_metrics_sanity():
    metrics = BoundaryMetrics()
    pred = np.zeros((16, 16), dtype=np.float32)
    pred[4:12, 4:12] = 1.0
    gt = pred.copy()

    metrics.step(pred, gt)
    res = metrics.get_results()
    assert np.isclose(res["boundary_MAE"], 0.0)
    assert np.isclose(res["boundary_F1"], 1.0)

    metrics.step(1.0 - pred, gt)
    res = metrics.get_results()
    assert res["boundary_F1"] < 1.0