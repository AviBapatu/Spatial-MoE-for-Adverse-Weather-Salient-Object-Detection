"""SOD evaluation metrics.

Two layers:

- Standalone, testable metric functions (:func:`compute_mae`, :func:`compute_fbeta`,
  :func:`compute_fmax`).  These implement the definitions directly (MAE, F-beta /
  max-F1) and are the reference implementations used by the tests.
- :class:`SODMetrics` / :class:`BoundaryMetrics`: the aggregating wrappers used by
  ``src/evaluate.py`` and ``src/training/loop.py``.  ``SODMetrics`` keeps the
  ``py_sod_metrics`` backends for S/E/F (which have no single-expression
  definition) and computes MAE via :func:`compute_mae`.

Input conventions for the standalone functions (and ``SODMetrics.step``):
``pred`` and ``gt`` are 2-D arrays of identical ``(H, W)`` shape; ``pred`` is a
float saliency map in ``[0, 1]`` (or ``uint8`` in ``[0, 255]``); ``gt`` is a
binary mask in ``{0, 1}`` (or ``{0, 255}``).  The float range is auto-detected
from the ``max`` value.
"""
from typing import List, Optional

import numpy as np
import py_sod_metrics

F_EPS = 1e-6


def _as_float01(values: np.ndarray, invert255: bool = False) -> np.ndarray:
    """Convert a 2-D ``pred``/``gt`` array into float ``[0, 1]`` values.

    Values with ``max() > 1.0`` are treated as 8-bit ``[0, 255]`` and scaled;
    otherwise they are assumed to already be in ``[0, 1]``.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    if v.max() > 1.0:
        v = v / 255.0
    if invert255:
        v = (v > 0.5).astype(np.float64)
    return v


def _as_mae_pred01(values: np.ndarray) -> np.ndarray:
    """Normalize ``pred`` exactly like ``py_sod_metrics.prepare_data``.

    ``py_sod_metrics`` divides an 8-bit map by 255, then applies per-image
    min-max scaling (``(p - min) / (max - min)``) unless the map is constant.
    ``SODMetrics`` has always reported MAE through ``py_sod_metrics``, so these
    semantics are part of the metric's contract and must not change.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    if v.max() > 1.0:
        v = v / 255.0
    if v.max() != v.min():
        v = (v - v.min()) / (v.max() - v.min())
    return v


def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute error between a saliency map and its binary mask.

    Numerically identical to ``py_sod_metrics.MAE`` (including its per-image
    min-max normalization of the prediction), so ``SODMetrics`` can delegate to
    this function without changing any existing evaluation result.

    Args:
        pred: ``(H, W)`` saliency map, floats in ``[0, 1]`` or ``uint8`` in
            ``[0, 255]``.
        gt: ``(H, W)`` binary mask in ``{0, 1}`` or ``{0, 255}``.

    Returns:
        ``mean(|pred - gt|)`` over all pixels, in ``[0, 1]``.
    """
    p = _as_mae_pred01(pred)
    g = _as_float01(gt, invert255=True)
    return float(np.mean(np.abs(p - g)))


def compute_fbeta(
    pred: np.ndarray, gt: np.ndarray, beta: float = 1.0, threshold: float = 0.5
) -> float:
    """F-beta score of a saliency map against a binary mask at one threshold.

    Args:
        pred: ``(H, W)`` saliency map in ``[0, 1]`` (or ``uint8`` ``[0, 255]``).
        gt: ``(H, W)`` binary mask in ``{0, 1}`` (or ``{0, 255}``).
        beta: F-beta parameter (``beta=1`` gives F1; ``beta>1`` weights recall
            more heavily).
        threshold: Binarization threshold for ``pred``.

    Returns:
        ``(1 + beta**2) * P * R / (beta**2 * P + R)`` in ``[0, 1]``.
    """
    p = _as_float01(pred)
    g = _as_float01(gt, invert255=True)
    p_bin = (p > threshold).astype(np.float64)

    tp = float(np.sum((p_bin == 1.0) & (g == 1.0)))
    fp = float(np.sum((p_bin == 1.0) & (g == 0.0)))
    fn = float(np.sum((p_bin == 0.0) & (g == 1.0)))

    precision = tp / max(tp + fp, F_EPS)
    recall = tp / max(tp + fn, F_EPS)
    denom = beta ** 2 * precision + recall
    f = (1.0 + beta ** 2) * precision * recall / max(denom, F_EPS)
    return float(f)


def compute_fmax(pred: np.ndarray, gt: np.ndarray, beta: float = 1.0) -> float:
    """Maximum F-beta over all distinct thresholds present in ``pred``.

    Standard SOD F-max score: sweep the binarization threshold across the
    unique prediction values and report the best F-beta.

    Args:
        pred: ``(H, W)`` saliency map in ``[0, 1]`` (or ``uint8`` ``[0, 255]``).
        gt: ``(H, W)`` binary mask in ``{0, 1}`` (or ``{0, 255}``).
        beta: F-beta parameter.

    Returns:
        Maximum F-beta over the threshold sweep.
    """
    p = _as_float01(pred)
    g = _as_float01(gt, invert255=True)
    thresholds = np.unique(p)
    if thresholds.size == 0:
        return 0.0
    return max(compute_fbeta(p, g, beta=beta, threshold=float(t)) for t in thresholds)


class SODMetrics:
    """Aggregate SOD metrics over many images (MAE, S, E, F).

    ``step`` accepts ``pred``/``gt`` as ``(H, W)`` numpy arrays (saliency in
    ``[0, 1]`` or ``[0, 255]``, mask binary) and accumulates running statistics;
    ``get_results`` returns the aggregate dictionary.  MAE is computed by
    :func:`compute_mae`; S-/E-/F-measures keep the ``py_sod_metrics`` backends.
    """

    def __init__(self) -> None:
        self.MAE: List[float] = []
        self.SM = py_sod_metrics.Smeasure()
        self.EM = py_sod_metrics.Emeasure()
        self.FM = py_sod_metrics.Fmeasure()

    def step(self, pred: np.ndarray, gt: np.ndarray, image_id: Optional[str] = None) -> None:
        """Record metrics for a single image pair.

        Args:
            pred: ``(H, W)`` saliency array, floats in ``[0, 1]`` or ``uint8`` in
                ``[0, 255]``.
            gt: ``(H, W)`` binary array in ``{0, 1}`` or ``{0, 255}``.
            image_id: Reserved; kept for compatibility (currently unused).
        """
        if pred.max() <= 1.0:
            p = (pred * 255).astype(np.uint8)
        else:
            p = pred.astype(np.uint8)

        if gt.max() <= 1.0:
            g = (gt > 0.5).astype(np.uint8) * 255
        else:
            g = (gt > 127).astype(np.uint8) * 255

        self.MAE.append(compute_mae(p, g))
        self.SM.step(pred=p, gt=g)
        self.EM.step(pred=p, gt=g)
        self.FM.step(pred=p, gt=g)

    def get_results(self) -> dict:
        """Aggregate results across all ``step`` calls.

        Returns:
            Dict with ``MAE``, ``S_measure``, ``E_adaptive``/``E_mean``/``E_max``
            and ``F_adaptive``/``F_mean``/``F_max``.
        """
        mae = float(np.mean(np.array(self.MAE))) if self.MAE else 0.0
        sm = self.SM.get_results()["sm"]
        em_res = self.EM.get_results()["em"]
        fm_res = self.FM.get_results()["fm"]

        return {
            "MAE": mae,
            "S_measure": sm,
            "E_adaptive": em_res["adp"],
            "E_mean": em_res["curve"].mean(),
            "E_max": em_res["curve"].max(),
            "F_adaptive": fm_res["adp"],
            "F_mean": fm_res["curve"].mean(),
            "F_max": fm_res["curve"].max(),
        }


class BoundaryMetrics:
    """Boundary-focused metrics computed without ``py_sod_metrics``.

    Accumulates boundary-region MAE and boundary F1 (thresholded at 0.5) over
    many images; ``get_results`` returns the aggregates.
    """

    def __init__(self) -> None:
        self.total_mae = 0.0
        self.total_pixels = 0

        self.tp = 0
        self.fp = 0
        self.fn = 0

    def step(self, pred_prob: np.ndarray, gt_boundary: np.ndarray) -> None:
        """Record boundary metrics for one image pair.

        Args:
            pred_prob: ``(H, W)`` continuous boundary predictions in ``[0, 1]``.
            gt_boundary: ``(H, W)`` binary boundary mask in ``{0, 1}``.
        """
        p = np.clip(pred_prob, 0.0, 1.0)
        g = (gt_boundary > 0.5).astype(np.float32)

        # Boundary MAE over the boundary pixels only.
        boundary_mask = g == 1.0
        if np.any(boundary_mask):
            abs_diff = np.abs(p[boundary_mask] - g[boundary_mask])
            self.total_mae += np.sum(abs_diff)
            self.total_pixels += int(np.sum(boundary_mask))

        # Boundary F1 (binarize prediction at 0.5).
        p_bin = (p > 0.5).astype(np.float32)

        self.tp += int(np.sum((p_bin == 1) & (g == 1)))
        self.fp += int(np.sum((p_bin == 1) & (g == 0)))
        self.fn += int(np.sum((p_bin == 0) & (g == 1)))

    def get_results(self) -> dict:
        """Aggregate boundary results.

        Returns:
            Dict with ``boundary_MAE`` and ``boundary_F1``.
        """
        mae = self.total_mae / max(1, self.total_pixels)

        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)

        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "boundary_MAE": mae,
            "boundary_F1": f1,
        }
