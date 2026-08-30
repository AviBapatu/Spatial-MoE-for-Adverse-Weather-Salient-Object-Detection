import py_sod_metrics
import numpy as np

class SODMetrics:
    """
    Thin wrapper around py_sod_metrics for standardized SOD evaluation.
    Computes MAE, S-measure, E-measure (max, mean, adaptive), and F-measure (max, mean, adaptive).
    """
    def __init__(self):
        self.MAE = py_sod_metrics.MAE()
        self.SM = py_sod_metrics.Smeasure()
        self.EM = py_sod_metrics.Emeasure()
        self.FM = py_sod_metrics.Fmeasure()
        self.image_metrics = {}

    def step(self, pred, gt, image_id=None):
        """
        pred: numpy array of shape (H, W). Values in [0, 255].
        gt: numpy array of shape (H, W). Values in {0, 255}.
        """
        if pred.max() <= 1.0:
            p = (pred * 255).astype(np.uint8)
        else:
            p = pred.astype(np.uint8)

        if gt.max() <= 1.0:
            g = (gt > 0.5).astype(np.uint8) * 255
        else:
            g = (gt > 127).astype(np.uint8) * 255

        self.MAE.step(pred=p, gt=g)
        self.SM.step(pred=p, gt=g)
        self.EM.step(pred=p, gt=g)
        self.FM.step(pred=p, gt=g)
        
        # Optionally, compute isolated per-image metrics if image_id is provided
        if image_id is not None:
            # Note: py_sod_metrics maintains internal running lists for some metrics,
            # but S-measure and MAE are simple enough to calculate locally for a single image,
            # or we rely on the final output arrays. 
            # We'll just store the id for now, and rely on the full evaluation for aggregate stats.
            pass

    def get_results(self):
        mae = self.MAE.get_results()["mae"]
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
            "F_max": fm_res["curve"].max()
        }

class BoundaryMetrics:
    """
    Custom auxiliary metrics for boundary analysis.
    Explicitly NOT using py_sod_metrics to maintain controlled definitions.
    """
    def __init__(self):
        self.total_mae = 0.0
        self.total_pixels = 0
        
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def step(self, pred_prob, gt_boundary):
        """
        pred_prob: continuous predictions in [0, 1]
        gt_boundary: binary boundary mask in {0, 1}
        """
        # Ensure correct formats
        p = np.clip(pred_prob, 0.0, 1.0)
        g = (gt_boundary > 0.5).astype(np.float32)
        
        # Boundary MAE over the boundary pixels only
        boundary_mask = (g == 1.0)
        if np.any(boundary_mask):
            abs_diff = np.abs(p[boundary_mask] - g[boundary_mask])
            self.total_mae += np.sum(abs_diff)
            self.total_pixels += np.sum(boundary_mask)
            
        # Boundary F1 (binarize prediction at 0.5 for F1)
        p_bin = (p > 0.5).astype(np.float32)
        
        tp = np.sum((p_bin == 1) & (g == 1))
        fp = np.sum((p_bin == 1) & (g == 0))
        fn = np.sum((p_bin == 0) & (g == 1))
        
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def get_results(self):
        mae = self.total_mae / max(1, self.total_pixels)
        
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        return {
            "boundary_MAE": mae,
            "boundary_F1": f1
        }
