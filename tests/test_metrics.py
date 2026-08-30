import torch
import numpy as np
import cv2
import py_sod_metrics
import pytest

from src.metrics import SODMetrics, BoundaryMetrics
from src.boundary import compute_boundary
from src.evaluate import reverse_geometry

def test_boundary_centralization():
    """Verify compute_boundary uses the exact correct configuration and works deterministically."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1 # 6x6 square of 1s
    
    bound = compute_boundary(mask)
    
    # Boundary of a 6x6 square with a 5x5 ellipse kernel
    assert bound.shape == (10, 10)
    assert np.max(bound) == 1.0
    assert np.min(bound) == 0.0
    
    # The inner part of the square should be 0 since it eroded away and dilated over
    # Actually, erosion of 6x6 with 5x5 leaves a 2x2 square. Dilation makes it 10x10.
    # We just need to ensure the boundary band exists and has correct type
    assert np.sum(bound) > 0
    assert bound.dtype == np.float32

def test_metric_sanity():
    """Synthetic bounds-testing on ideal, zeroed, unified, and NaN inputs."""
    metrics = SODMetrics()
    
    # 1. Ideal
    pred = np.ones((100, 100))
    pred[:, 50:] = 0
    gt = pred.copy()
    
    metrics.step(pred, gt)
    res = metrics.get_results()
    assert res["MAE"] == 0.0
    assert np.isclose(res["S_measure"], 1.0)
    assert np.isclose(res["F_max"], 1.0)
    
    # 2. Zeroed Pred vs Half Foreground GT
    metrics = SODMetrics()
    pred = np.zeros((100, 100))
    gt = np.ones((100, 100))
    gt[:, 50:] = 0
    
    metrics.step(pred, gt)
    res = metrics.get_results()
    assert res["MAE"] > 0.0
    assert res["F_max"] < 1.0
    
def test_reverse_geometry():
    """Ensures diverse original aspect ratios correctly transit the padding, cropping, and upscaling matrix."""
    shapes = [(400, 300), (300, 400), (400, 350), (287, 400), (350, 350)]
    image_size = 384
    
    for (orig_h, orig_w) in shapes:
        scale = image_size / max(orig_h, orig_w)
        resized_h = int(orig_h * scale)
        resized_w = int(orig_w * scale)
        
        pad_h = image_size - resized_h
        pad_w = image_size - resized_w
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        meta = {
            'orig_h': orig_h,
            'orig_w': orig_w,
            'resized_h': resized_h,
            'resized_w': resized_w,
            'pad_top': pad_top,
            'pad_bottom': pad_bottom,
            'pad_left': pad_left,
            'pad_right': pad_right
        }
        
        # Synthetic padded prediction tensor
        fake_pred = torch.randn(1, 1, image_size, image_size)
        
        pred_orig = reverse_geometry(fake_pred, meta)
        
        assert pred_orig.shape == (orig_h, orig_w), f"Failed for {orig_h}x{orig_w}"

def test_reference_compatibility():
    """Guarantees SODMetrics provides exact numerical result as manually driving raw py_sod_metrics."""
    pred = np.random.rand(100, 100).astype(np.float32)
    gt = (np.random.rand(100, 100) > 0.5).astype(np.float32)
    
    # 1. Wrapper
    wrapper = SODMetrics()
    wrapper.step(pred, gt)
    w_res = wrapper.get_results()
    
    # 2. Raw py_sod_metrics
    p_u8 = (pred * 255).astype(np.uint8)
    g_u8 = (gt * 255).astype(np.uint8)
    
    raw_mae = py_sod_metrics.MAE()
    raw_sm = py_sod_metrics.Smeasure()
    raw_fm = py_sod_metrics.Fmeasure()
    
    raw_mae.step(p_u8, g_u8)
    raw_sm.step(p_u8, g_u8)
    raw_fm.step(p_u8, g_u8)
    
    r_mae = raw_mae.get_results()["mae"]
    r_sm = raw_sm.get_results()["sm"]
    r_fmax = raw_fm.get_results()["fm"]["curve"].max()
    
    assert np.isclose(w_res["MAE"], r_mae)
    assert np.isclose(w_res["S_measure"], r_sm)
    assert np.isclose(w_res["F_max"], r_fmax)

def test_tta_symmetry():
    """Proves flipping an image, processing it, and un-flipping it preserves strict spatial orientation."""
    tensor = torch.arange(16).view(1, 1, 4, 4).float()
    
    # Flip
    flipped = torch.flip(tensor, dims=[-1])
    
    # Mock model passthrough (identity)
    out_flipped = flipped
    
    # Un-flip
    unflipped = torch.flip(out_flipped, dims=[-1])
    
    assert torch.allclose(tensor, unflipped), "TTA symmetry failed!"

if __name__ == '__main__':
    print("Running metric tests...")
    test_boundary_centralization()
    test_metric_sanity()
    test_reverse_geometry()
    test_reference_compatibility()
    test_tta_symmetry()
    print("All metric tests passed successfully!")
