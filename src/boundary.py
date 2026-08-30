import cv2
import numpy as np

BOUNDARY_KERNEL_SIZE = 5
BOUNDARY_KERNEL_SHAPE = cv2.MORPH_ELLIPSE

def get_boundary_kernel():
    return cv2.getStructuringElement(
        BOUNDARY_KERNEL_SHAPE, 
        (BOUNDARY_KERNEL_SIZE, BOUNDARY_KERNEL_SIZE)
    )

def compute_boundary(mask_binary, kernel=None):
    """
    Computes a morphological boundary band from a binary mask.
    mask_binary: numpy array of shape (H, W), values in {0, 1}
    kernel: optional structuring element. If None, uses default 5x5 ellipse.
    Returns: numpy array of shape (H, W), values in {0, 1}
    """
    if kernel is None:
        kernel = get_boundary_kernel()
        
    mask_u8 = mask_binary.astype(np.uint8)
    
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    
    boundary = dilated - eroded
    return boundary.astype(np.float32)
