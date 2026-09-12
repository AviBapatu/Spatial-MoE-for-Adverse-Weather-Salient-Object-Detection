"""Morphological boundary-band computation for binary segmentation masks."""
from typing import Optional

import cv2
import numpy as np

BOUNDARY_KERNEL_SIZE = 5
BOUNDARY_KERNEL_SHAPE = cv2.MORPH_ELLIPSE


def get_boundary_kernel() -> np.ndarray:
    """Return the default structuring element used for boundary extraction."""
    return cv2.getStructuringElement(
        BOUNDARY_KERNEL_SHAPE,
        (BOUNDARY_KERNEL_SIZE, BOUNDARY_KERNEL_SIZE),
    )


def compute_boundary(
    mask_binary: np.ndarray,
    kernel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a morphological boundary band from a binary mask.

    The boundary is defined as ``dilate(mask) - erode(mask)``, a one-band
    ring around the foreground/background transition.

    Args:
        mask_binary: Binary mask of shape ``(H, W)`` with values in ``{0, 1}``.
        kernel: Optional structuring element. If ``None``, the default 5x5
            ellipse kernel is used.

    Returns:
        Boundary mask of shape ``(H, W)`` with values in ``{0, 1}``
        (dtype ``np.float32``).
    """
    if kernel is None:
        kernel = get_boundary_kernel()

    mask_u8 = mask_binary.astype(np.uint8)

    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)

    boundary = dilated - eroded
    return boundary.astype(np.float32)
