"""Loss components for the Spatial-MoE SOD pipeline.

The module provides:

- :func:`gaussian`, :func:`create_window`, :func:`_ssim` and :class:`SSIM`:
  the canonical ``pytorch_ssim``-style SSIM implementation (unchanged).
- One standalone function per loss component (``bce_loss``, ``soft_iou_loss``,
  ``ssim_loss``, ``image_gradient_magnitude_loss``, ``moe_routing_losses``,
  ``aux_boundary_loss``, ``deep_supervision_loss``).  Each is independently
  testable and does not touch any configuration.
- :class:`CombinedLoss`: the single class that reads its weights from
  :class:`src.config.LossConfig` and combines the components, returning
  ``(total_loss, dict_of_components)`` per the shared contract.
- :class:`SpatialMoELoss`: legacy constructor-compatible wrapper returning a
  plain ``{...}`` dict for callers outside this refactor's scope (e.g.
  ``src/training/*``, ``src/smoke_test.py``).  It delegates to
  :class:`CombinedLoss` and keeps the old keyword-constructor + dict return
  behaviour bit-for-bit.
"""
from math import exp
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import LossConfig
from src.moe_layer import MoEOutput

ZLOSS_EPS = 1e-6

# ============================================================================
# 1. SSIM (canonical pytorch_ssim style, unchanged)
# ============================================================================


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Build a 1-D Gaussian kernel of ``window_size`` samples."""
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Build an ``[channel, 1, window_size, window_size]`` 2-D Gaussian window."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    """Compute SSIM between ``img1`` and ``img2`` using the given window."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    ssim_map = torch.clamp(ssim_map, min=-1.0, max=1.0)

    if size_average:
        return ssim_map.mean()
    return ssim_map


class SSIM(nn.Module):
    """Single-channel SSIM loss helper with a cached Gaussian window."""

    def __init__(self, window_size: int = 11, size_average: bool = True) -> None:
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer("window", create_window(window_size, self.channel))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype and self.window.device == img1.device:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# ============================================================================
# 2. Standalone loss-component functions (one per component, fully testable)
# ============================================================================


def _masked_mean(values: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Reduce ``values`` with ``pad_mask``; fall back to the plain mean."""
    if pad_mask is not None:
        return (values * pad_mask).sum() / pad_mask.sum().clamp(min=1.0)
    return values.mean()


def bce_loss(
    logits: torch.Tensor, target: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Binary cross-entropy on ``logits`` (logits in, not sigmoided), with ``pad_mask``."""
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _masked_mean(bce_map, pad_mask)


def soft_iou_loss(
    P: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6
) -> torch.Tensor:
    """Soft IoU computed independently per image then averaged over the batch.

    ``P`` and ``target`` are ``[B, 1, H, W]``; ``P`` is already sigmoided.
    """
    if mask is not None:
        P = P * mask
        target = target * mask
    intersection = torch.sum(P * target, dim=(2, 3))
    union = torch.sum(P, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)  # [B, 1]
    return torch.mean(1.0 - iou)


def ssim_loss(
    P: torch.Tensor, target: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
    window_size: int = 11,
) -> torch.Tensor:
    """Return ``1 - SSIM(P, target)`` where ``P`` is a sigmoided ``[B, 1, H, W]`` tensor."""
    window = create_window(window_size, 1).to(P.device).type_as(P)
    ssim_map = _ssim(P, target, window, window_size, 1, size_average=False)
    return 1.0 - _masked_mean(ssim_map, pad_mask)


def image_gradient_magnitude_loss(
    P: torch.Tensor,
    gt_boundary: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Boundary loss: Smooth-L1 between predicted gradient magnitude and GT boundary.

    ``P`` is a sigmoided ``[B, 1, H, W]`` prediction; ``gt_boundary`` the binary
    boundary target.  Finite differences approximate the gradient magnitude.
    """
    gx = torch.zeros_like(P)
    gy = torch.zeros_like(P)

    gx[:, :, :, :-1] = P[:, :, :, 1:] - P[:, :, :, :-1]
    gy[:, :, :-1, :] = P[:, :, 1:, :] - P[:, :, :-1, :]

    grad_mag = torch.sqrt(gx**2 + gy**2 + eps)

    l1_map = F.smooth_l1_loss(grad_mag, gt_boundary, reduction="none")
    if mask is not None:
        return (l1_map * mask).sum() / mask.sum().clamp(min=1.0)
    return l1_map.mean()


def moe_routing_losses(
    moe_outputs: List[MoEOutput], z_enabled: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load-balance, importance, and (optionally) Z-loss averaged over MoE scales.

    ``moe_outputs`` is the list of per-scale :class:`MoEOutput` s from the model.
    Returns ``(l_lb, l_imp, l_z)`` each scaled by ``1 / num_scales``.
    """
    if not moe_outputs:
        return torch.tensor(0.0).to("cpu"), torch.tensor(0.0).to("cpu"), torch.tensor(0.0).to("cpu")

    device = moe_outputs[0].features.device

    total_lb = torch.tensor(0.0, device=device)
    total_imp = torch.tensor(0.0, device=device)
    total_z = torch.tensor(0.0, device=device)

    for out in moe_outputs:
        B, N_tokens, K = out.topk_indices.shape
        # clean_logits is [B, E, H, W] in spatial dispatch mode.
        E = out.clean_logits.shape[1]
        total_assignments = B * N_tokens * K

        # 1. Load Balancing Loss (f_j = fraction of top-k assignments to expert j)
        flat_indices = out.topk_indices.view(-1)
        f_j_counts = torch.bincount(flat_indices, minlength=E)
        f_j = (f_j_counts.float() / total_assignments).detach()

        # P_j = soft routing mass (mean gate weight assigned to each expert)
        flat_gates = out.topk_gates.view(-1)
        P_j_sum = torch.zeros(E, dtype=flat_gates.dtype, device=device)
        P_j_sum.scatter_add_(0, flat_indices, flat_gates)
        P_j = P_j_sum / (B * N_tokens)

        l_lb = E * torch.sum(f_j * P_j)
        total_lb = total_lb + l_lb

        # 2. Importance Loss (spread of per-expert soft mass)
        mean_imp = P_j_sum.mean()
        std_imp = P_j_sum.std(unbiased=False)
        l_imp = (std_imp / (mean_imp + ZLOSS_EPS)) ** 2
        total_imp = total_imp + l_imp

        # 3. Z-Loss (discourage large router logits)
        if z_enabled:
            l_z = torch.mean(torch.logsumexp(out.clean_logits, dim=1).square())
            total_z = total_z + l_z

    num_scales = len(moe_outputs)
    return total_lb / num_scales, total_imp / num_scales, total_z / num_scales


def aux_boundary_loss(
    edge_logits: torch.Tensor, gt_boundary: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Binary cross-entropy between auxiliary boundary ``edge_logits`` and GT boundary."""
    return bce_loss(edge_logits, gt_boundary, pad_mask=pad_mask)


def deep_supervision_loss(
    aux_logits_16: torch.Tensor,
    aux_logits_8: torch.Tensor,
    aux_logits_4: torch.Tensor,
    target: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean BCE of three auxiliary saliency heads against a downsampled target.

    ``target`` is the full-resolution binary map ``[B, 1, H, W]``; aux logits are
    at 1/16, 1/8 and 1/4 scales.  ``pad_mask`` is interpolated with nearest
    neighbour to match each aux scale.
    """
    target_16 = F.interpolate(target, size=aux_logits_16.shape[2:], mode="nearest")
    target_8 = F.interpolate(target, size=aux_logits_8.shape[2:], mode="nearest")
    target_4 = F.interpolate(target, size=aux_logits_4.shape[2:], mode="nearest")

    if pad_mask is not None:
        mask_16 = F.interpolate(pad_mask, size=aux_logits_16.shape[2:], mode="nearest")
        mask_8 = F.interpolate(pad_mask, size=aux_logits_8.shape[2:], mode="nearest")
        mask_4 = F.interpolate(pad_mask, size=aux_logits_4.shape[2:], mode="nearest")

        l_ds_16 = bce_loss(aux_logits_16, target_16, pad_mask=mask_16)
        l_ds_8 = bce_loss(aux_logits_8, target_8, pad_mask=mask_8)
        l_ds_4 = bce_loss(aux_logits_4, target_4, pad_mask=mask_4)
    else:
        l_ds_16 = bce_loss(aux_logits_16, target_16)
        l_ds_8 = bce_loss(aux_logits_8, target_8)
        l_ds_4 = bce_loss(aux_logits_4, target_4)

    return (l_ds_16 + l_ds_8 + l_ds_4) / 3.0


# ============================================================================
# 3. Combined loss (single class reading weights from LossConfig)
# ============================================================================

_LOSS_KEY_NAMES: Tuple[str, ...] = (
    "L_total",
    "L_bce",
    "L_iou",
    "L_ssim",
    "L_boundary",
    "L_lb",
    "L_importance",
    "L_z",
    "L_aux_boundary",
    "L_deep_supervision",
)


class CombinedLoss(nn.Module):
    """Combine all loss components using the weights in a :class:`LossConfig`.

    Args:
        loss_cfg: The ``LossConfig`` whose ``*_weight`` fields drive the
            component combination.  A ``*_weight`` of ``0`` disables that
            component.

    ``forward`` returns ``(total_loss, component_dict)`` where ``component_dict``
    uses the shared-contract keys (``L_total``, ``L_bce``, ``L_iou``, ``L_ssim``,
    ``L_boundary``, ``L_lb``, ``L_importance``, ``L_z``, ``L_aux_boundary``,
    ``L_deep_supervision``).
    """

    def __init__(self, loss_cfg: LossConfig) -> None:
        super().__init__()
        self.loss_cfg = loss_cfg

    def forward(
        self,
        saliency_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        moe_outputs: List[MoEOutput],
        target: torch.Tensor,
        gt_boundary: Optional[torch.Tensor] = None,
        aux_logits_16: Optional[torch.Tensor] = None,
        aux_logits_8: Optional[torch.Tensor] = None,
        aux_logits_4: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the full loss; returns ``(total, components)``.

        Args:
            saliency_logits: Main saliency logits ``[B, 1, H, W]``.
            edge_logits: Boundary-head logits ``[B, 1, H, W]``.
            moe_outputs: Per-scale ``MoEOutput`` list (may be empty).
            target: Binary GT saliency ``[B, 1, H, W]``.
            gt_boundary: Optional binary boundary target.
            aux_logits_16/8/4: Optional deep-supervision logits at each scale.
            pad_mask: Optional valid-region mask used to exclude padding.

        Returns:
            ``(total_loss, components)`` per the shared contract.
        """
        cfg = self.loss_cfg
        target = target.float()
        if gt_boundary is not None:
            gt_boundary = gt_boundary.float()

        P = torch.sigmoid(saliency_logits)

        # 1. Primary saliency losses
        l_bce = bce_loss(saliency_logits, target, pad_mask=pad_mask)
        l_iou = soft_iou_loss(P, target, mask=pad_mask)
        l_ssim = ssim_loss(P, target, pad_mask=pad_mask)

        # 2. Direct boundary loss (only if a GT boundary exists and is weighted)
        l_boundary = torch.tensor(0.0, device=saliency_logits.device)
        if gt_boundary is not None and cfg.boundary_weight > 0:
            l_boundary = image_gradient_magnitude_loss(P, gt_boundary, mask=pad_mask)

        # 3. Router (MoE) losses
        l_lb, l_imp, l_z = moe_routing_losses(moe_outputs, z_enabled=cfg.z_loss_weight > 0)

        # 4. Auxiliary boundary loss
        l_aux_boundary = torch.tensor(0.0, device=saliency_logits.device)
        if gt_boundary is not None and cfg.aux_boundary_weight > 0:
            l_aux_boundary = aux_boundary_loss(edge_logits, gt_boundary, pad_mask=pad_mask)

        # 5. Deep-supervision loss
        l_deep_supervision = torch.tensor(0.0, device=saliency_logits.device)
        if (
            aux_logits_16 is not None
            and aux_logits_8 is not None
            and aux_logits_4 is not None
            and cfg.deep_supervision_weight > 0
        ):
            l_deep_supervision = deep_supervision_loss(
                aux_logits_16, aux_logits_8, aux_logits_4, target, pad_mask=pad_mask
            )

        # 6. Total aggregation
        l_total = (
            cfg.bce_weight * l_bce
            + cfg.iou_weight * l_iou
            + cfg.ssim_weight * l_ssim
            + cfg.boundary_weight * l_boundary
            + cfg.load_balance_weight * l_lb
            + cfg.importance_weight * l_imp
            + cfg.z_loss_weight * l_z
            + cfg.aux_boundary_weight * l_aux_boundary
            + cfg.deep_supervision_weight * l_deep_supervision
        )

        components = {
            "L_total": l_total,
            "L_bce": l_bce,
            "L_iou": l_iou,
            "L_ssim": l_ssim,
            "L_boundary": l_boundary,
            "L_lb": l_lb,
            "L_importance": l_imp,
            "L_z": l_z,
            "L_aux_boundary": l_aux_boundary,
            "L_deep_supervision": l_deep_supervision,
        }
        return l_total, components


class SpatialMoELoss(nn.Module):
    """Legacy compatibility wrapper around :class:`CombinedLoss`.

    Keeps the original keyword-constructor (``lambda_*`` floats) and plain-dict
    return contract for callers that predate the ``CombinedLoss`` refactor and
    are outside this session's scope (``src/training/*``, ``src/smoke_test.py``,
    ``tests/test_moe_ddp.py``).  Behaviour is identical to the pre-refactor
    ``SpatialMoELoss``; new code should use :class:`CombinedLoss`.
    """

    def __init__(
        self,
        lambda_iou: float = 1.0,
        lambda_ssim: float = 1.0,
        lambda_boundary: float = 0.5,
        lambda_lb: float = 0.01,
        lambda_importance: float = 0.01,
        lambda_z: float = 0.0,
        lambda_aux_boundary: float = 0.0,
        lambda_deep_supervision: float = 0.4,
    ) -> None:
        super().__init__()
        self._combined = CombinedLoss(
            LossConfig(
                iou_weight=lambda_iou,
                ssim_weight=lambda_ssim,
                boundary_weight=lambda_boundary,
                load_balance_weight=lambda_lb,
                importance_weight=lambda_importance,
                z_loss_weight=lambda_z,
                aux_boundary_weight=lambda_aux_boundary,
                deep_supervision_weight=lambda_deep_supervision,
            )
        )

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        _, components = self._combined(*args, **kwargs)
        return components
