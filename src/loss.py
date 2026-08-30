import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

# ==============================================================================
# 1. SSIM Implementation (Canonical pytorch_ssim style)
# ==============================================================================
def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
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

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer('window', create_window(window_size, self.channel))

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype and self.window.device == img1.device:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# ==============================================================================
# 2. Main Spatial-MoE Loss System
# ==============================================================================
class SpatialMoELoss(nn.Module):
    def __init__(self, 
                 lambda_iou=1.0, 
                 lambda_ssim=1.0, 
                 lambda_boundary=0.5,
                 lambda_lb=0.01,
                 lambda_importance=0.01,
                 lambda_z=0.0,
                 lambda_aux_boundary=0.0,
                 lambda_deep_supervision=0.4):
        super().__init__()
        self.lambda_iou = lambda_iou
        self.lambda_ssim = lambda_ssim
        self.lambda_boundary = lambda_boundary
        self.lambda_lb = lambda_lb
        self.lambda_importance = lambda_importance
        self.lambda_z = lambda_z
        self.lambda_aux_boundary = lambda_aux_boundary
        self.lambda_deep_supervision = lambda_deep_supervision
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.smooth_l1 = nn.SmoothL1Loss()
        
    def _soft_iou_loss(self, P, target, eps=1e-6):
        """Computes soft IoU independently per image and averages across batch."""
        # P and target are [B, 1, H, W]
        intersection = torch.sum(P * target, dim=(2, 3))
        union = torch.sum(P, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection
        
        iou = (intersection + eps) / (union + eps) # [B, 1]
        return torch.mean(1.0 - iou)

    def _boundary_loss(self, P, gt_boundary, eps=1e-6):
        """Computes gradient magnitude on predictions and compares with GT boundary."""
        # P is [B, 1, H, W]
        gx = torch.zeros_like(P)
        gy = torch.zeros_like(P)
        
        gx[:, :, :, :-1] = P[:, :, :, 1:] - P[:, :, :, :-1]
        gy[:, :, :-1, :] = P[:, :, 1:, :] - P[:, :, :-1, :]
        
        grad_mag = torch.sqrt(gx**2 + gy**2 + eps)
        
        # SmoothL1 loss between predicted gradient magnitude and binary GT boundary
        return self.smooth_l1(grad_mag, gt_boundary)

    def _moe_losses(self, moe_outputs):
        """Computes load balancing, importance, and Z-loss across MoE scales."""
        if not moe_outputs:
            return torch.tensor(0.0).to('cpu'), torch.tensor(0.0).to('cpu'), torch.tensor(0.0).to('cpu')
            
        device = moe_outputs[0].features.device
                   
        total_lb = torch.tensor(0.0, device=device)
        total_imp = torch.tensor(0.0, device=device)
        total_z = torch.tensor(0.0, device=device)
        
        for out in moe_outputs:
            B, N_tokens, K = out.topk_indices.shape
            E = out.clean_logits.shape[1]
            total_assignments = B * N_tokens * K
            
            # 1. Load Balancing Loss
            # f_j = hard assignment fraction (detached)
            flat_indices = out.topk_indices.view(-1)
            f_j_counts = torch.bincount(flat_indices, minlength=E)
            f_j = (f_j_counts.float() / total_assignments).detach()
            
            # P_j = soft routing mass (mean of gate weights assigned to expert j over all tokens)
            flat_gates = out.topk_gates.view(-1)
            P_j_sum = torch.zeros(E, dtype=flat_gates.dtype, device=device)
            P_j_sum.scatter_add_(0, flat_indices, flat_gates)
            P_j = P_j_sum / (B * N_tokens)
            
            l_lb = E * torch.sum(f_j * P_j)
            total_lb = total_lb + l_lb
            
            # 2. Importance Loss
            mean_imp = P_j_sum.mean()
            std_imp = P_j_sum.std(unbiased=False)
            l_imp = (std_imp / (mean_imp + 1e-6))**2
            total_imp = total_imp + l_imp
            
            # 3. Z-Loss (Optional)
            if self.lambda_z > 0:
                l_z = torch.mean(torch.logsumexp(out.clean_logits, dim=1)**2)
                total_z = total_z + l_z
                
        num_scales = len(moe_outputs)
        return total_lb / num_scales, total_imp / num_scales, total_z / num_scales

    def forward(self, saliency_logits, edge_logits, moe_outputs, target, gt_boundary=None, aux_logits_16=None, aux_logits_8=None, aux_logits_4=None):
        """
        Computes the complete loss.
        """
        target = target.float()
        if gt_boundary is not None:
            gt_boundary = gt_boundary.float()
        
        P = torch.sigmoid(saliency_logits)
        
        # 1. Primary Saliency Losses
        l_bce = self.bce_loss(saliency_logits, target)
        l_iou = self._soft_iou_loss(P, target)
        l_ssim = 1.0 - self.ssim_loss(P, target)
        
        # 2. Direct Boundary Loss
        l_boundary = torch.tensor(0.0, device=saliency_logits.device)
        if gt_boundary is not None and self.lambda_boundary > 0:
            l_boundary = self._boundary_loss(P, gt_boundary)
            
        # 3. Router Losses
        l_lb, l_imp, l_z = self._moe_losses(moe_outputs)
        
        # 4. Auxiliary Boundary Loss
        l_aux_boundary = torch.tensor(0.0, device=saliency_logits.device)
        if gt_boundary is not None and self.lambda_aux_boundary > 0:
            l_aux_boundary = self.bce_loss(edge_logits, gt_boundary)
            
        # 5. Deep Supervision Loss
        l_deep_supervision = torch.tensor(0.0, device=saliency_logits.device)
        if aux_logits_16 is not None and aux_logits_8 is not None and aux_logits_4 is not None and self.lambda_deep_supervision > 0:
            # Interpolate target to match aux logits spatial dimensions using nearest neighbor
            target_16 = F.interpolate(target, size=aux_logits_16.shape[2:], mode='nearest')
            target_8 = F.interpolate(target, size=aux_logits_8.shape[2:], mode='nearest')
            target_4 = F.interpolate(target, size=aux_logits_4.shape[2:], mode='nearest')
            
            l_ds_16 = self.bce_loss(aux_logits_16, target_16)
            l_ds_8 = self.bce_loss(aux_logits_8, target_8)
            l_ds_4 = self.bce_loss(aux_logits_4, target_4)
            
            # Average across the 3 scales
            l_deep_supervision = (l_ds_16 + l_ds_8 + l_ds_4) / 3.0
            
        # 6. Total Aggregation
        l_total = l_bce \
                + self.lambda_iou * l_iou \
                + self.lambda_ssim * l_ssim \
                + self.lambda_boundary * l_boundary \
                + self.lambda_lb * l_lb \
                + self.lambda_importance * l_imp \
                + self.lambda_z * l_z \
                + self.lambda_aux_boundary * l_aux_boundary \
                + self.lambda_deep_supervision * l_deep_supervision
                
        return {
            "L_total": l_total,
            "L_bce": l_bce,
            "L_iou": l_iou,
            "L_ssim": l_ssim,
            "L_boundary": l_boundary,
            "L_lb": l_lb,
            "L_importance": l_imp,
            "L_z": l_z,
            "L_aux_boundary": l_aux_boundary,
            "L_deep_supervision": l_deep_supervision
        }
