import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    """
    Region-level Intersection-over-Union loss for structural consistency.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute Intersection and Union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        # IoU and loss
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou

def load_balancing_loss(routing_probs):
    """
    Computes Shazeer/Switch-Transformer Load Balancing Loss.
    L = N * sum(f_j * P_mean_j)
    where f_j is the fraction of tokens routed to expert j (approximated via argmax),
    and P_mean_j is the mean routing probability to expert j.
    """
    B, N, H, W = routing_probs.shape
    
    # 1. Fraction of tokens routed to expert j (f_j)
    # Using hard argmax assignment as a proxy for the actual routing decision
    top1_idx = torch.argmax(routing_probs, dim=1) # (B, H, W)
    f_j = torch.zeros(N, device=routing_probs.device, dtype=routing_probs.dtype)
    for j in range(N):
        f_j[j] = (top1_idx == j).float().mean()
        
    # 2. Mean routing probability for expert j (P_mean_j)
    P_mean_j = routing_probs.mean(dim=(0, 2, 3))
    
    # Compute loss
    loss = N * torch.sum(f_j * P_mean_j)
    return loss

def importance_loss(routing_probs):
    """
    Computes Importance Loss using the Coefficient of Variation across expert assignment mass.
    Prevents mode collapse by ensuring uniform utilization.
    """
    # Sum of probabilities across all tokens for each expert
    mass = routing_probs.sum(dim=(0, 2, 3)) # (N,)
    
    mean_mass = mass.mean()
    std_mass = mass.std(unbiased=False)
    
    # Squared Coefficient of Variation
    cv_squared = (std_mass / (mean_mass + 1e-9)) ** 2
    return cv_squared

class CombinedMoELoss(nn.Module):
    def __init__(self, moe_weight=0.01):
        super().__init__()
        # Saliency and Edge losses
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss() # Fallback if inputs are already probabilities
        self.iou_loss = IoULoss()
        self.moe_weight = moe_weight
        
    def forward(self, pred_saliency, pred_edge, target_saliency, target_edge, routing_probs_list):
        """
        pred_saliency: (B, 1, H, W)
        pred_edge: (B, 1, H, W)
        target_saliency: (B, 1, H, W)
        target_edge: (B, 1, H, W)
        routing_probs_list: list of tensors or a single tensor of shape (B, N, H, W)
        """
        
        # Determine if predictions are logits or probabilities
        # Safe check to seamlessly handle if the decoder used torch.sigmoid or not
        is_prob_sal = (pred_saliency.max() <= 1.0 and pred_saliency.min() >= 0.0)
        if is_prob_sal:
            loss_sal_bce = self.bce(pred_saliency.float(), target_saliency.float())
            pred_sal_prob = pred_saliency
        else:
            loss_sal_bce = self.bce_logits(pred_saliency, target_saliency)
            pred_sal_prob = torch.sigmoid(pred_saliency)
            
        is_prob_edge = (pred_edge.max() <= 1.0 and pred_edge.min() >= 0.0)
        if is_prob_edge:
            loss_edge_bce = self.bce(pred_edge.float(), target_edge.float())
        else:
            loss_edge_bce = self.bce_logits(pred_edge, target_edge)
            
        # IoU Loss (requires probabilities)
        loss_sal_iou = self.iou_loss(pred_sal_prob, target_saliency)
        
        # MoE Auxiliary Regularization Losses
        if isinstance(routing_probs_list, torch.Tensor):
            routing_probs_list = [routing_probs_list]
            
        lb_loss = torch.tensor(0.0, device=pred_saliency.device)
        imp_loss = torch.tensor(0.0, device=pred_saliency.device)
        
        for rp in routing_probs_list:
            lb_loss += load_balancing_loss(rp)
            imp_loss += importance_loss(rp)
            
        if len(routing_probs_list) > 0:
            lb_loss /= len(routing_probs_list)
            imp_loss /= len(routing_probs_list)
            
        # Total Loss formulation
        total_loss = loss_sal_bce + loss_sal_iou + loss_edge_bce + \
                     self.moe_weight * lb_loss + self.moe_weight * imp_loss
                     
        # Dictionary for TensorBoard/WandB logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'bce_saliency': loss_sal_bce.item(),
            'iou_saliency': loss_sal_iou.item(),
            'bce_edge': loss_edge_bce.item(),
            'moe_load_balancing': lb_loss.item(),
            'moe_importance': imp_loss.item()
        }
        
        return total_loss, loss_dict

if __name__ == '__main__':
    print("Testing CombinedMoELoss...")
    
    batch_size = 2
    H, W = 96, 96
    num_experts = 6
    
    # Dummy Logit Predictions
    dummy_pred_saliency = torch.randn(batch_size, 1, H, W, requires_grad=True)
    dummy_pred_edge = torch.randn(batch_size, 1, H, W, requires_grad=True)
    
    # Dummy Targets (0 or 1)
    dummy_target_saliency = torch.randint(0, 2, (batch_size, 1, H, W)).float()
    dummy_target_edge = torch.randint(0, 2, (batch_size, 1, H, W)).float()
    
    # Dummy Routing Probabilities (must sum to 1 across dim=1)
    raw_routing = torch.randn(batch_size, num_experts, H, W, requires_grad=True)
    dummy_routing_probs = F.softmax(raw_routing, dim=1)
    
    # Instantiate Loss
    criterion = CombinedMoELoss(moe_weight=0.01)
    
    # Forward Pass
    total_loss, metrics = criterion(
        dummy_pred_saliency, 
        dummy_pred_edge, 
        dummy_target_saliency, 
        dummy_target_edge, 
        [dummy_routing_probs]
    )
    
    print("\nLoss Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print(f"\nTotal Loss Tensor: {total_loss}")
    
    # Backward Pass Verification
    total_loss.backward()
    
    # Assert gradients exist before accessing .shape
    assert dummy_pred_saliency.grad is not None, "Gradient did not flow to Saliency Predictor!"
    assert dummy_pred_edge.grad is not None, "Gradient did not flow to Edge Predictor!"
    assert raw_routing.grad is not None, "Gradient did not flow to Routing Router!"
    
    print("\nGradient Verification:")
    print(f"  pred_saliency grad shape: {dummy_pred_saliency.grad.shape}")
    print(f"  pred_edge grad shape: {dummy_pred_edge.grad.shape}")
    print(f"  routing_probs grad shape: {raw_routing.grad.shape}")
    
    print("\nAll tests passed successfully! Gradients are flowing.")
