import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

class MoEOutput(NamedTuple):
    features: torch.Tensor
    routing_probs: torch.Tensor
    topk_indices: torch.Tensor
    topk_gates: torch.Tensor
    entropy: torch.Tensor
    clean_logits: torch.Tensor

class TokenWiseMLPExpert(nn.Module):
    """
    Token-wise residual MLP expert:
    LayerNorm -> Linear(C -> 4C) -> GELU -> Linear(4C -> C) -> residual
    Input/Output shape: [N_tokens, C]
    """
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        z = x
        z = self.norm(z)
        z = self.fc1(z)
        z = self.act(z)
        z = self.fc2(z)
        return x + z

class SpatialMoELayer(nn.Module):
    def __init__(self, dim=256, num_experts=8, k=2, router_hidden=128, router_noise_enabled=True, router_noise_scale=1.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k
        self.router_noise_enabled = router_noise_enabled
        self.router_noise_scale = router_noise_scale
        
        # 1. Expert Pool
        self.experts = nn.ModuleList([
            TokenWiseMLPExpert(dim=dim) for _ in range(num_experts)
        ])
        
        # 2. Spatial Context Router
        self.router_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # MLP for clean routing logits
        self.router_mlp = nn.Sequential(
            nn.Linear(2 * dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts)
        )
        
        # Shazeer-style noisy routing linear projection
        self.noise_linear = nn.Linear(dim, num_experts)
        
    def forward(self, x, force_expert_id=None):
        B, C, H, W = x.shape
        N_tokens = H * W
        
        x_tokens = x.flatten(2).transpose(1, 2)       # [B, H*W, C]
        
        # --- Counterfactual Ablation Bypass ---
        if force_expert_id is not None:
            expert = self.experts[force_expert_id]
            flat_x = x_tokens.reshape(B * N_tokens, C)
            expert_out = expert(flat_x)
            fused_feature_map = expert_out.view(B, N_tokens, C).transpose(1, 2).view(B, C, H, W).contiguous()
            
            topk_indices = torch.full((B, N_tokens, self.k), force_expert_id, device=x.device, dtype=torch.long)
            topk_gates = torch.zeros(B, N_tokens, self.k, device=x.device, dtype=x.dtype)
            topk_gates[:, :, 0] = 1.0
            
            routing_probs = torch.zeros(B, self.num_experts, H, W, device=x.device, dtype=x.dtype)
            routing_probs[:, force_expert_id, :, :] = 1.0
            
            entropy_map = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
            
            clean_logits_spatial = torch.zeros(B, self.num_experts, H, W, device=x.device, dtype=x.dtype)
            clean_logits_spatial[:, force_expert_id, :, :] = 1.0
            
            return MoEOutput(
                features=fused_feature_map,
                routing_probs=routing_probs,
                topk_indices=topk_indices,
                topk_gates=topk_gates,
                entropy=entropy_map,
                clean_logits=clean_logits_spatial
            )
        
        # --- Routing Input ---
        local_feat = self.router_dwconv(x)
        local_tokens = local_feat.flatten(2).transpose(1, 2) # [B, H*W, C]
        router_input = torch.cat([x_tokens, local_tokens], dim=-1) # [B, H*W, 2C]
        
        # --- Routing ---
        clean_logits = self.router_mlp(router_input) # [B, H*W, E]
        
        if self.training and self.router_noise_enabled:
            noise_std = self.router_noise_scale * F.softplus(self.noise_linear(x_tokens))
            noise = torch.randn_like(clean_logits) * noise_std
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
            
        topk_values, topk_indices = torch.topk(noisy_logits, k=self.k, dim=-1) # [B, H*W, K]
        topk_gates = F.softmax(topk_values, dim=-1) # [B, H*W, K]
        
        # --- True Sparse Dispatch ---
        flat_x = x_tokens.reshape(B * N_tokens, C)
        flat_topk_indices = topk_indices.reshape(B * N_tokens, self.k)
        flat_topk_gates = topk_gates.reshape(B * N_tokens, self.k)
        
        output_tokens = torch.zeros_like(flat_x)
        
        for i, expert in enumerate(self.experts):
            # Find tokens that selected this expert
            active_mask = (flat_topk_indices == i) # [B*N_tokens, K]
            token_active = active_mask.any(dim=-1) # [B*N_tokens] boolean
            
            if not token_active.any():
                continue
                
            selected_tokens = flat_x[token_active] # [N_active, C]
            
            # Execute expert
            expert_out = expert(selected_tokens) # [N_active, C]
            
            # Find which gate (0 to K-1) applies to each active token
            expert_gates = flat_topk_gates[token_active][active_mask[token_active]] # [N_active]
            
            weighted_out = expert_out * expert_gates.unsqueeze(-1)
            
            # Scatter add back to output
            output_tokens[token_active] += weighted_out

        fused_feature_map = output_tokens.view(B, N_tokens, C).transpose(1, 2).view(B, C, H, W).contiguous()

        # --- Diagnostics / Reconstruction ---
        # Reconstruct full routing probs [B, E, H, W] for auxiliary losses
        zeros = torch.zeros_like(noisy_logits, dtype=topk_gates.dtype)
        routing_gates = zeros.scatter(-1, topk_indices, topk_gates)
        routing_probs = routing_gates.transpose(1, 2).view(B, self.num_experts, H, W).contiguous()
        
        # Entropy
        entropy = -torch.sum(topk_gates * torch.log(topk_gates + 1e-9), dim=-1)
        entropy_map = entropy.view(B, 1, H, W).contiguous()
        
        # Reshape logits as [B, E, H, W] for consistency if needed, or leave flat.
        # We will reshape for consistency with other spatial maps
        clean_logits_spatial = clean_logits.transpose(1, 2).view(B, self.num_experts, H, W).contiguous()

        return MoEOutput(
            features=fused_feature_map,
            routing_probs=routing_probs,
            topk_indices=topk_indices,
            topk_gates=topk_gates,
            entropy=entropy_map,
            clean_logits=clean_logits_spatial
        )

if __name__ == '__main__':
    # Test the SpatialMoELayer
    print("Testing SpatialMoELayer...")
    
    batch_size = 2
    channels = 256
    H, W = 48, 48
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, channels, H, W)
    print(f"Input shape: {dummy_input.shape}")
    
    # Initialize MoE layer
    moe_layer = SpatialMoELayer(dim=channels, num_experts=8, k=2)
    
    # Forward pass
    out = moe_layer(dummy_input)
    
    print("\nOutput shapes:")
    print(f"  Fused features: {out.features.shape}")
    print(f"  Routing probabilities: {out.routing_probs.shape}")
    print(f"  TopK indices: {out.topk_indices.shape}")
    print(f"  TopK gates: {out.topk_gates.shape}")
    print(f"  Entropy map: {out.entropy.shape}")
    print(f"  Clean logits: {out.clean_logits.shape}")
    
    print("Basic execution successful!")
