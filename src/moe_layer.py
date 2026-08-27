import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightExpert(nn.Module):
    """
    Residual inverted bottleneck block as specified:
    Depthwise Conv 3x3 -> Pointwise Conv 1x1 -> GELU -> Pointwise Conv 1x1
    """
    def __init__(self, dim, expansion=4):
        super().__init__()
        # Depthwise Conv 3x3
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # Pointwise Conv 1x1 (Expansion)
        self.pwconv1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        # GELU activation
        self.act = nn.GELU()
        # Pointwise Conv 1x1 (Projection)
        self.pwconv2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return res + x

class SpatialMoELayer(nn.Module):
    def __init__(self, dim, num_experts=6, k=2, mlp_hidden=128):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k
        
        # 1. Expert Pool
        self.experts = nn.ModuleList([
            LightweightExpert(dim=dim) for _ in range(num_experts)
        ])
        
        # 2. Spatial Context Router
        self.router_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # MLP for clean routing logits
        self.router_mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_experts)
        )
        
        # Shazeer-style noisy routing
        self.noise_linear = nn.Linear(dim, num_experts)
        
    def forward(self, x):
        B, C, H, W = x.shape
        N_tokens = H * W
        
        # --- Routing ---
        # Capture local texture statistics
        router_ctx = self.router_dwconv(x)
        
        # Flatten spatial dimensions into tokens: (B, C, H, W) -> (B, H*W, C)
        tokens = router_ctx.flatten(2).transpose(1, 2)
        
        # 2-layer MLP to obtain raw routing logits
        clean_logits = self.router_mlp(tokens)
        
        # 3. Noisy Top-k Gating (Shazeer-style)
        if self.training:
            noise_std = F.softplus(self.noise_linear(tokens))
            noise = torch.randn_like(clean_logits) * noise_std
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
            
        # Top-k selection
        topk_logits, topk_indices = torch.topk(noisy_logits, k=self.k, dim=-1)
        
        # Softmax over the Top-k active experts
        topk_gates = F.softmax(topk_logits, dim=-1)
        
        # Scatter gates back to full expert dimension (B, N_tokens, N_experts)
        zeros = torch.zeros_like(noisy_logits, dtype=topk_gates.dtype)
        routing_gates = zeros.scatter(-1, topk_indices, topk_gates)
        
        # --- Token Dispatch & Reconstruction ---
        # Reshape to spatial format for applying on feature maps: (B, N_experts, H, W)
        routing_gates_spatial = routing_gates.transpose(1, 2).view(B, self.num_experts, H, W)
        
        fused_feature_map = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            gate_i = routing_gates_spatial[:, i:i+1, :, :] # (B, 1, H, W)
            
            # Optimization: Only compute expert if it's active for at least one token in the batch
            if gate_i.sum() > 0:
                expert_out = expert(x)
                fused_feature_map += gate_i * expert_out
                
        # --- Entropy & Stats ---
        # Compute token-level routing entropy H_i = -sum(p * log(p))
        # We compute entropy only over the top-k selected probabilities to avoid log(0) on non-selected
        entropy = -torch.sum(topk_gates * torch.log(topk_gates + 1e-9), dim=-1)
        
        # Reshape to (B, 1, H, W)
        entropy_map = entropy.view(B, 1, H, W)
        
        return fused_feature_map, routing_gates_spatial, entropy_map

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
    moe_layer = SpatialMoELayer(dim=channels, num_experts=6, k=2)
    
    # Forward pass
    fused_features, routing_probs, entropy_map = moe_layer(dummy_input)
    
    print("\nOutput shapes:")
    print(f"  Fused features: {fused_features.shape}")
    print(f"  Routing probabilities: {routing_probs.shape}")
    print(f"  Entropy map: {entropy_map.shape}")
    
    # Verifications
    assert fused_features.shape == (batch_size, channels, H, W), "Fused feature shape mismatch"
    assert routing_probs.shape == (batch_size, 6, H, W), "Routing probability shape mismatch"
    assert entropy_map.shape == (batch_size, 1, H, W), "Entropy map shape mismatch"
    
    # Verify gate normalization (sum over experts for any token should equal 1.0)
    gate_sums = routing_probs.sum(dim=1) # Sum across experts -> (B, H, W)
    max_dev = torch.max(torch.abs(gate_sums - 1.0))
    print(f"\nMax deviation from gate sum=1.0: {max_dev.item():.6f}")
    assert max_dev < 1e-5, "Gates are not properly normalized to 1 across active experts!"
    
    print("All tests passed successfully!")
