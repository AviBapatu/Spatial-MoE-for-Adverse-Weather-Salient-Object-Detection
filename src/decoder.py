import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialCrossAttention(nn.Module):
    """
    Cross-attention between two spatial feature maps.
    Queries come from q_x, Keys and Values come from kv_x.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
    def forward(self, q_x, kv_x):
        B, C, H, W = q_x.shape
        _, _, H_kv, W_kv = kv_x.shape
        
        # Flatten spatial dimensions
        q = q_x.flatten(2).transpose(1, 2)      # (B, H*W, C)
        kv = kv_x.flatten(2).transpose(1, 2)    # (B, H_kv*W_kv, C)
        
        # Apply layer normalization
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        
        # Cross Attention
        attn_out, _ = self.mha(q_norm, kv_norm, kv_norm)
        
        # Reshape back to spatial dimensions
        out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        
        # Residual connection
        return q_x + out

class EntropyConvBlock(nn.Module):
    """
    Concatenates the entropy map and applies a convolution block to merge them.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim + 1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, feat, entropy):
        # Concatenate entropy along channel dimension
        x = torch.cat([feat, entropy], dim=1)
        return self.conv(x)

class ProgressiveUpBlock(nn.Module):
    """
    Progressive upsampling block as specified:
    Conv 3x3 -> BatchNorm -> ReLU -> Bilinear Upsample
    """
    def __init__(self, in_dim, out_dim, scale_factor=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x

class SpatialMoEDecoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        
        # 1. Entropy concatenators for each level
        self.ent_16 = EntropyConvBlock(dim)
        self.ent_8 = EntropyConvBlock(dim)
        self.ent_4 = EntropyConvBlock(dim)
        
        # 2. Cross-Attention modules for top-down fusion
        self.cross_attn_8 = SpatialCrossAttention(dim)
        self.cross_attn_4 = SpatialCrossAttention(dim)
        
        # 3. Progressive Upsampling Heads (from 1/4 to 1/2, then 1/2 to 1/1)
        # We can gradually decrease dimensions if needed, or keep them. Let's step down: 256 -> 128 -> 64
        self.up_4to2 = ProgressiveUpBlock(dim, dim // 2)
        self.up_2to1 = ProgressiveUpBlock(dim // 2, dim // 4)
        
        # Final 1x1 convolutions for prediction maps
        final_dim = dim // 4
        self.saliency_head = nn.Conv2d(final_dim, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(final_dim, 1, kernel_size=1)
        
    def forward(self, Y_4, Y_8, Y_16, H_4, H_8, H_16):
        # Level 16 (1/16 scale)
        feat_16 = self.ent_16(Y_16, H_16)
        feat_16_up = F.interpolate(feat_16, scale_factor=2, mode='bilinear', align_corners=False) # -> 1/8 scale
        
        # Level 8 (1/8 scale)
        attn_8 = self.cross_attn_8(q_x=Y_8, kv_x=feat_16_up)
        feat_8 = self.ent_8(attn_8, H_8)
        feat_8_up = F.interpolate(feat_8, scale_factor=2, mode='bilinear', align_corners=False) # -> 1/4 scale
        
        # Level 4 (1/4 scale)
        attn_4 = self.cross_attn_4(q_x=Y_4, kv_x=feat_8_up)
        feat_4 = self.ent_4(attn_4, H_4)
        
        # Progressive Segmentation Decoder (from 1/4 scale back to original 1/1 scale)
        x = self.up_4to2(feat_4)  # -> 1/2 scale
        x = self.up_2to1(x)       # -> 1/1 scale
        
        # Output probability maps
        saliency_map = torch.sigmoid(self.saliency_head(x))
        edge_map = torch.sigmoid(self.edge_head(x))
        
        return saliency_map, edge_map

if __name__ == '__main__':
    print("Testing SpatialMoEDecoder...")
    
    batch_size = 2
    dim = 256
    
    # Original image size 384x384
    # Therefore scales are: 1/4 -> 96x96, 1/8 -> 48x48, 1/16 -> 24x24
    
    Y_4 = torch.randn(batch_size, dim, 96, 96)
    Y_8 = torch.randn(batch_size, dim, 48, 48)
    Y_16 = torch.randn(batch_size, dim, 24, 24)
    
    H_4 = torch.rand(batch_size, 1, 96, 96)
    H_8 = torch.rand(batch_size, 1, 48, 48)
    H_16 = torch.rand(batch_size, 1, 24, 24)
    
    print("Input shapes:")
    print(f"  Y_4: {Y_4.shape}, H_4: {H_4.shape}")
    print(f"  Y_8: {Y_8.shape}, H_8: {H_8.shape}")
    print(f"  Y_16: {Y_16.shape}, H_16: {H_16.shape}")
    
    decoder = SpatialMoEDecoder(dim=dim)
    
    # Forward pass
    saliency_map, edge_map = decoder(Y_4, Y_8, Y_16, H_4, H_8, H_16)
    
    print("\nOutput shapes:")
    print(f"  Saliency Map: {saliency_map.shape}")
    print(f"  Edge Map: {edge_map.shape}")
    
    # Validation against expected sizes
    expected_shape = (batch_size, 1, 384, 384)
    assert saliency_map.shape == expected_shape, f"Expected {expected_shape}, got {saliency_map.shape}"
    assert edge_map.shape == expected_shape, f"Expected {expected_shape}, got {edge_map.shape}"
    
    print("All shapes verified successfully!")
