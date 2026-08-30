import torch
import torch.nn as nn
import timm

class MultiScaleBackbone(nn.Module):
    def __init__(self, model_name='pvt_v2_b4', pretrained=True, d=256):
        super().__init__()
        
        # Instantiate hierarchical backbone using timm
        # We extract features at out_indices=(0, 1, 2) which correspond to 1/4, 1/8, 1/16 scales in PVT
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(0, 1, 2)
        )
        
        # Get the channel dimensions of the extracted features
        feature_channels = self.backbone.feature_info.channels()
        
        # Add 1x1 projection layers to map to unified dimension `d`
        self.proj_4 = nn.Conv2d(feature_channels[0], d, kernel_size=1)
        self.proj_8 = nn.Conv2d(feature_channels[1], d, kernel_size=1)
        self.proj_16 = nn.Conv2d(feature_channels[2], d, kernel_size=1)
        
    def forward(self, x):
        # Extract hierarchical features
        feats = self.backbone(x)
        
        # Project each scale to dimension `d`
        res_4 = self.proj_4(feats[0])
        res_8 = self.proj_8(feats[1])
        res_16 = self.proj_16(feats[2])
        
        return {
            'res_4': res_4,
            'res_8': res_8,
            'res_16': res_16
        }

if __name__ == '__main__':
    print("Testing MultiScaleBackbone...")
    
    # Instantiate backbone with dimension d=256
    model = MultiScaleBackbone(model_name='pvt_v2_b4', pretrained=True, d=256)
    
    # Create dummy tensor of shape (1, 3, 384, 384)
    dummy_input = torch.randn(1, 3, 384, 384)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Feed to model
    outputs = model(dummy_input)
    
    # Verify outputs
    print("\nOutput shapes:")
    print(f"  res_4: {outputs['res_4'].shape}")
    print(f"  res_8: {outputs['res_8'].shape}")
    print(f"  res_16: {outputs['res_16'].shape}")
    
    # Validate against expected shapes
    assert outputs['res_4'].shape == (1, 256, 96, 96), f"Expected (1, 256, 96, 96), got {outputs['res_4'].shape}"
    assert outputs['res_8'].shape == (1, 256, 48, 48), f"Expected (1, 256, 48, 48), got {outputs['res_8'].shape}"
    assert outputs['res_16'].shape == (1, 256, 24, 24), f"Expected (1, 256, 24, 24), got {outputs['res_16'].shape}"
    
    print("\nAll shapes verified successfully!")
