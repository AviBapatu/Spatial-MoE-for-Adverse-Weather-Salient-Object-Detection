import torch
import torch.nn as nn
from .backbone import MultiScaleBackbone
from .moe_layer import SpatialMoELayer
from .decoder import SpatialMoEDecoder

class SpatialMoESODNet(nn.Module):
    def __init__(self, dim=256, num_experts=6, k=2):
        super().__init__()
        
        # 1. Hierarchical Multi-Scale Backbone
        self.backbone = MultiScaleBackbone(d=dim)
        
        # 2. Independent Spatial MoE Layers for each scale
        self.moe_4 = SpatialMoELayer(dim=dim, num_experts=num_experts, k=k)
        self.moe_8 = SpatialMoELayer(dim=dim, num_experts=num_experts, k=k)
        self.moe_16 = SpatialMoELayer(dim=dim, num_experts=num_experts, k=k)
        
        # 3. Spatial MoE Decoder for feature fusion and prediction
        self.decoder = SpatialMoEDecoder(dim=dim)
        
    def forward(self, x):
        # Backbone feature extraction
        feats = self.backbone(x)
        res_4, res_8, res_16 = feats['res_4'], feats['res_8'], feats['res_16']
        
        # Pass intermediate features through independent MoE layers
        Y_4, P_4, H_4 = self.moe_4(res_4)
        Y_8, P_8, H_8 = self.moe_8(res_8)
        Y_16, P_16, H_16 = self.moe_16(res_16)
        
        # Fuse routed features in Decoder
        saliency_map, edge_map = self.decoder(Y_4, Y_8, Y_16, H_4, H_8, H_16)
        
        return saliency_map, edge_map, [P_4, P_8, P_16]

if __name__ == '__main__':
    print("Testing SpatialMoESODNet end-to-end...")
    model = SpatialMoESODNet(dim=256)
    
    dummy_input = torch.randn(1, 3, 384, 384)
    print(f"Input image shape: {dummy_input.shape}")
    
    sal, edge, routing_probs = model(dummy_input)
    
    print("\nOutput shapes:")
    print(f"  Saliency: {sal.shape}")
    print(f"  Edge Map: {edge.shape}")
    for i, p in enumerate(routing_probs):
        print(f"  Routing Prob Level {i+1}: {p.shape}")
        
    assert sal.shape == (1, 1, 384, 384)
    assert edge.shape == (1, 1, 384, 384)
    print("All integration tests passed!")
