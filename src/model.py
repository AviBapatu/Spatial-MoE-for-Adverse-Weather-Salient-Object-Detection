import torch
import torch.nn as nn
from .backbone import MultiScaleBackbone
from .moe_layer import SpatialMoELayer
from .decoder import SpatialMoEDecoder, DecoderOutput
class SpatialMoESODNet(nn.Module):
    def __init__(self, dim=256, num_experts=6, k=2, window_size=8, use_deep_supervision=True):
        super().__init__()
        
        # 1. Hierarchical Multi-Scale Backbone
        self.backbone = MultiScaleBackbone(d=dim)
        
        # 2. Independent Spatial MoE Layers for each scale
        self.moe_4 = SpatialMoELayer(dim=dim, num_experts=num_experts, k=k)
        self.moe_8 = SpatialMoELayer(dim=dim, num_experts=num_experts, k=k)
        self.moe_16 = SpatialMoELayer(dim=dim, num_experts=num_experts, k=k)
        
        # 3. Spatial MoE Decoder for feature fusion and prediction
        self.decoder = SpatialMoEDecoder(
            dim=dim, 
            window_size=window_size, 
            use_deep_supervision=use_deep_supervision
        )
        
    def forward(self, x, ablation_cfg=None):
        # Backbone feature extraction
        feats = self.backbone(x)
        res_4, res_8, res_16 = feats['res_4'], feats['res_8'], feats['res_16']
        
        # Counterfactual ablation setup
        force_4, force_8, force_16 = None, None, None
        if ablation_cfg is not None:
            if ablation_cfg.get('scale') == 4: force_4 = ablation_cfg.get('expert_id')
            if ablation_cfg.get('scale') == 8: force_8 = ablation_cfg.get('expert_id')
            if ablation_cfg.get('scale') == 16: force_16 = ablation_cfg.get('expert_id')

        # Pass intermediate features through independent MoE layers
        out_4 = self.moe_4(res_4, force_expert_id=force_4)
        out_8 = self.moe_8(res_8, force_expert_id=force_8)
        out_16 = self.moe_16(res_16, force_expert_id=force_16)
        
        # Fuse routed features in Decoder
        decoder_out = self.decoder(
            out_4.features, out_8.features, out_16.features, 
            out_4.entropy, out_8.entropy, out_16.entropy
        )
        
        return decoder_out, [out_4, out_8, out_16]

if __name__ == '__main__':
    print("Testing SpatialMoESODNet end-to-end...")
    model = SpatialMoESODNet(dim=256)
    
    dummy_input = torch.randn(1, 3, 384, 384)
    print(f"Input image shape: {dummy_input.shape}")
    
    decoder_out, moe_outputs = model(dummy_input)
    
    print("\nOutput shapes:")
    print(f"  Saliency Logits: {decoder_out.saliency_logits.shape}")
    print(f"  Edge Logits: {decoder_out.boundary_logits.shape}")
    print(f"  Aux 16: {decoder_out.aux_logits_16.shape}")
    print(f"  Aux 8: {decoder_out.aux_logits_8.shape}")
    print(f"  Aux 4: {decoder_out.aux_logits_4.shape}")
    for i, out in enumerate(moe_outputs):
        print(f"  Routing Prob Level {i+1}: {out.routing_probs.shape}")
        
    assert decoder_out.saliency_logits.shape == (1, 1, 384, 384)
    assert decoder_out.boundary_logits.shape == (1, 1, 384, 384)
    assert decoder_out.aux_logits_16.shape == (1, 1, 24, 24)
    assert decoder_out.aux_logits_8.shape == (1, 1, 48, 48)
    assert decoder_out.aux_logits_4.shape == (1, 1, 96, 96)
    print("All integration tests passed!")
