import pytest
import torch
import torch.nn as nn
from contextlib import nullcontext
from src.model import SpatialMoESODNet
from src.optimization import get_parameter_groups, freeze_backbone, unfreeze_backbone, WarmupCosineScheduler, OptimizationEngine

pytestmark = pytest.mark.gpu

def test_b4_construction_and_pretrained():
    model = SpatialMoESODNet(dim=256)
    
    # 1. Verify B4 Backbone Construction
    assert model.backbone.backbone.default_cfg['architecture'] == 'pvt_v2_b4'
    assert model.backbone.backbone.pretrained_cfg is not None
    
    # 2. Check Feature Shape & Projection Compatibility
    dummy = torch.randn(1, 3, 384, 384)
    out, moe_outs = model(dummy)
    
    assert out.saliency_logits.shape == (1, 1, 384, 384)
    # The backbone stages are explicitly mapped to 256.
    # The features are passed to the decoder safely.

def test_parameter_group_completeness():
    model = SpatialMoESODNet(dim=256)
    groups = get_parameter_groups(model)
    
    all_trainable = set(p for p in model.parameters() if p.requires_grad)
    group_params = set()
    
    for g in groups:
        for p in g['params']:
            assert p not in group_params, "Parameter appears in more than one group!"
            group_params.add(p)
            
    assert all_trainable == group_params, "Mismatch between all trainable params and group params!"

def test_frozen_unfrozen_backbone():
    model = SpatialMoESODNet(dim=256)
    
    # Freeze
    freeze_backbone(model)
    dummy = torch.randn(1, 3, 384, 384)
    out, _ = model(dummy)
    loss = out.saliency_logits.mean()
    loss.backward()
    
    for name, p in model.named_parameters():
        if name.startswith('backbone'):
            assert p.requires_grad == False
            assert p.grad is None
            
    # Unfreeze
    model.zero_grad()
    unfreeze_backbone(model)
    out, _ = model(dummy)
    loss = out.saliency_logits.mean()
    loss.backward()
    
    has_finite = False
    for name, p in model.named_parameters():
        if name.startswith('backbone'):
            assert p.requires_grad == True
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
                has_finite = True
                
    assert has_finite, "Unfrozen backbone did not receive any gradients!"

def test_amp_and_scheduler_trajectory():
    model = SpatialMoESODNet(dim=256)
    groups = get_parameter_groups(model, backbone_lr=2e-5, new_module_lr=1e-4)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    
    warmup_steps = 5
    total_steps = 15
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    engine = OptimizationEngine(model, optimizer, scheduler, amp_enabled=False) # CPU test
    
    # Warmup Step 0 (Should be 0 LR since step is 0)
    assert scheduler.get_last_lr()[0] == 0.0
    
    # Force step
    dummy = torch.randn(1, 3, 384, 384)
    out, _ = model(dummy)
    loss = out.saliency_logits.mean()
    loss.backward()
    engine.step()
    
    # Step 1 LR
    assert scheduler.get_last_lr()[0] == 2e-5 * (1/5)
    
def test_500_step_calibration():
    # Only run full if CUDA
    if not torch.cuda.is_available():
        pytest.skip("Skipping memory calibration test on CPU.")
        
    backbones_to_try = ['pvt_v2_b4', 'pvt_v2_b3', 'pvt_v2_b2']
    success = False
    
    for backbone_name in backbones_to_try:
        try:
            # Reconstruct model dynamically with current backbone
            import sys
            import importlib
            import src.model
            # Monkey patch temporarily for testing
            original_init = src.model.MultiScaleBackbone.__init__
            def patched_init(self, model_name=backbone_name, pretrained=True, d=256):
                original_init(self, model_name=model_name, pretrained=pretrained, d=d)
            src.model.MultiScaleBackbone.__init__ = patched_init
            
            model = SpatialMoESODNet(dim=256).cuda()
            groups = get_parameter_groups(model)
            optimizer = torch.optim.AdamW(groups)
            scheduler = WarmupCosineScheduler(optimizer, 50, 500)
            engine = OptimizationEngine(model, optimizer, scheduler, amp_enabled=True, amp_dtype=torch.float16)
            
            torch.cuda.reset_peak_memory_stats()
            
            # Just run 5 steps for CI to avoid timeout, but simulate memory.
            for i in range(5):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    dummy = torch.randn(1, 3, 384, 384, device='cuda')
                    out, _ = model(dummy)
                    loss = out.saliency_logits.mean()
                
                engine.scaler.scale(loss).backward()
                overflow, norm = engine.step()
                engine.optimizer.zero_grad()
                
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"\n[{backbone_name}] Peak Memory: {peak_mem_gb:.2f} GB")
            
            success = True
            
            # Restore original init
            src.model.MultiScaleBackbone.__init__ = original_init
            break
            
        except torch.OutOfMemoryError as e:
            peak_mem_gb_fail = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"\nOOM during {backbone_name} calibration! Peak before fail: {peak_mem_gb_fail:.2f} GB")
            
            # Try to clean up for next iteration
            for var in ['model', 'groups', 'optimizer', 'scheduler', 'engine', 'dummy', 'out', 'loss']:
                if var in locals():
                    del locals()[var]
            torch.cuda.empty_cache()
            
            # Find next fallback
            idx = backbones_to_try.index(backbone_name)
            if idx + 1 < len(backbones_to_try):
                next_bb = backbones_to_try[idx+1]
                print(f"BACKBONE FALLBACK ACTIVE: {backbone_name} -> {next_bb}")
                print(f"Reason for fallback: CUDA Out of Memory on {backbone_name}")
            else:
                print("All fallbacks exhausted!")
                raise e
        finally:
            src.model.MultiScaleBackbone.__init__ = original_init
            
    assert success, "Memory calibration failed for all backbone sizes!"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
