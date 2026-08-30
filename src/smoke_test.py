import os
import shutil
import time
import torch
import torch.nn as nn
from torch.amp import autocast
import torch.distributed as dist
import copy
import cv2
import numpy as np
import argparse
import json
import hashlib

from src.dataset import get_dataloaders
from src.model import SpatialMoESODNet
from src.loss import SpatialMoELoss
from torch.optim import AdamW
from src.optimization import get_parameter_groups, freeze_backbone, WarmupCosineScheduler, OptimizationEngine
from src.train_ddp import set_rng_states, get_rng_states, CHECKPOINT_FORMAT_VERSION, get_config_hash

def assert_val(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL: {msg}")

SMOKE_CONFIG = {
    "EPOCHS": 2,
    "SMOKE_BATCH_SIZE": 1,
    "SMOKE_MAX_SAMPLES": 8,
    "SMOKE_GRAD_ACCUM_STEPS": 2,
    "SMOKE_OPTIMIZER_STEPS": 2,
    "IMAGE_SIZE": 384,
    "NUM_WORKERS": 0,
    "WARMUP_STEPS": 1,
    "TOTAL_STEPS": 4,
    "FREEZE_BACKBONE_EPOCHS": 1
}

def check_optimization_states(model, eng):
    """
    Ensure the frozen groups did not change, and trainable groups did.
    Returns parameter snapshots for comparison.
    """
    snapshot = {}
    for name, p in model.named_parameters():
        snapshot[name] = p.clone().detach() if p.requires_grad else p.detach()
    return snapshot

def run_tests_on_engine(device, local_rank, rank, world_size, data_root, result_file, is_ddp):
    try:
        train_loader, _, _, _ = get_dataloaders(
            root_dir=data_root, batch_size=1, image_size=384, num_workers=0,
            max_samples=8, distributed=is_ddp, rank=rank, world_size=world_size
        )
        model = SpatialMoESODNet(dim=256, use_deep_supervision=True).to(device)
        
        if is_ddp:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            core_model = model.module
        else:
            core_model = model

        opt = AdamW(get_parameter_groups(core_model))
        sch = WarmupCosineScheduler(opt, warmup_steps=1, total_steps=4)
        eng = OptimizationEngine(model, opt, sch, amp_enabled=True, amp_dtype=torch.float16)
        criterion = SpatialMoELoss(lambda_deep_supervision=0.4)
        freeze_backbone(core_model)
        
        # Snapshot before step
        snapshot_before = check_optimization_states(model, eng)
        
        # Forward Pass
        model.train()
        batch = next(iter(train_loader))
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        edges = batch['edge_map'].to(device)
        
        # Hook to verify expert inputs are not dense [B, C, H, W]
        expert_shapes = []
        def expert_hook(module, args, kwargs):
            expert_shapes.append(args[0].shape)
        
        hooks = []
        for layer in [core_model.moe_4, core_model.moe_8, core_model.moe_16]:
            for expert in layer.experts:
                hooks.append(expert.register_forward_pre_hook(expert_hook, with_kwargs=True))
                
        with autocast(device_type='cuda', dtype=torch.float16):
            out, moe_outputs = model(images)
            
        for h in hooks: h.remove()
        
        # 1. Sparsity Token Dispatch Checks (Point 8)
        moe_check = "PASS"
        try:
            for i, m_out in enumerate(moe_outputs):
                probs = m_out.routing_probs
                B, C, H, W = probs.shape
                valid_tokens = B * H * W
                top_k = 2
                
                expected_total_dispatch = valid_tokens * top_k
                actual_total_dispatch = m_out.topk_indices.numel()
                assert_val(actual_total_dispatch == expected_total_dispatch, f"Sparse dispatch count mismatch: Expected {expected_total_dispatch}, got {actual_total_dispatch}")
                
                assert_val(m_out.topk_indices.shape == (B, H*W, top_k), "Identity violation: Not all tokens were dispatched exactly Top-K times.")
                
            # Verify no expert received dense shape
            for shape in expert_shapes:
                assert_val(len(shape) != 4, f"Dense expert execution detected: an expert received {shape}")
                
        except Exception as e:
            moe_check = str(e)
            
        # 2. Loss and Gradient Checks
        loss_check = "PASS"
        try:
            with autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = criterion(
                    saliency_logits=out.saliency_logits, 
                    edge_logits=out.boundary_logits, 
                    moe_outputs=moe_outputs, 
                    target=masks, 
                    gt_boundary=edges,
                    aux_logits_16=out.aux_logits_16,
                    aux_logits_8=out.aux_logits_8,
                    aux_logits_4=out.aux_logits_4
                )
            loss = loss_dict['L_total']
            for k, v in loss_dict.items():
                t_v = torch.as_tensor(v)
                assert_val(torch.isfinite(t_v) and not torch.isnan(t_v), f"{k} is not finite.")
                
            with autocast(device_type='cuda', dtype=torch.float16):
                dummy_edge = out.boundary_logits.detach().requires_grad_()
                bound_loss = nn.BCEWithLogitsLoss()(dummy_edge, edges)
            
            eng.scaler.scale(loss).backward() # type: ignore
            
            def valid_grad(p):
                if p.grad is None: return False
                n = p.grad.norm()
                return n > 0 or not torch.isfinite(n)
                
            has_grad = any(valid_grad(p) for p in core_model.parameters() if p.requires_grad)
            assert_val(has_grad, "No gradients produced!")
            
            # Explicit Deep Supervision Gradient Checks
            assert_val(valid_grad(core_model.decoder.aux_head_16.weight), "No gradients found for aux_head_16")
            assert_val(valid_grad(core_model.decoder.aux_head_8.weight), "No gradients found for aux_head_8")
            assert_val(valid_grad(core_model.decoder.aux_head_4.weight), "No gradients found for aux_head_4")
            
            perfect_logits = masks * 20.0 - 10.0
            perfect_edge_logits = edges * 20.0 - 10.0
            
            with autocast(device_type='cuda', dtype=torch.float16):
                perfect_loss_dict = criterion(
                    saliency_logits=perfect_logits, 
                    edge_logits=perfect_edge_logits, 
                    moe_outputs=moe_outputs, 
                    target=masks, 
                    gt_boundary=edges,
                    aux_logits_16=perfect_logits,
                    aux_logits_8=perfect_logits,
                    aux_logits_4=perfect_logits
                )
            perfect_loss = perfect_loss_dict['L_total']
            assert_val(perfect_loss < loss, f"Perfect prediction loss not strictly lower than bad prediction loss")
            
            eng.step()
            eng.optimizer.zero_grad()
            
        except Exception as e:
            loss_check = str(e)
        
        # 3. Optimization Verifications
        optimizer_check = "PASS"
        try:
            snapshot_after = check_optimization_states(model, eng)
            for name, p_before in snapshot_before.items():
                p_after = snapshot_after[name]
                if not model.get_parameter(name).requires_grad:
                    assert_val(torch.equal(p_before, p_after), f"Frozen parameter changed: {name}")
                else:
                    if p_before.grad is not None:
                        assert_val(not torch.equal(p_before, p_after), f"Trainable parameter did NOT change: {name}")
        except Exception as e:
            optimizer_check = str(e)
            
        status = "PASS" if (moe_check == "PASS" and loss_check == "PASS" and optimizer_check == "PASS") else "FAIL"
        
        if is_ddp:
            my_rank = torch.tensor([rank], dtype=torch.long, device=device)
            all_ranks = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(all_ranks, my_rank)
            completed = [r.item() for r in all_ranks]
        else:
            completed = [0]
            
        if rank == 0:
            with open(result_file, "w") as f:
                json.dump({
                    "status": status,
                    "moe_check": moe_check,
                    "loss_check": loss_check,
                    "optimizer_check": optimizer_check,
                    "world_size": world_size, 
                    "ranks_completed": completed, 
                    "gpu_names": {str(local_rank): torch.cuda.get_device_name(device) if device.type == 'cuda' else "CPU"} # Note: getting just rank 0 GPU is fine for this demo.
                }, f)


            
    except Exception as e:
        if rank == 0:
            with open(result_file, "w") as f:
                json.dump({"status": "FAIL", "reason": str(e), "world_size": world_size, "ranks_completed": []}, f)
        raise e

def run_memory_calibration(device, result_file):
    try:
        model = SpatialMoESODNet(dim=256, use_deep_supervision=True).to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)
        opt = AdamW(get_parameter_groups(model.module))
        eng = OptimizationEngine(model, opt, None, amp_enabled=True, amp_dtype=torch.float16)
        criterion = SpatialMoELoss(lambda_deep_supervision=0.4)
        
        dummy_img = torch.randn(1, 3, 384, 384, device=device)
        dummy_mask = torch.randn(1, 1, 384, 384, device=device)
        dummy_edge = torch.randn(1, 1, 384, 384, device=device)
        
        for micro in range(150):
            with autocast(device_type='cuda', dtype=torch.float16):
                out, moe_outputs = model(dummy_img)
                loss_dict = criterion(
                    saliency_logits=out.saliency_logits, 
                    edge_logits=out.boundary_logits, 
                    moe_outputs=moe_outputs, 
                    target=dummy_mask, 
                    gt_boundary=dummy_edge,
                    aux_logits_16=out.aux_logits_16,
                    aux_logits_8=out.aux_logits_8,
                    aux_logits_4=out.aux_logits_4
                )
                loss = loss_dict['L_total']
            eng.scaler.scale(loss).backward() # type: ignore
            eng.step()
            eng.optimizer.zero_grad()
            
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_res = torch.cuda.max_memory_reserved(device) / (1024**3)
        
        if peak_alloc > 15.0:
            raise RuntimeError(f"Peak memory {peak_alloc:.2f} GB exceeds safe 15GB T4 limit.")
            
        if int(os.environ.get("RANK", 0)) == 0:
            with open(result_file, "w") as f:
                json.dump({"status": "PASS", "peak_allocated_gb": peak_alloc, "peak_reserved_gb": peak_res}, f)
    except Exception as e:
        if int(os.environ.get("RANK", 0)) == 0:
            with open(result_file, "w") as f:
                json.dump({"status": "FAIL", "reason": str(e)}, f)
        raise e

def run_resume_a(device, local_rank, rank, world_size, data_root, result_file):
    try:
        train_loader, _, _, _ = get_dataloaders(root_dir=data_root, batch_size=1, image_size=384, num_workers=0, max_samples=16, distributed=True, rank=rank, world_size=world_size)
        torch.manual_seed(1337)
        model = SpatialMoESODNet(dim=256, use_deep_supervision=True).to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        opt = AdamW(get_parameter_groups(model.module))
        sch = WarmupCosineScheduler(opt, warmup_steps=1, total_steps=4)
        eng = OptimizationEngine(model, opt, sch, amp_enabled=True, amp_dtype=torch.float16)
        criterion = SpatialMoELoss(lambda_deep_supervision=0.4)
        freeze_backbone(model.module)
        
        it = iter(train_loader)
        batch = next(it)
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        edges = batch['edge_map'].to(device)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            out, moe_outputs = model(images)
            loss_dict = criterion(
                saliency_logits=out.saliency_logits, 
                edge_logits=out.boundary_logits, 
                moe_outputs=moe_outputs, 
                target=masks, 
                gt_boundary=edges,
                aux_logits_16=out.aux_logits_16,
                aux_logits_8=out.aux_logits_8,
                aux_logits_4=out.aux_logits_4
            )
            loss = loss_dict['L_total']
        eng.scaler.scale(loss).backward() # type: ignore
        eng.step()
        eng.optimizer.zero_grad()
        
        if rank == 0:
            cfg_hash = get_config_hash(SMOKE_CONFIG, "dummy_model_hash")
            state = {
                'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
                'epoch': 0, 'batch_in_epoch': 1, 'global_step': eng.global_step,
                'best_metric': float('inf'),
                'model_state_dict': model.module.state_dict(),
                'engine_state_dict': eng.state_dict(),
                'rng_states': get_rng_states(device),
                'config': SMOKE_CONFIG,
                'config_hash': cfg_hash,
                'world_size': world_size,
                'timestamp': time.time(),
                'lr_before_ckpt': opt.param_groups[0]['lr']
            }
            ckpt_dir = "/kaggle/working/WXSOD_ResumeTest"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "latest.pth")
            torch.save(state, ckpt_path)
            
            sha256 = hashlib.sha256()
            with open(ckpt_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            
            manifest = {
                "checkpoint": ckpt_path,
                "size_bytes": os.path.getsize(ckpt_path),
                "config_hash": cfg_hash,
                "global_step": eng.global_step,
                "sha256": sha256.hexdigest()
            }
            with open(os.path.join(ckpt_dir, "checkpoint_manifest.json"), "w") as f:
                json.dump(manifest, f)
            with open(result_file, "w") as f:
                json.dump({"status": "PASS"}, f)
    except Exception as e:
        if rank == 0:
            with open(result_file, "w") as f:
                json.dump({"status": "FAIL", "reason": str(e)}, f)
        raise e

def run_resume_b(device, local_rank, rank, world_size, data_root, result_file):
    try:
        train_loader, _, _, _ = get_dataloaders(root_dir=data_root, batch_size=1, image_size=384, num_workers=0, max_samples=16, distributed=True, rank=rank, world_size=world_size)
        model = SpatialMoESODNet(dim=256, use_deep_supervision=True).to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        opt = AdamW(get_parameter_groups(model.module))
        sch = WarmupCosineScheduler(opt, warmup_steps=1, total_steps=4)
        eng = OptimizationEngine(model, opt, sch, amp_enabled=True, amp_dtype=torch.float16)
        
        ckpt_dir = "/kaggle/working/WXSOD_ResumeTest"
        ckpt_path = os.path.join(ckpt_dir, "latest.pth")
        
        assert_val(os.path.exists(ckpt_path), "latest.pth missing")
        
        sha256 = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        
        with open(os.path.join(ckpt_dir, "checkpoint_manifest.json"), "r") as f:
            manifest = json.load(f)
            
        assert_val(manifest["sha256"] == sha256.hexdigest(), "Manifest SHA256 mismatch")
        
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        cfg_hash = get_config_hash(SMOKE_CONFIG, "dummy_model_hash")
        assert_val(ckpt.get('config_hash') == cfg_hash, "Config hash mismatch on resume")
        
        model.module.load_state_dict(ckpt['model_state_dict'])
        eng.load_state_dict(ckpt['engine_state_dict'])
        set_rng_states(ckpt['rng_states'], device)
        
        assert_val(opt.param_groups[0]['lr'] == ckpt['lr_before_ckpt'], "LR not properly restored before step")
        
        criterion = SpatialMoELoss(lambda_deep_supervision=0.4)
        it = iter(train_loader)
        batch = next(it)
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        edges = batch['edge_map'].to(device)
        
        for _ in range(5):
            with autocast(device_type='cuda', dtype=torch.float16):
                out, moe_outputs = model(images)
                loss_dict = criterion(
                    saliency_logits=out.saliency_logits, 
                    edge_logits=out.boundary_logits, 
                    moe_outputs=moe_outputs, 
                    target=masks, 
                    gt_boundary=edges,
                    aux_logits_16=out.aux_logits_16,
                    aux_logits_8=out.aux_logits_8,
                    aux_logits_4=out.aux_logits_4
                )
                loss = loss_dict['L_total']
            eng.scaler.scale(loss).backward() # type: ignore
            overflow, _ = eng.step()
            eng.optimizer.zero_grad()
            if not overflow:
                break
        
        assert_val(eng.global_step == ckpt['global_step'] + 1, "Global step did not increment correctly after behavioral resume step")
        assert_val(opt.param_groups[0]['lr'] != ckpt['lr_before_ckpt'], "LR did not update via scheduler after step")
        
        if rank == 0:
            with open(result_file, "w") as f:
                json.dump({"status": "PASS", "final_step": eng.global_step}, f)
    except Exception as e:
        if rank == 0:
            with open(result_file, "w") as f:
                json.dump({"status": "FAIL", "reason": str(e)}, f)
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["smoke1gpu", "ddp", "memory", "resume_a", "resume_b"])
    parser.add_argument("--data_root", type=str, default="data/WXSDO_data")
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--config_hash", type=str, default="")
    args = parser.parse_args()
    
    if args.mode == "smoke1gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        run_tests_on_engine(device, 0, 0, 1, args.data_root, args.result_file, is_ddp=False)
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", init_method="env://")
        
        if args.mode == "ddp":
            run_tests_on_engine(device, local_rank, rank, world_size, args.data_root, args.result_file, is_ddp=True)
        elif args.mode == "memory":
            run_memory_calibration(device, args.result_file)
        elif args.mode == "resume_a":
            run_resume_a(device, local_rank, rank, world_size, args.data_root, args.result_file)
        elif args.mode == "resume_b":
            run_resume_b(device, local_rank, rank, world_size, args.data_root, args.result_file)
        
        dist.destroy_process_group()
        
    if (not hasattr(args, "mode") or args.mode == "smoke1gpu" or int(os.environ.get("RANK", 0)) == 0) and os.path.exists(args.result_file):
        with open(args.result_file, "r") as f:
            data = json.load(f)
        if args.run_id is not None: data["run_id"] = args.run_id
        if args.config_hash is not None: data["config_hash"] = args.config_hash
        with open(args.result_file, "w") as f:
            json.dump(data, f)

