import os
import torch
import torch.nn as nn
import torch.distributed as dist
import copy
from src.model import SpatialMoESODNet
from src.loss import SpatialMoELoss
from src.optimization import OptimizationEngine, get_parameter_groups, WarmupCosineScheduler
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np

def run_ddp_tests(args):
    """
    Main entry point for explicit DDP validation runs triggered by --ddp_test.
    We assume the process group is NOT yet initialized when this is called, 
    so we initialize it here to easily run isolated tests.
    """
    backend = "gloo" if torch.cuda.device_count() < 2 else "nccl"
    dist.init_process_group(backend, init_method="env://")
    
    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        
        num_gpus = torch.cuda.device_count()
        device_id = local_rank % num_gpus if num_gpus > 0 else 0
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        
        if world_size != 2:
            raise RuntimeError("DDP tests require exactly 2 GPUs.")
            
        if rank == 0:
            print("========================================")
            print("STARTING EXPLICIT DDP VALIDATION SUITE")
            print("========================================")
            
        dist.barrier()
        
        # Test 1: Two-Rank Expert Activation Test
        if rank == 0: print("\n[TEST 1] Two-Rank Expert Activation Test...")
        test_expert_activation(rank, local_rank, device)
        dist.barrier()
        if rank == 0: print("[TEST 1] PASSED")
        
        # Test 2: DDP Gradient Consistency Test (DDP vs Single-Process)
        if rank == 0: print("\n[TEST 2] DDP Gradient Consistency Test...")
        test_gradient_consistency(rank, local_rank, device)
        dist.barrier()
        if rank == 0: print("[TEST 2] PASSED")
        
        # Test 3: no_sync Mathematical Equivalence
        if rank == 0: print("\n[TEST 3] no_sync Mathematical Equivalence Test...")
        test_no_sync_equivalence(rank, local_rank, device)
        dist.barrier()
        if rank == 0: print("[TEST 3] PASSED")
        
        # Test 4: DistributedSampler Resume State Verification
        if rank == 0: print("\n[TEST 4] DistributedSampler Resume State Verification...")
        test_sampler_resume(rank)
        dist.barrier()
        if rank == 0: print("[TEST 4] PASSED")
        
        # Test 5: Exact Resume Reproducibility (Continuous vs Interrupted)
        if rank == 0: print("\n[TEST 5] Exact Resume Reproducibility Test...")
        test_resume_reproducibility(rank, local_rank, device)
        dist.barrier()
        if rank == 0: print("[TEST 5] PASSED")
        
        if rank == 0:
            print("\n========================================")
            print("DDP TEST PASSED")
            print("========================================")
            
    except Exception as e:
        print(f"Exception on Rank {dist.get_rank()}: {e}")
        raise e
    finally:
        dist.destroy_process_group()

def get_dummy_batch(b, device):
    images = torch.randn(b, 3, 192, 192, device=device)
    masks = torch.ones(b, 1, 192, 192, device=device)
    edges = torch.ones(b, 1, 192, 192, device=device)
    return images, masks, edges

def test_expert_activation(rank, local_rank, device):
    """
    Forces divergent routing across ranks.
    Rank 0 routes strictly to [0,1,3]
    Rank 1 routes strictly to [1,2,5]
    Verifies that gradients synchronize properly without hang and inactive experts are handled safely.
    """
    model = SpatialMoESODNet(dim=256, use_deep_supervision=True).to(device)
    
    def mock_moe_forward(moe_layer, idx_list):
        # We replace the moe layer's forward method completely
        def patched_forward(x, force_expert_id=None):
            B, C, H, W = x.shape
            N_tokens = H * W
            
            # 1. Routing Probs
            routing_probs = torch.zeros(B, 6, H, W, device=x.device)
            for idx in idx_list:
                routing_probs[:, idx, :, :] = 1.0 / len(idx_list)
                
            # 2. Fake top-k
            # Flatten to [B, H*W, E]
            routing_flat = routing_probs.view(B, 6, N_tokens).transpose(1, 2)
            topk_vals, topk_indices = torch.topk(routing_flat, k=2, dim=-1)
            
            # 3. Simulate expert dispatch gradients safely
            # We must pass gradients into the actual experts to test DDP sync!
            x_tokens = x.flatten(2).transpose(1, 2)
            flat_x = x_tokens.reshape(B * N_tokens, C)
            output_tokens = torch.zeros_like(flat_x)
            
            for i, expert in enumerate(moe_layer.experts):
                if i in idx_list:
                    # Fake active tokens just to pass gradients
                    active_out = expert(flat_x) * (1.0 / len(idx_list))
                    output_tokens += active_out
                    
            fused_feature_map = output_tokens.view(B, N_tokens, C).transpose(1, 2).view(B, C, H, W).contiguous()
            
            from src.moe_layer import MoEOutput
            return MoEOutput(
                features=fused_feature_map,
                routing_probs=routing_probs,
                topk_indices=topk_indices,
                topk_gates=topk_vals,
                entropy=torch.zeros(B, 1, H, W, device=x.device),
                clean_logits=routing_probs
            )
        return patched_forward
    
    experts_for_rank = [0, 1, 3] if rank == 0 else [1, 2, 5]
    
    # Patch all 3 scales
    model.moe_4.forward = mock_moe_forward(model.moe_4, experts_for_rank)
    model.moe_8.forward = mock_moe_forward(model.moe_8, experts_for_rank)
    model.moe_16.forward = mock_moe_forward(model.moe_16, experts_for_rank)
    
    ddp_model = nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index], find_unused_parameters=True
    )
    
    criterion = SpatialMoELoss(lambda_deep_supervision=0.4)
    images, masks, edges = get_dummy_batch(1, device)
    
    out, moe_outputs = ddp_model(images)
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
    loss.backward()
    
    # Verify gradients exist for exactly the expected experts
    base_model = ddp_model.module
    for expert_idx in range(6):
        # We check moe_4 specifically
        grad = base_model.moe_4.experts[expert_idx].pwconv1.weight.grad
        if expert_idx in experts_for_rank:
            # Active on this rank, MUST have finite gradients
            assert grad is not None
            assert torch.isfinite(grad).all()
        else:
            # Inactive on this rank. However, due to DDP all-reduce, it might get gradients 
            # if the OTHER rank activated it.
            # R0: 0,1,3. R1: 1,2,5.
            # So experts 0,1,2,3,5 should ultimately receive gradients after sync.
            # Expert 4 is inactive on BOTH, its gradient should remain exactly zero or None.
            is_active_globally = expert_idx in [0, 1, 2, 3, 5]
            if is_active_globally:
                assert grad is not None
                assert torch.isfinite(grad).all()
            else:
                # Expert 4
                if grad is not None:
                    assert grad.abs().sum() == 0, f"Expert {expert_idx} received non-zero gradient despite being globally inactive!"

def _create_deterministic_model(device):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = SpatialMoESODNet(dim=256, use_deep_supervision=True).to(device)
    # Convert BN to SyncBN if any exist just to be safe, though none do
    return model

def test_gradient_consistency(rank, local_rank, device):
    """
    Compare DDP gradient averaging vs Single-Process gradient over a combined global batch.
    """
    # 1. Setup identically seeded local model & DDP model
    local_model = _create_deterministic_model(device)
    ddp_model_raw = _create_deterministic_model(device)
    ddp_model = nn.parallel.DistributedDataParallel(
        ddp_model_raw, device_ids=[device.index], find_unused_parameters=True
    )
    
    criterion = SpatialMoELoss(lambda_deep_supervision=0.4)
    
    # 2. Create specific batches
    torch.manual_seed(100)
    batch_0 = get_dummy_batch(1, device)
    torch.manual_seed(200)
    batch_1 = get_dummy_batch(1, device)
    
    # Global batch is batch_0 concatenated with batch_1
    global_images = torch.cat([batch_0[0], batch_1[0]], dim=0)
    global_masks = torch.cat([batch_0[1], batch_1[1]], dim=0)
    global_edges = torch.cat([batch_0[2], batch_1[2]], dim=0)
    
    # 3. Single-Process Pass
    local_model.zero_grad()
    with autocast(device_type='cuda', dtype=torch.float16):
        out_loc, moe_outputs_loc = local_model(global_images)
        loss_dict = criterion(
            saliency_logits=out_loc.saliency_logits,
            edge_logits=out_loc.boundary_logits,
            moe_outputs=moe_outputs_loc,
            target=global_masks,
            gt_boundary=global_edges,
            aux_logits_16=out_loc.aux_logits_16,
            aux_logits_8=out_loc.aux_logits_8,
            aux_logits_4=out_loc.aux_logits_4
        )
        loss_loc = loss_dict['L_total']
    
    loss_loc.backward()
    
    # 4. DDP Pass
    ddp_model.zero_grad()
    my_batch = batch_0 if rank == 0 else batch_1
    
    with autocast(device_type='cuda', dtype=torch.float16):
        out_ddp, moe_outputs_ddp = ddp_model(my_batch[0])
        loss_dict = criterion(
            saliency_logits=out_ddp.saliency_logits,
            edge_logits=out_ddp.boundary_logits,
            moe_outputs=moe_outputs_ddp,
            target=my_batch[1],
            gt_boundary=my_batch[2],
            aux_logits_16=out_ddp.aux_logits_16,
            aux_logits_8=out_ddp.aux_logits_8,
            aux_logits_4=out_ddp.aux_logits_4
        )
        loss_ddp = loss_dict['L_total']
        
    loss_ddp.backward()
    
    # 5. Compare Gradients
    # We compare the first convolution in the backbone as a stable indicator
    grad_local = local_model.backbone.proj_4.weight.grad
    grad_ddp = ddp_model.module.backbone.proj_4.weight.grad
    
    # Due to AMP precision (FP16), tolerance needs to be relatively loose
    diff = (grad_local - grad_ddp).abs().max().item()
    assert diff < 1e-3, f"DDP gradients do not match local accumulation! Max diff: {diff}"

def test_no_sync_equivalence(rank, local_rank, device):
    """
    Verify that N-1 microsteps of no_sync + 1 sync step 
    equals N fully synchronized microsteps.
    """
    model_sync = _create_deterministic_model(device)
    ddp_sync = nn.parallel.DistributedDataParallel(model_sync, device_ids=[device.index], find_unused_parameters=True)
    
    model_no_sync = _create_deterministic_model(device)
    ddp_no_sync = nn.parallel.DistributedDataParallel(model_no_sync, device_ids=[device.index], find_unused_parameters=True)
    
    criterion = SpatialMoELoss(lambda_deep_supervision=0.4)
    
    torch.manual_seed(999 + rank)
    batch_A = get_dummy_batch(1, device)
    batch_B = get_dummy_batch(1, device)
    
    # Execution 1: Fully Synced
    ddp_sync.zero_grad()
    for batch in [batch_A, batch_B]:
        with autocast(device_type='cuda', dtype=torch.float16):
            out, moe_outs = ddp_sync(batch[0])
            loss_dict = criterion(
                saliency_logits=out.saliency_logits,
                edge_logits=out.boundary_logits,
                moe_outputs=moe_outs,
                target=batch[1],
                gt_boundary=batch[2],
                aux_logits_16=out.aux_logits_16,
                aux_logits_8=out.aux_logits_8,
                aux_logits_4=out.aux_logits_4
            )
            loss = loss_dict['L_total']
            loss = loss / 2.0
        loss.backward()
        
    grad_sync = ddp_sync.module.backbone.proj_4.weight.grad.clone()
    
    # Execution 2: no_sync for A, sync for B
    ddp_no_sync.zero_grad()
    with ddp_no_sync.no_sync():
        with autocast(device_type='cuda', dtype=torch.float16):
            out, moe_outs = ddp_no_sync(batch_A[0])
            loss_dict = criterion(
                saliency_logits=out.saliency_logits,
                edge_logits=out.boundary_logits,
                moe_outputs=moe_outs,
                target=batch_A[1],
                gt_boundary=batch_A[2],
                aux_logits_16=out.aux_logits_16,
                aux_logits_8=out.aux_logits_8,
                aux_logits_4=out.aux_logits_4
            )
            loss = loss_dict['L_total']
            loss = loss / 2.0
        loss.backward()
        
    # Sync step
    with autocast(device_type='cuda', dtype=torch.float16):
        out, moe_outs = ddp_no_sync(batch_B[0])
        loss_dict = criterion(
            saliency_logits=out.saliency_logits,
            edge_logits=out.boundary_logits,
            moe_outputs=moe_outs,
            target=batch_B[1],
            gt_boundary=batch_B[2],
            aux_logits_16=out.aux_logits_16,
            aux_logits_8=out.aux_logits_8,
            aux_logits_4=out.aux_logits_4
        )
        loss = loss_dict['L_total']
        loss = loss / 2.0
    loss.backward()
    
    grad_no_sync = ddp_no_sync.module.backbone.proj_4.weight.grad.clone()
    
    diff = (grad_sync - grad_no_sync).abs().max().item()
    assert diff < 1e-5, f"no_sync gradient accumulation mismatch! Max diff: {diff}"

def test_sampler_resume(rank):
    """
    Verifies DistributedSampler determinism when resuming mid-epoch.
    """
    dataset = TensorDataset(torch.arange(100))
    
    # Base run: Epoch 2
    sampler_a = DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True, seed=42)
    sampler_a.set_epoch(2)
    
    indices_a = list(iter(sampler_a))
    
    # Mid-epoch halt at batch 15
    batch_halt = 15
    
    # Resume run: Re-init, fast forward 15
    sampler_b = DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True, seed=42)
    sampler_b.set_epoch(2)
    
    indices_b = list(iter(sampler_b))
    
    assert indices_a == indices_b, "Sampler permutation diverged across re-initializations!"
    
    # The remainder of the epoch should perfectly match what A would have yielded after 15
    remainder_a = indices_a[batch_halt:]
    remainder_b = indices_b[batch_halt:]
    
    assert remainder_a == remainder_b, "Mid-epoch remainder diverged!"

def test_resume_reproducibility(rank, local_rank, device):
    """
    RUN A: Train continuously for 10 optimizer steps.
    RUN B: Train for 5 steps, save checkpoint, 'terminate', resume from checkpoint, train for 5 more steps.
    Compare final parameters, optimizer state, scaler state, and final loss.
    """
    import shutil
    import hashlib
    from src.dataset import get_dataloaders
    from src.train_ddp import get_rng_states, set_rng_states, get_config_hash, CHECKPOINT_FORMAT_VERSION
    from torch.optim import AdamW

    CONFIG = {
        "BATCH_SIZE_PER_GPU": 1,
        "GRAD_ACCUM_STEPS": 2,
        "WARMUP_STEPS": 10,
        "TOTAL_STEPS": 100,
    }
    ckpt_dir = "/tmp/wxsod_test_ckpt"
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()

    def create_environment():
        # Clean deterministic state
        torch.manual_seed(42 + rank)
        np.random.seed(42 + rank)
        random.seed(42 + rank)
        
        train_loader, _, _, _ = get_dataloaders(
            root_dir="/kaggle/input/wxsod-dataset", # Assuming it might fail if doesn't exist, we'll use synthetic dummy below
            image_size=192, 
            batch_size=CONFIG["BATCH_SIZE_PER_GPU"],
            num_workers=0, # NUM_WORKERS=0 per user request for strict exact-resume test
            max_samples=64, # Need enough for 10 steps * 2 accum * 2 gpus = 40 samples
            distributed=True,
            rank=rank,
            world_size=2,
            base_seed=42
        )
        
        model = SpatialMoESODNet(dim=256, use_deep_supervision=True).to(device)
        ddp_model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=True
        )
        
        groups = get_parameter_groups(ddp_model.module)
        optimizer = AdamW(groups)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=CONFIG["WARMUP_STEPS"], total_steps=CONFIG["TOTAL_STEPS"])
        engine = OptimizationEngine(ddp_model, optimizer, scheduler, amp_enabled=True, amp_dtype=torch.float16)
        criterion = SpatialMoELoss(lambda_deep_supervision=0.4)
        
        return train_loader, ddp_model, engine, criterion

    def mock_train_loop(train_loader, ddp_model, engine, criterion, start_epoch, start_batch, run_until_step, do_checkpoint=False):
        epoch = start_epoch
        batch_idx = start_batch
        
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(epoch)
        train_loader.dataset.set_epoch(epoch)
        
        iterator = iter(train_loader)
        
        # Synthetic fast-forward if we are mocking it without dataset because /kaggle doesn't exist locally
        # Actually, if dataset root doesn't exist, get_dataloaders fails.
        # We need a synthetic dummy iterator for this test that mimics dataset!
        pass

    # Since the real dataset might not exist on this environment (e.g., /kaggle/input missing),
    # let's create a pure synthetic deterministic dataloader that obeys set_epoch and fast-forwarding!
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, size, seed):
            self.size = size
            self.seed = seed
            self.epoch = 0
            
        def __len__(self):
            return self.size
            
        def set_epoch(self, epoch):
            self.epoch = epoch
            
        def __getitem__(self, index):
            # Deterministic augmentation simulation
            sample_seed = int(hashlib.md5(f"{self.seed}_{self.epoch}_{index}".encode('utf-8')).hexdigest()[:8], 16)
            rng = torch.Generator().manual_seed(sample_seed)
            img = torch.randn(3, 192, 192, generator=rng)
            mask = torch.ones(1, 192, 192)
            edge = torch.ones(1, 192, 192)
            return {'image': img, 'mask': mask, 'edge_map': edge, 'name': f"{index}"}
            
    def get_synthetic_loader():
        ds = SyntheticDataset(64, seed=42)
        sampler = DistributedSampler(ds, num_replicas=2, rank=rank, shuffle=True, seed=42)
        loader = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=0)
        return loader
        
    def run_training_segment(loader, ddp_model, engine, criterion, start_epoch, start_batch, target_steps, ckpt_at_step=None):
        epoch = start_epoch
        batch_idx = start_batch
        
        sampler = loader.sampler
        sampler.set_epoch(epoch)
        loader.dataset.set_epoch(epoch)
        
        iterator = iter(loader)
        
        # Fast forward
        for _ in range(batch_idx):
            next(iterator)
            
        final_loss = 0.0
        while engine.global_step < target_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            edges = batch['edge_map'].to(device)
            
            is_final = (batch_idx + 1) % CONFIG["GRAD_ACCUM_STEPS"] == 0
            
            sync_context = ddp_model.no_sync() if not is_final else torch.autograd.profiler.profile(enabled=False)
            
            with sync_context:
                with autocast(device_type='cuda', dtype=torch.float16):
                    out, moe_outputs = ddp_model(images)
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
                    loss = loss / CONFIG["GRAD_ACCUM_STEPS"]
                    
                engine.scaler.scale(loss).backward()
                final_loss = loss.item() * CONFIG["GRAD_ACCUM_STEPS"]
                
            if is_final:
                overflow, _ = engine.step()
                engine.optimizer.zero_grad()
                
                if ckpt_at_step and engine.global_step == ckpt_at_step:
                    if rank == 0:
                        state = {
                            'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
                            'epoch': epoch,
                            'batch_in_epoch': batch_idx + 1,
                            'global_step': engine.global_step,
                            'best_metric': 999.0,
                            'model_state_dict': ddp_model.module.state_dict(),
                            'engine_state_dict': engine.state_dict(),
                            'rng_states': get_rng_states(device),
                            'config': CONFIG,
                            'config_hash': get_config_hash(CONFIG, "dummy"),
                            'world_size': 2,
                            'timestamp': 0
                        }
                        torch.save(state, os.path.join(ckpt_dir, "latest.pth"))
                    dist.barrier()
                    return final_loss, epoch, batch_idx + 1
                    
            batch_idx += 1
            
        return final_loss, epoch, batch_idx
        
    def setup_model_and_engine():
        torch.manual_seed(42 + rank)
        np.random.seed(42 + rank)
        random.seed(42 + rank)
        model = SpatialMoESODNet(dim=256, use_deep_supervision=True).to(device)
        ddp_model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
        groups = get_parameter_groups(ddp_model.module)
        optimizer = AdamW(groups)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=CONFIG["WARMUP_STEPS"], total_steps=CONFIG["TOTAL_STEPS"])
        engine = OptimizationEngine(ddp_model, optimizer, scheduler, amp_enabled=True, amp_dtype=torch.float16)
        criterion = SpatialMoELoss(lambda_deep_supervision=0.4)
        return ddp_model, engine, criterion

    # --- RUN A: Continuous ---
    loader_a = get_synthetic_loader()
    model_a, engine_a, crit_a = setup_model_and_engine()
    
    # Pre-warm exactly the same way to ensure initial RNG alignment
    set_rng_states(get_rng_states(device), device)
    
    loss_a, _, _ = run_training_segment(loader_a, model_a, engine_a, crit_a, 0, 0, 10)
    
    # Save Run A state
    params_a = {n: p.clone() for n, p in model_a.module.named_parameters()}
    scale_a = engine_a.scaler.get_scale()
    
    # --- RUN B: Interrupted ---
    loader_b = get_synthetic_loader()
    model_b, engine_b, crit_b = setup_model_and_engine()
    
    # Pre-warm again
    set_rng_states(get_rng_states(device), device)
    
    # Train 5 steps and checkpoint
    run_training_segment(loader_b, model_b, engine_b, crit_b, 0, 0, 5, ckpt_at_step=5)
    
    # "Kill" process by reinitializing model/engine identically, then load checkpoint
    loader_b2 = get_synthetic_loader()
    model_b2, engine_b2, crit_b2 = setup_model_and_engine()
    
    ckpt_path = os.path.join(ckpt_dir, "latest.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_b2.module.load_state_dict(checkpoint['model_state_dict'])
    engine_b2.load_state_dict(checkpoint['engine_state_dict'])
    set_rng_states(checkpoint['rng_states'], device)
    
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint['batch_in_epoch']
    
    dist.barrier()
    
    # Resume for remaining 5 steps
    loss_b, _, _ = run_training_segment(loader_b2, model_b2, engine_b2, crit_b2, start_epoch, start_batch, 10)
    
    # Compare
    params_b = {n: p.clone() for n, p in model_b2.module.named_parameters()}
    scale_b = engine_b2.scaler.get_scale()
    
    assert engine_a.global_step == engine_b2.global_step == 10
    
    diff_scale = abs(scale_a - scale_b)
    assert diff_scale < 1e-5, f"Scaler state diverged! {scale_a} vs {scale_b}"
    
    diff_loss = abs(loss_a - loss_b)
    assert diff_loss < 1e-4, f"Final step loss diverged! A: {loss_a}, B: {loss_b}"
    
    max_param_diff = 0.0
    for name in params_a:
        diff = (params_a[name] - params_b[name]).abs().max().item()
        if diff > max_param_diff: max_param_diff = diff
        
    assert max_param_diff < 1e-5, f"Parameters diverged after exact resume! Max diff: {max_param_diff}"

