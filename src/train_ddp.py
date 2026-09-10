import os
import sys
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast
from tqdm import tqdm
import hashlib
import time
import shutil

CHECKPOINT_FORMAT_VERSION = 1

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.dataset import get_dataloaders
from src.model import SpatialMoESODNet
from src.loss import SpatialMoELoss
from torch.optim import AdamW
from src.optimization import get_parameter_groups, freeze_backbone, unfreeze_backbone, WarmupCosineScheduler, OptimizationEngine
from src.config import ExperimentConfig
from src.experiment import setup_experiment_run, update_registry_status
from src import hf_sync

def default_checkpoint_base():
    """Kaggle and local VS Code share this file, so the checkpoint root
    can't be hardcoded to /kaggle/working. CHECKPOINT_ROOT env var wins if
    set; otherwise default to /kaggle/working when present (Kaggle), else
    a local ./checkpoints dir next to the project."""
    override = os.environ.get("CHECKPOINT_ROOT")
    if override:
        return override
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/WXSOD_Checkpoints"
    return os.path.join(_project_root, "checkpoints")

def default_preflight_base():
    override = os.environ.get("PREFLIGHT_ROOT")
    if override:
        return override
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/WXSOD_Preflight"
    return os.path.join(_project_root, "preflight")

def set_rng_states(states, device):
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch_cpu'])
    torch.cuda.set_rng_state(states['torch_cuda'], device)

def get_rng_states(device):
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state(device)
    }

def verify_parameter_consistency(model):
    """
    Hash the parameter names to ensure all ranks have identical architectures.
    """
    names = sorted([name for name, _ in model.named_parameters()])
    name_str = ",".join(names)
    hash_obj = hashlib.md5(name_str.encode('utf-8')).hexdigest()
    return hash_obj

def get_config_hash(config, model_hash):
    """
    Canonical experiment identity hash.
    Changes here are FATAL for resume.
    """
    import copy
    core_config = copy.deepcopy({
        k: v for k, v in config.items()
        if k not in ["NUM_WORKERS", "CHECKPOINT_EVERY_N_STEPS", "experiment_id", "run_id", "batch_equivalence"]
    })
    
    # Remove environment-specific paths from hash
    if 'data' in core_config and 'dataset_root' in core_config['data']:
        del core_config['data']['dataset_root']
        
    hash_str = f"v{CHECKPOINT_FORMAT_VERSION}_{model_hash}_{sorted(core_config.items())}"
    return hashlib.md5(hash_str.encode('utf-8')).hexdigest()

def check_normalization_layers(model, rank):
    """
    Statically inspect model for BatchNorm and log.
    """
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append(name)
            
    if rank == 0:
        if bn_layers:
            print(f"⚠️ Found BatchNorm layers: {len(bn_layers)} occurrences.")
        else:
            print("No BatchNorm layers detected.")
            print("SyncBatchNorm disabled.")

def get_device_diagnostics(device):
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_res = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    return allocated, reserved, peak_alloc, peak_res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/baseline_v1.json', help='Path to canonical experiment configuration JSON')
    parser.add_argument('--dry_run', action='store_true', help='Perform a dry run to inspect experiment details without executing')
    parser.add_argument('--smoke_test', action='store_true', help='Run a 1-GPU smoke test')
    parser.add_argument('--ddp_test', action='store_true', help='Run explicit DDP validation tests')
    parser.add_argument('--calibration', action='store_true', help='Run memory calibration')
    parser.add_argument('--resume', type=str, nargs='?', const='latest', default=None, help='Resume from latest checkpoint in experiment directory')
    parser.add_argument('--preflight', action='store_true', help='Run a 2-5 step dry run using a separate preflight output directory')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting existing experiment checkpoints')
    parser.add_argument('--max_optimizer_steps', type=int, default=None, help='Maximum optimizer steps to run (used for preflight)')
    args = parser.parse_args()

    if args.ddp_test:
        from tests.test_moe_ddp import run_ddp_tests
        run_ddp_tests(args)
        return

    # DDP Initialization
    dist.init_process_group("nccl", init_method="env://")
    
    rank = None
    config = None
    hf_pusher = None
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        import signal
        global_stop_flag = [False]
        def handle_signal(sig, frame):
            print(f"Rank {rank} received termination signal {sig}. Graceful shutdown initiated...")
            global_stop_flag[0] = True
            
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        # Load canonical config
        config = ExperimentConfig.load(args.config)
        
        # Determine dataset directory and run tracking
        DATA_DIR = config.data.dataset_root
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"DATA_ROOT could not be resolved: {DATA_DIR}")
            
        if world_size < 2 and not (args.dry_run or args.preflight):
            raise RuntimeError(f"Requested production configuration requires >=2 GPUs. Found {world_size}.")
            
        # Set up Experiment Run (Directory, Git ID, Hash, Registry)
        # Note: Setup only on Rank 0, but dry_run operates synchronously.
        if args.dry_run:
            if rank == 0:
                setup_experiment_run(config, dry_run=True)
            return

        run_dir = ""
        if rank == 0:
            run_dir = setup_experiment_run(config, dry_run=False)
            update_registry_status("experiments", config.run_id, "RUNNING")
            print("====================================")
            print(f"EXPERIMENT INITIALIZED: {config.experiment_id}")
            print(f"RUN ID: {config.run_id}")
            print(f"RUN DIRECTORY: {run_dir}", flush=True)
            print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
            print(f"World size: {world_size}", flush=True)
            print("====================================", flush=True)
            
        dist.barrier()
        print(f"Rank {rank} | Local Rank {local_rank} | World Size {world_size} | GPU {local_rank}")
        dist.barrier()
        
        # Broadcast run_dir to all ranks
        run_dir_obj = [run_dir]
        dist.broadcast_object_list(run_dir_obj, src=0)
        run_dir = run_dir_obj[0]
        
        cfg_hash = get_config_hash(config.to_dict(), "model_config_hash")
        
        # Override directories if preflight
        if args.preflight:
            base_dir = default_preflight_base()
            args.max_optimizer_steps = args.max_optimizer_steps or 5
        else:
            base_dir = default_checkpoint_base()
            # Overwrite prevention
            if os.path.exists(os.path.join(base_dir, "training_complete.json")) and not args.overwrite:
                raise RuntimeError("TRAINING ALREADY COMPLETE. Use --overwrite to bypass.")
            if os.path.exists(os.path.join(base_dir, "best.pth")) and not args.overwrite and not args.resume:
                raise RuntimeError("Production checkpoints already exist. Use --overwrite to bypass.")
            
        os.makedirs(base_dir, exist_ok=True)
        if rank == 0:
            with open(os.path.join(base_dir, "final_config.json"), "w") as f:
                json.dump(config.to_dict(), f, indent=4)
            with open(os.path.join(base_dir, "final_config_sha256"), "w") as f:
                f.write(cfg_hash)

        # Non-blocking HF pusher: rank 0 writes checkpoints to local disk as
        # before (fast), then hands the file off to a background thread so
        # the upload never stalls the training loop or the DDP barrier.
        # Disabled automatically (no-op enqueue) if HF_REPO_ID isn't set,
        # so local-only runs still work without touching HF at all.
        hf_pusher = None
        if rank == 0 and not args.preflight and os.environ.get("HF_REPO_ID"):
            hf_pusher = hf_sync.AsyncCheckpointPusher()
            
        checkpoint_dir = base_dir
            
        # Deterministic Seeding (Rank-Aware)
        rank_seed = config.train.seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)
        
        # Helper references for compatibility with legacy logic
        # (Over time these should be entirely replaced by `config.`)
        effective_batch = config.opt.effective_global_batch
        
        # Dataloader
        train_loader, val_loader, _, _ = get_dataloaders(
            root_dir=DATA_DIR, 
            batch_size=config.opt.batch_per_gpu,
            image_size=384, 
            num_workers=config.train.num_workers,
            max_samples=config.data.max_samples,
            distributed=True,
            rank=rank,
            world_size=world_size
        )
        train_sampler = train_loader.sampler
        
        expected_optim_steps = (len(train_loader) // config.opt.grad_accum_steps) * config.train.epochs
        if rank == 0:
            print(f"\n--- Dataset Info ---")
            print(f"Train Dataset Size: {len(train_loader.dataset)}") # type: ignore
            print(f"World Size: {world_size}")
            print(f"Samples/Rank: {len(train_loader.dataset) // world_size}") # type: ignore
            print(f"Batch Size/Rank: {config.opt.batch_per_gpu}", flush=True)
            print(f"Effective Global Batch: {effective_batch}", flush=True)
            print(f"Expected Optimizer Steps: {expected_optim_steps}\n", flush=True)

        # Model Construction
        model = SpatialMoESODNet(
            use_deep_supervision=config.model.deep_supervision,
            num_experts=config.model.num_experts,
            window_size=config.model.window_size
        ).to(device)
        
        # Verify architecture consistency across ranks
        local_hash = verify_parameter_consistency(model)
        hash_tensor = torch.zeros(1, dtype=torch.int64, device=device)
        # Convert hex string to integer for simple synchronization verification
        hash_tensor[0] = int(local_hash[:15], 16) 
        
        gathered_hashes = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(gathered_hashes, hash_tensor)
        
        for i in range(world_size):
            if gathered_hashes[i].item() != hash_tensor.item():
                raise RuntimeError(f"Architecture mismatch detected! Rank 0 Hash != Rank {i} Hash")
                
        if rank == 0:
            print("Architecture verified consistent across all ranks.")
            total_params = sum(p.numel() for p in model.parameters())
            train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total Parameters: {total_params:,} | Trainable: {train_params:,}")
            
        check_normalization_layers(model, rank)
        
        # DDP Wrap
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        
        # Optimization
        groups = get_parameter_groups(model.module)
        optimizer = AdamW(groups)
        total_steps = expected_optim_steps if rank == 0 else 50000
        # Broadcast total steps if we want strict sync, but we can just compute it globally
        total_steps_tensor = torch.tensor([len(train_loader) // config.opt.grad_accum_steps * config.train.epochs], device=device)
        dist.broadcast(total_steps_tensor, src=0)
        total_steps = int(total_steps_tensor.item())
        warmup_steps = int(config.opt.warmup_ratio * total_steps)
        
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
        engine = OptimizationEngine(model, optimizer, scheduler, amp_enabled=config.opt.amp, amp_dtype=torch.float16)
        criterion = SpatialMoELoss(
            lambda_iou=config.loss.iou_weight,
            lambda_ssim=config.loss.ssim_weight,
            lambda_boundary=config.loss.boundary_weight,
            lambda_lb=config.loss.load_balance_weight,
            lambda_importance=config.loss.importance_weight,
            lambda_z=config.loss.z_loss_weight,
            lambda_aux_boundary=config.loss.aux_boundary_weight,
            lambda_deep_supervision=config.loss.deep_supervision_weight
        )

        start_epoch = 0
        start_batch = 0
        best_mae = float('inf')
        best_mae_epoch = 0
        best_mae_step = 0
        
        config_hash = config.get_canonical_hash()

        # Resume
        if args.resume:
            if args.resume == "latest":
                source_ckpt_dir = default_checkpoint_base() if args.preflight else checkpoint_dir
                ckpt_path = os.path.join(source_ckpt_dir, "latest.pth")
            else:
                ckpt_path = args.resume
            if os.path.exists(ckpt_path):
                if rank == 0: print(f"Loading checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                
                # Checkpoint validation
                if checkpoint.get('checkpoint_format_version') != CHECKPOINT_FORMAT_VERSION:
                    raise RuntimeError(f"FATAL: Checkpoint version mismatch! Expected {CHECKPOINT_FORMAT_VERSION}")
                    
                if checkpoint.get('world_size') != world_size:
                    raise RuntimeError(f"FATAL: World size mismatch! Checkpoint was {checkpoint.get('world_size')}, current is {world_size}")
                    
                if checkpoint.get('config_hash') != config_hash:
                    raise RuntimeError(f"FATAL: Configuration hash mismatch! Architecture or critical hyperparams changed.")
                
                model.module.load_state_dict(checkpoint['model_state_dict'])
                engine.load_state_dict(checkpoint['engine_state_dict'])
                
                start_epoch = checkpoint['epoch']
                start_batch = checkpoint['batch_in_epoch']
                best_mae = checkpoint['best_metric']
                
                if 'rng_states' in checkpoint:
                    set_rng_states(checkpoint['rng_states'], device)
                    
                dist.barrier()
                
                if rank == 0:
                    print(f"========================================")
                    print(f"RESUME TRAINING")
                    print(f"========================================")
                    print(f"Checkpoint:\n    {ckpt_path}")
                    print(f"Saved state:\n    Epoch: {start_epoch}\n    Batch in epoch: {start_batch}\n    Global step: {engine.global_step}\n    Best metric: {best_mae:.4f}", flush=True)
                    print(f"Configuration compatibility:\n    PASS", flush=True)
                    print(f"========================================", flush=True)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Training Loop
        for epoch in range(start_epoch, config.train.epochs):
            model.train()
            if hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch) # type: ignore
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch) # type: ignore
            
            # Unfreeze backbone after epoch 0
            if epoch == 0 and not args.resume:
                if rank == 0: print("Warmup Phase: Backbone Frozen")
                freeze_backbone(model.module)
            elif epoch == 1:
                if rank == 0: print("Phase 2: Backbone Unfrozen")
                unfreeze_backbone(model.module)
                
            iterator = iter(train_loader)
            
            # Fast-forward exact sampler position if resuming mid-epoch
            if epoch == start_epoch and start_batch > 0:
                if rank == 0: print(f"Fast-forwarding {start_batch} batches to resume exact mid-epoch state...")
                for _ in range(start_batch):
                    next(iterator)
            
            batch_idx = start_batch if epoch == start_epoch else 0
            start_batch = 0 # reset
            
            if rank == 0:
                if rank == 0: print(f"--- Starting Epoch {epoch+1}/{config.train.epochs} [Train] ---", flush=True)
            engine.optimizer.zero_grad()
            
            while True:
                try:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                        
                    images = batch['image'].to(device, non_blocking=True)
                    masks = batch['mask'].to(device, non_blocking=True)
                    edges = batch['edge_map'].to(device, non_blocking=True)
                    pad_masks = batch['pad_mask'].to(device, non_blocking=True)
                    
                    is_final_microstep = (batch_idx + 1) % config.opt.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader)
                    overflow = False
                    
                    # Context manager for no_sync
                    from contextlib import nullcontext
                    sync_context = model.no_sync() if not is_final_microstep else nullcontext()
                    
                    with sync_context:
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
                                aux_logits_4=out.aux_logits_4,
                                pad_mask=pad_masks
                            )
                            loss = loss_dict['L_total'] / config.opt.grad_accum_steps

                        engine.scaler.scale(loss).backward() # type: ignore
                    
                    if is_final_microstep:
                        overflow, clip_norm = engine.step()
                        engine.optimizer.zero_grad()
                        
                        if rank == 0 and overflow:
                            if rank == 0: print(f"[Epoch {epoch+1} Batch {batch_idx+1}] AMP Overflow Detected! Skipping optimizer step.")
                        
                        if engine.global_step > 0 and engine.global_step % config.train.checkpoint_every_n_steps == 0:
                            dist.barrier()
                            if rank == 0:
                                state = {
                                    'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
                                    'epoch': epoch,
                                    'batch_in_epoch': batch_idx + 1,
                                    'global_step': engine.global_step,
                                    'best_metric': best_mae,
                                    'model_state_dict': model.module.state_dict(),
                                    'engine_state_dict': engine.state_dict(),
                                    'rng_states': get_rng_states(device),
                                    'config': config.to_dict(),
                                    'config_hash': config_hash,
                                    'world_size': world_size,
                                    'timestamp': time.time()
                                }
                                tmp_path = os.path.join(checkpoint_dir, "latest.pth.tmp")
                                final_path = os.path.join(checkpoint_dir, "latest.pth")
                                backup_path = os.path.join(checkpoint_dir, "latest_prev.pth")
                                
                                try:
                                    torch.save(state, tmp_path)
                                    # Validate temporary
                                    _ = torch.load(tmp_path, map_location='cpu', weights_only=False)
                                    
                                    if os.path.exists(final_path):
                                        shutil.copy2(final_path, backup_path)
                                        
                                    os.replace(tmp_path, final_path)
                                    if rank == 0: print(f"[Epoch {epoch+1} Batch {batch_idx+1}] Checkpoint correctly promoted.", flush=True)
                                    
                                    # Write checkpoint manifest
                                    manifest = {
                                        "checkpoint": "latest.pth",
                                        "size_bytes": os.path.getsize(final_path),
                                        "epoch": epoch,
                                        "global_step": engine.global_step,
                                        "config_hash": config_hash,
                                        "timestamp": state['timestamp']
                                    }
                                    with open(os.path.join(checkpoint_dir, "checkpoint_manifest.json"), "w") as mf:
                                        json.dump(manifest, mf)

                                    if hf_pusher is not None:
                                        hf_pusher.enqueue_checkpoint(
                                            final_path, name="latest.pth",
                                            extra_meta={"epoch": epoch, "global_step": engine.global_step, "config_hash": config_hash},
                                        )
                                except Exception as e:
                                    print(f"CHECKPOINT WRITE FAILED: {e}")
                            dist.barrier()
                            
                    if args.max_optimizer_steps and engine.global_step >= args.max_optimizer_steps:
                        break
                        
                    if global_stop_flag[0]:
                        print(f"Rank {rank} terminating early at step {engine.global_step} due to signal.")
                        break
                        
                    if args.calibration and is_final_microstep and engine.global_step >= 5:
                        alloc, res, p_alloc, p_res = get_device_diagnostics(device)
                        print(f"\nCalibration Rank {rank} - Peak Alloc: {p_alloc:.2f} GB | Peak Res: {p_res:.2f} GB")
                        dist.barrier()
                        break

                    if rank == 0:
                        if args.preflight:
                            if is_final_microstep:
                                if rank == 0: 
                                    print(f"[Preflight] Step {engine.global_step} | L_total: {loss_dict['L_total'].item():.4f} | L_bce: {loss_dict['L_bce'].item():.4f} | L_iou: {loss_dict['L_iou'].item():.4f} | L_ssim: {loss_dict['L_ssim'].item():.4f} | L_boundary: {loss_dict['L_boundary'].item():.4f} | L_aux_boundary: {loss_dict['L_aux_boundary'].item():.4f} | L_deep_supervision: {loss_dict['L_deep_supervision'].item():.4f} | L_lb: {loss_dict['L_lb'].item():.4f} | L_importance: {loss_dict['L_importance'].item():.4f} | overflow: {overflow} | pad_valid_frac: {pad_masks.mean().item():.4f}", flush=True)
                                    if moe_outputs and moe_outputs[0].noise_std is not None:
                                        noise = moe_outputs[0].noise_std
                                        print(f"            [Noise_std] scale=4 | mean: {noise.mean().item():.6f} | min: {noise.min().item():.6f} | max: {noise.max().item():.6f}")
                        else:
                            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                                if rank == 0: print(f"[Train] Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item()*config.opt.grad_accum_steps:.4f} | Step: {engine.global_step}", flush=True)
                        
                    batch_idx += 1
                
                except RuntimeError as e:
                    print(f"Runtime error during training: {e}")
                    raise e
                    
            pass
                
            if args.calibration or global_stop_flag[0] or (args.max_optimizer_steps and engine.global_step >= args.max_optimizer_steps):
                break
                
            # Validation
            dist.barrier()
            if rank == 0:
                from src.metrics import SODMetrics
                import cv2
                model.eval()
                
                sod_metrics = SODMetrics()
                with torch.no_grad():
                    if rank == 0: print(f"--- Starting Epoch {epoch+1}/{config.train.epochs} [Val] ---", flush=True)
                    for v_idx, v_batch in enumerate(val_loader):
                        if (v_idx + 1) % 50 == 0 or (v_idx + 1) == len(val_loader):
                            if rank == 0: print(f"[Val] Epoch {epoch+1} | Batch {v_idx+1}/{len(val_loader)}", flush=True)
                        v_images = v_batch['image'].to(device)
                        with autocast(device_type='cuda', dtype=torch.float16):
                            out, _ = model(v_images)
                        sal_pred = torch.sigmoid(out.saliency_logits).detach().float().cpu().numpy()

                        for i in range(v_images.size(0)):
                            meta = {k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i] for k, v in v_batch['meta'].items()}
                            p = sal_pred[i, 0]
                            p_cropped = p[meta['pad_top']:meta['pad_top']+meta['resized_h'], meta['pad_left']:meta['pad_left']+meta['resized_w']]
                            p_orig = cv2.resize(p_cropped, (meta['orig_w'], meta['orig_h']), interpolation=cv2.INTER_LINEAR)
                            
                            gt_path = meta.get('gt_path')
                            if not gt_path: continue
                            orig_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                            if orig_mask is None: continue
                            
                            sod_metrics.step(p_orig, orig_mask)
                            
                    # Diagnostics
                    if (epoch + 1) % 3 == 1:
                        from src.diagnostics import MoEDiagnosticsEngine
                        if rank == 0: print(f"Running MoE Diagnostics for Epoch {epoch+1}...")
                        diag_engine = MoEDiagnosticsEngine(
                            os.path.join(checkpoint_dir, "diagnostics"), 
                            num_experts=config.model.num_experts, 
                            top_k=config.model.top_k
                        )
                        
                        # Save RNG state before diagnostic random sampling
                        rng_pre_diag = get_rng_states(device)
                        
                        if rank == 0: print(f"--- Starting Diagnostics Epoch {epoch+1} ---", flush=True)
                        for d_idx, v_batch in enumerate(val_loader):
                            if (d_idx + 1) % 50 == 0 or (d_idx + 1) == len(val_loader):
                                if rank == 0: print(f"[Diagnostics] Batch {d_idx+1}/{len(val_loader)}", flush=True)
                            v_images = v_batch['image'].to(device)
                            with autocast(device_type='cuda', dtype=torch.float16):
                                out, moe_outputs = model(v_images)
                            diag_engine.update(v_images, moe_outputs, v_batch['meta'])
                            
                        stats = diag_engine.finalize(epoch+1)
                        set_rng_states(rng_pre_diag, device)
                        if rank == 0: 
                            print("Diagnostics completed safely. Check 'diagnostics' folder.")
                            if hf_pusher is not None:
                                diag_base = os.path.join(checkpoint_dir, f"diagnostics_ep{epoch+1}")
                                diag_zip = diag_base + ".zip"
                                shutil.make_archive(diag_base, 'zip', os.path.join(checkpoint_dir, "diagnostics"))
                                print(f"Uploading {diag_zip} to Hugging Face...")
                                hf_pusher.enqueue_checkpoint(
                                    diag_zip, 
                                    name=f"diagnostics_ep{epoch+1}.zip", 
                                    extra_meta={"epoch": epoch + 1, "type": "diagnostics"}
                                )
                        
                results = sod_metrics.get_results()
                val_mae = results.get('MAE', float('inf'))
                if rank == 0: print(f"\nEpoch {epoch+1} | Val MAE: {val_mae:.4f} | F-beta: {results.get('F_max', 0):.4f}", flush=True)

                if val_mae < best_mae:
                    best_mae = val_mae
                    best_mae_epoch = epoch + 1
                    best_mae_step = engine.global_step
                    state = {
                        'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
                        'epoch': epoch,
                        'batch_in_epoch': batch_idx, # Validation happens at end of epoch
                        'global_step': engine.global_step,
                        'best_metric': best_mae,
                        'model_state_dict': model.module.state_dict(),
                        'engine_state_dict': engine.state_dict(),
                        'rng_states': get_rng_states(device),
                        'config': config.to_dict(),
                        'config_hash': config_hash,
                        'world_size': world_size,
                        'timestamp': time.time()
                    }
                    tmp_path = os.path.join(checkpoint_dir, "best.pth.tmp")
                    final_path = os.path.join(checkpoint_dir, "best.pth")
                    try:
                        torch.save(state, tmp_path)
                        _ = torch.load(tmp_path, map_location='cpu', weights_only=False)
                        os.replace(tmp_path, final_path)
                        if rank == 0: print(f"  -> New best model saved to {final_path}!", flush=True)

                        if hf_pusher is not None:
                            hf_pusher.enqueue_checkpoint(
                                final_path, name="best.pth",
                                extra_meta={"epoch": epoch + 1, "global_step": engine.global_step, "best_metric": best_mae, "config_hash": config_hash},
                            )
                    except Exception as e:
                        print(f"BEST CHECKPOINT WRITE FAILED: {e}", flush=True)
                    
                # Counterfactual Ablation
                expert_ablation = getattr(args, 'expert_ablation', None)
                if expert_ablation is not None:
                    scale, exp_id = map(int, expert_ablation.split(','))
                    ablation_cfg = {'scale': scale, 'expert_id': exp_id}
                    print(f"\n--- COUNTERFACTUAL EXPERT ABLATION: Scale {scale}, Expert {exp_id} ---")
                    
                    ablation_metrics = SODMetrics()
                    with torch.no_grad():
                        print(f"--- Starting Ablation Val ---", flush=True)
                        for a_idx, v_batch in enumerate(val_loader):
                            if (a_idx + 1) % 50 == 0 or (a_idx + 1) == len(val_loader):
                                print(f"[Ablation] Batch {a_idx+1}/{len(val_loader)}", flush=True)
                            v_images = v_batch['image'].to(device)
                            with autocast(device_type='cuda', dtype=torch.float16):
                                # Pass ablation_cfg down to model
                                out, _ = model.module(v_images, ablation_cfg=ablation_cfg)
                            sal_pred = torch.sigmoid(out.saliency_logits).detach().float().cpu().numpy()
                            
                            for i in range(v_images.size(0)):
                                meta = {k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i] for k, v in v_batch['meta'].items()}
                                p = sal_pred[i, 0]
                                p_cropped = p[meta['pad_top']:meta['pad_top']+meta['resized_h'], meta['pad_left']:meta['pad_left']+meta['resized_w']]
                                p_orig = cv2.resize(p_cropped, (meta['orig_w'], meta['orig_h']), interpolation=cv2.INTER_LINEAR)
                                gt_path = meta.get('gt_path')
                                if not gt_path: continue
                                orig_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                                if orig_mask is None: continue
                                ablation_metrics.step(p_orig, orig_mask)
                                
                    abl_results = ablation_metrics.get_results()
                    print(f"Ablation Val MAE: {abl_results.get('MAE', float('inf')):.4f} | F-beta: {abl_results.get('F_max', 0):.4f}")
                    
        dist.barrier()
        if rank == 0:
            if not args.preflight and not global_stop_flag[0]:
                update_registry_status("experiments", config.run_id, "COMPLETED", val_metric_val=str(best_mae), best_epoch=str(best_mae_epoch), best_step=str(best_mae_step))
                completion_path = os.path.join(base_dir, "training_complete.json")
                with open(completion_path, "w") as f:
                    json.dump({
                        "status": "COMPLETE",
                        "epoch": epoch if 'epoch' in locals() else start_epoch,
                        "global_step": engine.global_step,
                        "best_checkpoint": "best.pth",
                        "config_hash": cfg_hash
                    }, f, indent=4)
                if hf_pusher is not None:
                    hf_pusher.enqueue_json(completion_path, name="training_complete.json")
            elif args.preflight:
                with open(os.path.join(base_dir, "preflight_results.json"), "w") as f:
                    json.dump({
                        "status": "PASS",
                        "run_id": config.run_id,
                        "optimizer_steps": engine.global_step,
                        "checkpoint_written": os.path.exists(os.path.join(checkpoint_dir, "latest.pth")),
                        "loss_finite": True,
                        "ddp_ok": True,
                        "memory_ok": True,
                        "config_hash": cfg_hash,
                        "timestamp": time.time(),
                        "config": config.to_dict()
                    }, f, indent=4)
            
    except Exception as e:
        if rank == 0 and config is not None:
            try:
                update_registry_status("experiments", config.run_id, "FAILED")
            except:
                pass
        print(f"Exception on Rank {int(os.environ.get('RANK', 0))}: {e}")
        raise e
    finally:
        # Kaggle can tear the session down right after the script exits, so
        # any still-queued checkpoint pushes need to finish (or clearly time
        # out) BEFORE we destroy the process group and return, not after.
        if hf_pusher is not None:
            print("[hf_sync] Flushing pending checkpoint pushes before exit...")
            hf_pusher.close(timeout=900)
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()