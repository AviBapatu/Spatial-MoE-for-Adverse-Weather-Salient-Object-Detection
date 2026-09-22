"""Training setup: config resolution, model/optimizer/dataloader construction.

Everything needed to build a ``TrainCtx`` for the training loop, including
optional resume from a checkpoint and workspace (checkpoint/preflight)
resolution.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, NamedTuple, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW

from src import hf_sync
from src.config import ExperimentConfig, LossConfig
from src.dataset import get_dataloaders
from src.experiment import setup_experiment_run, update_registry_status
from src.log import get_logger
from src.loss import CombinedLoss
from src.model import SpatialMoESODNet
from src.optimization import (
    OptimizationEngine,
    WarmupCosineScheduler,
    get_parameter_groups,
)
from src.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    load_checkpoint,
    validate_checkpoint_fields,
)
from src.training.distributed import (
    get_local_rank,
    get_rank,
    get_world_size,
    is_rank_zero,
    set_rng_states,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Workspace / path helpers
# ---------------------------------------------------------------------------

def default_checkpoint_base(project_root: str) -> str:
    """Resolve where production/promoted checkpoints live."""
    override = os.environ.get("CHECKPOINT_ROOT")
    if override:
        return override
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/WXSOD_Checkpoints"
    return os.path.join(project_root, "checkpoints")


def default_preflight_base(project_root: str) -> str:
    """Resolve where preflight (smoke-test) artifacts live."""
    override = os.environ.get("PREFLIGHT_ROOT")
    if override:
        return override
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/WXSOD_Preflight"
    return os.path.join(project_root, "preflight")


def should_run_diagnostics(epoch_1indexed: int, every_n: int = 3) -> bool:
    """Return True when diagnostics should run this epoch.

    Parameters
    ----------
    epoch_1indexed:
        Current epoch number, 1-indexed.
    every_n:
        Run diagnostics every *every_n* epochs (also always on epoch 1).
        Pass ``config.diag.routing_diagnostic_epochs`` here.
        ``1`` means every epoch.
    """
    return epoch_1indexed == 1 or (epoch_1indexed - 1) % every_n == 0


def device_memory_diagnostics(device: torch.device) -> Tuple[float, float, float, float]:
    """Return (allocated, reserved, peak_allocated, peak_reserved) in GB."""
    return (
        torch.cuda.memory_allocated(device) / (1024 ** 3),
        torch.cuda.memory_reserved(device) / (1024 ** 3),
        torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    )


# ---------------------------------------------------------------------------
# Model / dataloader / optimizer construction
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: ExperimentConfig,
    rank: int,
    world_size: int,
) -> Tuple[Any, Any]:
    """Build rank-aware train and validation loaders."""
    return get_dataloaders(
        root_dir=config.data.dataset_root,
        batch_size=config.opt.batch_per_gpu,
        image_size=384,
        num_workers=config.train.num_workers,
        max_samples=config.data.max_samples,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )[:2]


def _parameter_fingerprint(model: nn.Module) -> str:
    names = sorted(name for name, _ in model.named_parameters())
    return hashlib.md5(",".join(names).encode("utf-8")).hexdigest()


def build_model(config: ExperimentConfig, device: torch.device) -> Any:
    """Build the SpatialMoESODNet and verify parameter consistency across ranks."""
    world_size = get_world_size()

    model = SpatialMoESODNet(
        use_deep_supervision=config.model.deep_supervision,
        num_experts=config.model.num_experts,
        window_size=config.model.window_size,
        router_noise_enabled=config.model.router_noise_enabled,
        router_noise_scale=config.model.router_noise_scale,
        router_noise_min_std=config.model.router_noise_min_std,
        moe_16_mode=config.model.moe_16_mode,
    ).to(device)

    local_hash = _parameter_fingerprint(model)
    ht = torch.zeros(1, dtype=torch.int64, device=device)
    ht[0] = int(local_hash[:15], 16)
    gh = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(gh, ht)
    for i in range(world_size):
        if gh[i].item() != ht[0].item():
            raise RuntimeError(f"Architecture mismatch! Rank 0 != Rank {i}")

    bn_count = sum(1 for _, m in model.named_modules()
                   if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)))
    if is_rank_zero():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Architecture OK. Total: {total:,} | Trainable: {trainable:,}")
        log.info(f"Found {bn_count} BatchNorm layers." if bn_count
                 else "No BatchNorm layers detected. SyncBatchNorm disabled.")

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[get_local_rank()], output_device=get_local_rank(),
        find_unused_parameters=True,
    )
    return model


def build_optimizer_and_criterion(
    config: ExperimentConfig,
    model: Any,
    train_loader: Any,
    device: torch.device,
) -> Tuple[OptimizationEngine, CombinedLoss]:
    """Construct optimizer, LR scheduler, and the loss object."""
    groups = get_parameter_groups(model.module)
    optimizer = AdamW(groups)

    total_steps_t = torch.tensor(
        [len(train_loader) // config.opt.grad_accum_steps * config.train.epochs],
        device=device,
    )
    dist.broadcast(total_steps_t, src=0)
    total_steps = int(total_steps_t.item())
    warmup_steps = int(config.opt.warmup_ratio * total_steps)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps,
                                      total_steps=total_steps)
    engine = OptimizationEngine(model, optimizer, scheduler,
                                amp_enabled=config.opt.amp, amp_dtype=torch.float16)

    lb_weights = config.loss.load_balance_weights
    if lb_weights is None:
        import warnings
        warnings.warn("Using scalar load_balance_weight is deprecated. Set load_balance_weights in config.", DeprecationWarning, stacklevel=2)
        lb_weights = [config.loss.load_balance_weight] * 3

    criterion = CombinedLoss(
        LossConfig(
            bce_weight=config.loss.bce_weight,
            iou_weight=config.loss.iou_weight,
            ssim_weight=config.loss.ssim_weight,
            boundary_weight=config.loss.boundary_weight,
            load_balance_weight=config.loss.load_balance_weight,
            importance_weight=config.loss.importance_weight,
            z_loss_weight=config.loss.z_loss_weight,
            aux_boundary_weight=config.loss.aux_boundary_weight,
            deep_supervision_weight=config.loss.deep_supervision_weight,
            load_balance_weights=lb_weights,
        )
    )
    return engine, criterion


# ---------------------------------------------------------------------------
# Workspace resolution + resume
# ---------------------------------------------------------------------------

def _seed_everything(config: ExperimentConfig, seed_rank_offset: int) -> None:
    rank_seed = config.train.seed + seed_rank_offset
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)


def resolve_workspace(
    args: Any,
    config: ExperimentConfig,
    project_root: str,
) -> Tuple[str, str, Any]:
    """Return ``(base_dir, cfg_hash, hf_pusher)`` for this run."""
    from src.train_ddp import get_config_hash

    run_dir = ""
    if is_rank_zero():
        run_dir = setup_experiment_run(config, dry_run=False)
        update_registry_status("experiments", config.run_id, "RUNNING")
        log.info(f"EXPERIMENT INITIALIZED: {config.experiment_id}  "
                 f"RUN: {config.run_id}  DIR: {run_dir}")
    dist.barrier()
    dist.broadcast_object_list([run_dir], src=0)

    cfg_hash = get_config_hash(config.to_dict(), "model_config_hash")

    if args.preflight:
        base_dir = os.path.join(default_preflight_base(project_root), config.experiment_id)
        args.max_optimizer_steps = args.max_optimizer_steps or 5
    else:
        base_dir = os.path.join(default_checkpoint_base(project_root), config.experiment_id)
        if os.path.exists(os.path.join(base_dir, "training_complete.json")) and not args.overwrite:
            raise RuntimeError("TRAINING ALREADY COMPLETE. Use --overwrite to bypass.")
        if os.path.exists(os.path.join(base_dir, "best.pth")) and not args.overwrite and not args.resume:
            raise RuntimeError("Production checkpoints exist. Use --overwrite to bypass.")

    os.makedirs(base_dir, exist_ok=True)
    if is_rank_zero():
        with open(os.path.join(base_dir, "final_config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=4)
        with open(os.path.join(base_dir, "final_config_sha256"), "w") as f:
            f.write(cfg_hash)

    hf_pusher = None
    if is_rank_zero() and not args.preflight and os.environ.get("HF_REPO_ID"):
        hf_pusher = hf_sync.AsyncCheckpointPusher()

    _seed_everything(config, get_rank())
    return base_dir, cfg_hash, hf_pusher


def apply_resume(
    args: Any,
    config: ExperimentConfig,
    model: Any,
    engine: OptimizationEngine,
    base_dir: str,
    device: torch.device,
    cfg_hash: str,
    project_root: str,
) -> Tuple[int, int, float, int, int]:
    """Restore model+optimizer state from ``--resume`` if requested."""
    if not args.resume:
        return 0, 0, float("inf"), 0, 0

    ckpt_path = args.resume
    if args.resume == "latest":
        ckpt_path = os.path.join(
            default_preflight_base(project_root) if args.preflight else base_dir,
            "latest.pth",
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if is_rank_zero():
        log.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path)
    validate_checkpoint_fields(
        ckpt,
        expected_version=CHECKPOINT_FORMAT_VERSION,
        expected_world_size=get_world_size(),
        expected_config_hash=cfg_hash,
    )
    model.module.load_state_dict(ckpt["model_state_dict"])
    engine.load_state_dict(ckpt["engine_state_dict"])
    if "rng_states" in ckpt:
        set_rng_states(ckpt["rng_states"], device)
    dist.barrier()
    if is_rank_zero():
        log.info(f"RESUME from {ckpt_path} | epoch={ckpt['epoch']} "
                 f"batch={ckpt['batch_in_epoch']} step={engine.global_step} "
                 f"best={ckpt['best_metric']:.4f}")
    return (
        ckpt["epoch"],
        ckpt["batch_in_epoch"],
        ckpt["best_metric"],
        0,
        0,
    )


# ---------------------------------------------------------------------------
# TrainCtx + top-level init
# ---------------------------------------------------------------------------

class TrainCtx(NamedTuple):
    """Bundled state handed to the per-epoch training loop."""

    config: ExperimentConfig
    model: Any
    engine: OptimizationEngine
    criterion: CombinedLoss
    train_loader: Any
    val_loader: Any
    train_sampler: Any
    device: torch.device
    rank: int
    world_size: int
    checkpoint_dir: str
    cfg_hash: str
    hf_pusher: Any
    start_epoch: int
    start_batch: int
    best_mae: float
    best_mae_epoch: int
    best_mae_step: int


def init_training(args: Any, project_root: str) -> TrainCtx:
    """Construct everything needed to start (or resume) training."""
    rank = get_rank()
    world_size = get_world_size()

    torch.cuda.set_device(get_local_rank())
    device = torch.device(f"cuda:{get_local_rank()}")

    config = ExperimentConfig.load(args.config)
    if args.max_epochs is not None:
        config.train.epochs = min(config.train.epochs, args.max_epochs)
        if is_rank_zero():
            log.info(f"[max_epochs] Capped training to {config.train.epochs} epochs")

    config.validate()  # Raises ValueError loudly for top_k > num_experts, bad moe_type, etc.

    data_dir = config.data.dataset_root
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"DATA_ROOT could not be resolved: {data_dir}")
    if world_size < 2 and not (args.dry_run or args.preflight):
        raise RuntimeError(f"Requires >=2 GPUs. Found {world_size}.")

    if args.dry_run:
        if is_rank_zero():
            setup_experiment_run(config, dry_run=True)
        raise SystemExit(0)

    base_dir, cfg_hash, hf_pusher = resolve_workspace(args, config, project_root)

    train_loader, val_loader = build_dataloaders(config, rank, world_size)
    model = build_model(config, device)
    if is_rank_zero():
        total = sum(p.numel() for p in model.parameters())
        with open(os.path.join(base_dir, "run_info.json"), "w") as f:
            json.dump({"total_params": total}, f, indent=4)
    engine, criterion = build_optimizer_and_criterion(config, model, train_loader, device)

    start_epoch, start_batch, best_mae, best_epoch, best_step = apply_resume(
        args, config, model, engine, base_dir, device, cfg_hash, project_root,
    )

    return TrainCtx(
        config=config, model=model, engine=engine, criterion=criterion,
        train_loader=train_loader, val_loader=val_loader,
        train_sampler=train_loader.sampler, device=device,
        rank=rank, world_size=world_size, checkpoint_dir=base_dir,
        cfg_hash=cfg_hash, hf_pusher=hf_pusher,
        start_epoch=start_epoch, start_batch=start_batch,
        best_mae=best_mae, best_mae_epoch=best_epoch, best_mae_step=best_step,
    )
