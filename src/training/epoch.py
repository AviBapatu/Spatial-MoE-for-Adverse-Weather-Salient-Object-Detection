"""Per-epoch training: one epoch of batches + validation/diagnostics.

This module owns the inner training loop for a single epoch.  It has no
knowledge of argparse; it consumes a ``TrainCtx`` (from ``setup``) and the
parsed ``args`` object it needs.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch.amp import autocast

from src.log import get_logger
from src.optimization import freeze_backbone, unfreeze_backbone
from src.training.distributed import is_rank_zero, should_stop
from src.training.loop import (
    distributed_diagnostics,
    distributed_validate,
    maybe_save_periodic_checkpoint,
    run_counterfactual_ablation,
)
from src.training.setup import (
    TrainCtx,
    device_memory_diagnostics,
    should_run_diagnostics,
)

log = get_logger(__name__)


def _start_epoch(ctx: TrainCtx, epoch: int, args: Any) -> None:
    """Put the model in train mode, set epoch-based samplers, tune backbone."""
    model = ctx.model.module
    model.train()
    if hasattr(ctx.train_sampler, "set_epoch"):
        ctx.train_sampler.set_epoch(epoch)
    if hasattr(ctx.train_loader.dataset, "set_epoch"):
        ctx.train_loader.dataset.set_epoch(epoch)

    if epoch == 0 and not args.resume:
        if is_rank_zero():
            log.info("Warmup Phase: Backbone Frozen")
        freeze_backbone(model)
    elif epoch == 1:
        if is_rank_zero():
            log.info("Phase 2: Backbone Unfrozen")
        unfreeze_backbone(model)


def _fast_forward(iterator: Any, epoch: int, ctx: TrainCtx) -> int:
    """Skip already-seen batches when resuming mid-epoch; return start index."""
    if epoch == ctx.start_epoch and ctx.start_batch > 0:
        if is_rank_zero():
            log.info(f"Fast-forwarding {ctx.start_batch} batches...")
        for _ in range(ctx.start_batch):
            next(iterator)
        return ctx.start_batch
    return 0


def _load_batch(batch: Any, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Move one batch's tensors to the device."""
    return (
        batch["image"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
        batch["edge_map"].to(device, non_blocking=True),
        batch["pad_mask"].to(device, non_blocking=True),
    )


def _forward_backward(
    model: Any,
    engine: Any,
    criterion: Any,
    images: torch.Tensor,
    masks: torch.Tensor,
    edges: torch.Tensor,
    pad_masks: torch.Tensor,
    config: Any,
    is_final: bool,
) -> torch.Tensor:
    """Run the forward pass and (scaled) backward; returns the batch loss."""
    with (model.no_sync() if not is_final else _null_mgr()):
        with autocast(device_type="cuda", dtype=torch.float16):
            out, moe_outputs = model(images)
            loss, loss_dict = criterion(
                saliency_logits=out.saliency_logits,
                edge_logits=out.boundary_logits, moe_outputs=moe_outputs,
                target=masks, gt_boundary=edges,
                aux_logits_16=out.aux_logits_16, aux_logits_8=out.aux_logits_8,
                aux_logits_4=out.aux_logits_4, pad_mask=pad_masks,
            )
            loss = loss / config.opt.grad_accum_steps
        engine.scaler.scale(loss).backward()
    return loss


def _null_mgr():
    from contextlib import nullcontext
    return nullcontext()


def _zero_frozen_backbone_grads(model: Any, epoch: int, resume: bool) -> None:
    """Drop backbone grads during the epoch-0 frozen warmup-phase step."""
    if epoch == 0 and not resume:
        for name, param in model.named_parameters():
            if name.startswith("backbone") and param.grad is not None:
                param.grad = None


def _should_break(args: Any, engine: Any, ctx: TrainCtx, is_final: bool) -> bool:
    """Return True when the batch loop must stop before processing more."""
    if args.max_optimizer_steps and engine.global_step >= args.max_optimizer_steps:
        return True
    if should_stop():
        log.warning(f"Rank {ctx.rank} stopping at step {engine.global_step}")
        return True
    if args.calibration and is_final and engine.global_step >= 5:
        _, _, peak_alloc, peak_res = device_memory_diagnostics(ctx.device)
        log.warning(f"Calibration Rank {ctx.rank}: peak_alloc={peak_alloc:.2f}GB "
                    f"peak_res={peak_res:.2f}GB")
        dist.barrier()
        return True
    return False


def _train_one_batch(
    iterator: Any,
    model: Any,
    engine: Any,
    criterion: Any,
    device: torch.device,
    config: Any,
    ctx: TrainCtx,
    epoch: int,
    batch_idx: int,
    args: Any,
) -> Tuple[int, bool]:
    """Process a single batch; returns ``(next_batch_idx, done)``."""
    try:
        batch = next(iterator)
    except StopIteration:
        return batch_idx, True

    images, masks, edges, pad_masks = _load_batch(batch, device)
    is_final = ((batch_idx + 1) % config.opt.grad_accum_steps == 0
                or (batch_idx + 1) == len(ctx.train_loader))

    loss = _forward_backward(model, engine, criterion, images, masks, edges,
                             pad_masks, config, is_final)

    if is_final:
        _zero_frozen_backbone_grads(model.module, epoch, args.resume)
        overflow, _ = engine.step()
        engine.optimizer.zero_grad()
        if is_rank_zero() and overflow:
            log.warning(f"[Epoch {epoch+1} Batch {batch_idx+1}] AMP overflow!")

        maybe_save_periodic_checkpoint(
            engine=engine, model=model, config=config,
            config_hash=ctx.cfg_hash, world_size=ctx.world_size,
            device=device, checkpoint_dir=ctx.checkpoint_dir,
            epoch=epoch, batch_idx=batch_idx,
            best_mae=ctx.best_mae, rank=ctx.rank, hf_pusher=ctx.hf_pusher,
        )

    if _should_break(args, engine, ctx, is_final):
        return batch_idx, True

    if is_rank_zero() and (batch_idx + 1) % 50 == 0:
        log.info(f"[Train] Ep {epoch+1} | {batch_idx+1}/{len(ctx.train_loader)} | "
                 f"Loss {loss.item()*config.opt.grad_accum_steps:.4f} | "
                 f"Step {engine.global_step}")

    return batch_idx + 1, False


def run_train_batches(
    iterator: Any,
    model: Any,
    engine: Any,
    criterion: Any,
    device: torch.device,
    config: Any,
    ctx: TrainCtx,
    epoch: int,
    start_batch: int,
    args: Any,
) -> None:
    """Iterate the train loader until the epoch ends or early exit triggers."""
    batch_idx = start_batch
    while True:
        try:
            batch_idx, done = _train_one_batch(
                iterator, model, engine, criterion, device, config, ctx,
                epoch, batch_idx, args,
            )
            if done:
                return
        except RuntimeError as e:
            log.error(f"Runtime error: {e}")
            raise


def _validate_and_diagnostics(ctx: TrainCtx, epoch: int, args: Any, wall_clock_s: float) -> TrainCtx:
    """Run distributed validation + periodic diagnostics; return updated ctx."""
    engine, model = ctx.engine, ctx.model
    val_loader, device = ctx.val_loader, ctx.device
    config = ctx.config

    val_mae, val_fbeta, best_mae, best_epoch, best_step = distributed_validate(
        model=model, val_loader=val_loader, device=device,
        epoch=epoch, total_epochs=config.train.epochs,
        checkpoint_dir=ctx.checkpoint_dir, config=config,
        config_hash=ctx.cfg_hash, world_size=ctx.world_size,
        rank=ctx.rank, engine=engine,
        best_mae=ctx.best_mae, best_mae_epoch=ctx.best_mae_epoch,
        best_mae_step=ctx.best_mae_step, hf_pusher=ctx.hf_pusher,
    )

    if is_rank_zero():
        import json
        import os
        metrics_path = os.path.join(ctx.checkpoint_dir, "epoch_metrics.json")
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = []
        metrics.append({
            "epoch": epoch + 1,
            "wall_clock_s": wall_clock_s,
            "val": {"MAE": val_mae, "S_measure": val_fbeta}
        })
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)


    if best_mae < ctx.best_mae:
        ctx = ctx._replace(best_mae=best_mae, best_mae_epoch=best_epoch,
                           best_mae_step=best_step)

    if should_run_diagnostics(epoch + 1):
        distributed_diagnostics(
            model=model, val_loader=val_loader, device=device, epoch=epoch + 1,
            checkpoint_dir=ctx.checkpoint_dir, config=config,
            rank=ctx.rank, world_size=ctx.world_size, hf_pusher=ctx.hf_pusher,
        )

    expert_ablation = getattr(args, "expert_ablation", None)
    if expert_ablation is not None:
        run_counterfactual_ablation(model, val_loader, device, expert_ablation)

    return ctx


def run_epoch(ctx: TrainCtx, epoch: int, args: Any) -> TrainCtx:
    """Execute one full epoch (training + optional validation/diagnostics).

    Returns a possibly-updated ``TrainCtx`` with fresh best-metric fields.
    """
    model, engine, criterion = ctx.model, ctx.engine, ctx.criterion
    device, config = ctx.device, ctx.config

    import time
    epoch_start_time = time.perf_counter()

    _start_epoch(ctx, epoch, args)
    iterator = iter(ctx.train_loader)
    start_batch = _fast_forward(iterator, epoch, ctx)

    if is_rank_zero():
        log.info(f"--- Epoch {epoch + 1}/{config.train.epochs} [Train] ---")
    engine.optimizer.zero_grad()

    run_train_batches(iterator, model, engine, criterion, device, config, ctx,
                      epoch, start_batch, args)

    if args.calibration or should_stop() or (
        args.max_optimizer_steps and engine.global_step >= args.max_optimizer_steps
    ):
        return ctx

    dist.barrier()
    wall_clock_s = time.perf_counter() - epoch_start_time
    return _validate_and_diagnostics(ctx, epoch, args, wall_clock_s)
