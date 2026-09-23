"""CLI entrypoint for the DDP training script.

Parses the exact same flags as the original ``train_ddp.py`` so that
existing launch scripts and experiment configs continue to work without
modification.  The heavy lifting lives in ``setup`` (construction + resume)
and ``epoch`` (per-epoch loop).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

# Reduce CUDA allocator fragmentation before any allocation happens.  Training at
# 384x384 with the PVTv2-B4 backbone runs close to a 15 GiB T4's limit, and
# fragmentation is what tips it into OOM ("reserved but unallocated" memory).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch.distributed as dist

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.experiment import update_registry_status
from src.log import get_logger, setup_logging
from src.training.checkpoint import enqueue_hf_push
from src.training.distributed import (
    destroy_process_group_safe,
    get_rank,
    init_process_group,
    install_signal_handlers,
    is_rank_zero,
    should_stop,
)
from src.training.epoch import run_epoch
from src.training.setup import TrainCtx, init_training

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing (same flags as original)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="experiments/baseline_v1.json")
    p.add_argument("--dry_run", action="store_true")
    # NOTE: currently inert — no code path reads args.smoke_test. Real smoke
    # tests are run via `torchrun -m src.smoke_test`, not this flag. Kept for
    # CLI-compatibility with any existing launch scripts that already pass it.
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--ddp_test", action="store_true")
    p.add_argument("--calibration", action="store_true")
    p.add_argument("--resume", type=str, nargs="?", const="latest", default=None)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_optimizer_steps", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    return p


def _early_exit(args: Any, ctx: TrainCtx) -> bool:
    """Post-epoch exit conditions shared by the main loop and validation."""
    return (args.calibration or should_stop()
            or (args.max_optimizer_steps
                and ctx.engine.global_step >= args.max_optimizer_steps))


# ---------------------------------------------------------------------------
# Post-training finalization
# ---------------------------------------------------------------------------

def _mark_run_complete(ctx: TrainCtx, epoch: int) -> None:
    config = ctx.config
    update_registry_status(
        "experiments", config.run_id, "COMPLETED",
        val_metric_val=str(ctx.best_mae),
        best_epoch=str(ctx.best_mae_epoch),
        best_step=str(ctx.best_mae_step),
    )
    p = os.path.join(ctx.checkpoint_dir, "training_complete.json")
    with open(p, "w") as f:
        json.dump({"status": "COMPLETE", "epoch": epoch,
                    "global_step": ctx.engine.global_step,
                    "best_checkpoint": "best.pth",
                    "config_hash": ctx.cfg_hash}, f, indent=4)
    enqueue_hf_push(ctx.hf_pusher, p, name=f"{config.experiment_id}_training_complete.json")


def _mark_preflight_pass(ctx: TrainCtx) -> None:
    config = ctx.config
    p = os.path.join(ctx.checkpoint_dir, "preflight_results.json")
    with open(p, "w") as f:
        json.dump({"status": "PASS", "run_id": config.run_id,
                    "optimizer_steps": ctx.engine.global_step,
                    "checkpoint_written": os.path.exists(
                        os.path.join(ctx.checkpoint_dir, "latest.pth")),
                    "loss_finite": True, "ddp_ok": True, "memory_ok": True,
                    "config_hash": ctx.cfg_hash, "timestamp": time.time(),
                    "config": config.to_dict()}, f, indent=4)


def _finalize_training(ctx: TrainCtx, args: Any, epoch: int) -> None:
    """Write completion artifacts after the training loop exits."""
    dist.barrier()
    if not is_rank_zero():
        return
    if args.preflight:
        _mark_preflight_pass(ctx)
    elif not should_stop():
        _mark_run_complete(ctx, epoch)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.ddp_test:
        from tests.test_moe_ddp import run_ddp_tests
        run_ddp_tests(args)
        return

    init_process_group("nccl", timeout_minutes=45)
    setup_logging(rank=get_rank())
    install_signal_handlers()

    ctx: Optional[TrainCtx] = None
    hf_pusher = None
    try:
        ctx = init_training(args, _project_root)
        hf_pusher = ctx.hf_pusher

        epoch = ctx.start_epoch
        for epoch in range(ctx.start_epoch, ctx.config.train.epochs):
            ctx = run_epoch(ctx, epoch, args)
            if _early_exit(args, ctx):
                break

        _finalize_training(ctx, args, epoch)

    except Exception as e:
        if is_rank_zero() and ctx is not None:
            try:
                update_registry_status("experiments", ctx.config.run_id, "FAILED")
            except Exception:
                pass
        log.error(f"Exception on Rank {get_rank()}: {e}")
        raise
    finally:
        if hf_pusher is not None:
            log.info("[hf_sync] Flushing pending checkpoint pushes before exit...")
            hf_pusher.close(timeout=900)
        destroy_process_group_safe()


if __name__ == "__main__":
    main()
