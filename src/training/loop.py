"""Per-epoch training step, distributed validation, and diagnostics.

All validation and diagnostics forward-passes run on **every rank**.
Metric aggregation happens via ``dist.all_gather_object``; only the final
numeric results are computed on rank 0.  This eliminates the rank-0-only
validation bottleneck that previously caused rank-1 to idle for minutes
each epoch.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import cv2
import torch
import torch.distributed as dist
from torch.amp import autocast

from src.log import get_logger
from src.metrics import SODMetrics
from src.training.checkpoint import (
    build_checkpoint_state,
    enqueue_hf_push,
    save_checkpoint,
    write_checkpoint_manifest,
)
from src.training.distributed import is_rank_zero, monitored_barrier

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Distributed validation
# ---------------------------------------------------------------------------

def distributed_validate(
    model: Any,
    val_loader: Any,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    *,
    checkpoint_dir: str,
    config: Any,
    config_hash: str,
    world_size: int,
    rank: int,
    engine: Any,
    best_mae: float,
    best_mae_epoch: int,
    best_mae_step: int,
    hf_pusher: Any,
) -> Tuple[float, float, float, int, int]:
    """Run validation on all ranks, aggregate on rank 0.

    Returns
    -------
    val_mae, val_fbeta, new_best_mae, new_best_epoch, new_best_step
    """
    model.eval()

    if is_rank_zero():
        log.info(
            f"--- Starting Epoch {epoch + 1}/{total_epochs} [Val] ---",
        )

    local_results: List[Dict[str, Any]] = []

    with torch.no_grad():
        for v_idx, v_batch in enumerate(val_loader):
            if is_rank_zero() and ((v_idx + 1) % 50 == 0 or (v_idx + 1) == len(val_loader)):
                log.info(
                    f"[Val] Epoch {epoch + 1} | Batch {v_idx + 1}/{len(val_loader)}",
                )

            v_images = v_batch["image"].to(device)
            with autocast(device_type="cuda", dtype=torch.float16):
                out, _ = model(v_images)
            sal_pred = torch.sigmoid(out.saliency_logits).detach().float().cpu().numpy()

            for i in range(v_images.size(0)):
                meta = {
                    k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i]
                    for k, v in v_batch["meta"].items()
                }
                p = sal_pred[i, 0]
                p_cropped = p[
                    meta["pad_top"] : meta["pad_top"] + meta["resized_h"],
                    meta["pad_left"] : meta["pad_left"] + meta["resized_w"],
                ]
                p_orig = cv2.resize(
                    p_cropped,
                    (meta["orig_w"], meta["orig_h"]),
                    interpolation=cv2.INTER_LINEAR,
                )

                gt_path = meta.get("gt_path")
                if not gt_path:
                    continue
                orig_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if orig_mask is None:
                    continue

                local_results.append(
                    {"pred": p_orig, "gt": orig_mask}
                )

    # ── Gather across ranks ──────────────────────────────────────────────
    gathered: List[List[Dict[str, Any]]] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(gathered, local_results)

    val_mae = float("inf")
    val_fbeta = 0.0

    if is_rank_zero():
        flat: List[Dict[str, Any]] = []
        for rank_results in gathered:
            flat.extend(rank_results)

        sod_metrics = SODMetrics()
        for item in flat:
            sod_metrics.step(item["pred"], item["gt"])
        results = sod_metrics.get_results()
        val_mae = results.get("MAE", float("inf"))
        val_fbeta = results.get("F_max", 0.0)
        log.info(
            f"Epoch {epoch + 1} | Val MAE: {val_mae:.4f} | "
            f"F-beta: {val_fbeta:.4f}",
        )

    # Broadcast final metric so all ranks agree on best_mae
    metric_tensor = torch.tensor([val_mae, val_fbeta], device=device)
    dist.broadcast(metric_tensor, src=0)
    val_mae = metric_tensor[0].item()
    val_fbeta = metric_tensor[1].item()

    new_best_mae = best_mae
    new_best_epoch = best_mae_epoch
    new_best_step = best_mae_step

    if val_mae < best_mae:
        new_best_mae = val_mae
        new_best_epoch = epoch + 1
        new_best_step = engine.global_step

        if is_rank_zero():
            state = build_checkpoint_state(
                model=model,
                engine=engine,
                epoch=epoch,
                batch_in_epoch=0,
                global_step=engine.global_step,
                best_metric=new_best_mae,
                config=config,
                config_hash=config_hash,
                world_size=world_size,
                device=device,
                rng_states=None,  # caller provides or we skip
            )
            final_path = f"{checkpoint_dir}/best.pth"
            save_checkpoint(state, final_path, do_verify=False)
            log.info(f"  -> New best model saved to {final_path}!")
            enqueue_hf_push(
                hf_pusher,
                final_path,
                name=f"{config.experiment_id}_best.pth",
                extra_meta={
                    "epoch": epoch + 1,
                    "global_step": engine.global_step,
                    "best_metric": new_best_mae,
                    "config_hash": config_hash,
                },
            )

    monitored_barrier(timeout_minutes=3)
    return val_mae, val_fbeta, new_best_mae, new_best_epoch, new_best_step


# ---------------------------------------------------------------------------
# Distributed diagnostics
# ---------------------------------------------------------------------------

def distributed_diagnostics(
    model: Any,
    val_loader: Any,
    device: torch.device,
    epoch: int,
    *,
    checkpoint_dir: str,
    config: Any,
    rank: int,
    world_size: int,
    hf_pusher: Any,
) -> None:
    """Run MoE diagnostics on all ranks; rank 0 writes output files.

    Each rank processes its own subset of the validation set (via the
    now-rank-aware ``val_loader``), updates its local tracker state,
    then we gather tracker stats to rank 0 for final aggregation.
    """
    from src.diagnostics import MoEDiagnosticsEngine

    if is_rank_zero():
        log.info(f"Running MoE Diagnostics for Epoch {epoch}...")

    diag_dir = f"{checkpoint_dir}/diagnostics"
    diag_engine = MoEDiagnosticsEngine(
        diag_dir,
        num_experts=config.model.num_experts,
        top_k=config.model.top_k,
    )

    with torch.no_grad():
        for d_idx, v_batch in enumerate(val_loader):
            if is_rank_zero() and ((d_idx + 1) % 50 == 0 or (d_idx + 1) == len(val_loader)):
                log.info(
                    f"[Diagnostics] Batch {d_idx + 1}/{len(val_loader)}",
                )
            v_images = v_batch["image"].to(device)
            with autocast(device_type="cuda", dtype=torch.float16):
                out, moe_outputs = model(v_images)
            skip_stages = [
                getattr(model.module.moe_4, 'is_dense', False),
                getattr(model.module.moe_8, 'is_dense', False),
                getattr(model.module.moe_16, 'is_dense', False),
            ]
            diag_engine.update(v_images, moe_outputs, v_batch["meta"], skip_stages=skip_stages)

    # Each rank finalize locally to get per-rank stats
    local_stats = diag_engine.finalize(epoch)

    # Gather stats dicts to rank 0 for logging / HF upload
    all_stats: List[Dict] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(all_stats, local_stats)

    if is_rank_zero():
        # Use the rank-0 stats as the canonical output (already written by finalize)
        log.info("Diagnostics completed safely. Check 'diagnostics' folder.")
        if hf_pusher is not None:
            diag_json = f"{diag_dir}/routing_stats_ep{epoch}.json"
            diag_csv = f"{diag_dir}/routing_stats_ep{epoch}.csv"
            if os.path.exists(diag_json):
                log.info(f"Uploading {diag_json} to Hugging Face...")
                enqueue_hf_push(
                    hf_pusher,
                    diag_json,
                    name=f"routing_stats_ep{epoch}.json",
                    extra_meta={"epoch": epoch, "type": "diagnostics"},
                )
            if os.path.exists(diag_csv):
                log.info(f"Uploading {diag_csv} to Hugging Face...")
                enqueue_hf_push(
                    hf_pusher,
                    diag_csv,
                    name=f"routing_stats_ep{epoch}.csv",
                    extra_meta={"epoch": epoch, "type": "diagnostics"},
                )

    monitored_barrier(timeout_minutes=2)


# ---------------------------------------------------------------------------
# Counterfactual expert ablation (rank-0-only, since it's an optional
# debug tool and doesn't affect training correctness)
# ---------------------------------------------------------------------------

def run_counterfactual_ablation(
    model: Any,
    val_loader: Any,
    device: torch.device,
    expert_ablation: str,
) -> None:
    """Run a single counterfactual expert ablation on rank 0."""
    from src.metrics import SODMetrics

    if not is_rank_zero():
        return

    scale, exp_id = map(int, expert_ablation.split(","))
    ablation_cfg = {"scale": scale, "expert_id": exp_id}
    log.info(f"COUNTERFACTUAL EXPERT ABLATION: Scale {scale}, Expert {exp_id}")

    ablation_metrics = SODMetrics()
    model.eval()
    with torch.no_grad():
        log.info("Starting Ablation Val")
        for a_idx, v_batch in enumerate(val_loader):
            if (a_idx + 1) % 50 == 0 or (a_idx + 1) == len(val_loader):
                log.info(f"[Ablation] Batch {a_idx + 1}/{len(val_loader)}")
            v_images = v_batch["image"].to(device)
            with autocast(device_type="cuda", dtype=torch.float16):
                out, _ = model.module(v_images, ablation_cfg=ablation_cfg)
            sal_pred = torch.sigmoid(out.saliency_logits).detach().float().cpu().numpy()

            for i in range(v_images.size(0)):
                meta = {
                    k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i]
                    for k, v in v_batch["meta"].items()
                }
                p = sal_pred[i, 0]
                p_cropped = p[
                    meta["pad_top"] : meta["pad_top"] + meta["resized_h"],
                    meta["pad_left"] : meta["pad_left"] + meta["resized_w"],
                ]
                p_orig = cv2.resize(
                    p_cropped,
                    (meta["orig_w"], meta["orig_h"]),
                    interpolation=cv2.INTER_LINEAR,
                )
                gt_path = meta.get("gt_path")
                if not gt_path:
                    continue
                orig_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if orig_mask is None:
                    continue
                ablation_metrics.step(p_orig, orig_mask)

    abl_results = ablation_metrics.get_results()
    log.info(
        f"Ablation Val MAE: {abl_results.get('MAE', float('inf')):.4f} | "
        f"F-beta: {abl_results.get('F_max', 0):.4f}"
    )


# ---------------------------------------------------------------------------
# Periodic checkpoint saving (inline-optimizer-steps)
# ---------------------------------------------------------------------------

def maybe_save_periodic_checkpoint(
    *,
    engine: Any,
    model: Any,
    config: Any,
    config_hash: str,
    world_size: int,
    device: torch.device,
    checkpoint_dir: str,
    epoch: int,
    batch_idx: int,
    best_mae: float,
    rank: int,
    hf_pusher: Any,
) -> None:
    """Save ``latest.pth`` every N optimizer steps (if due)."""
    if engine.global_step <= 0:
        return
    if engine.global_step % config.train.checkpoint_every_n_steps != 0:
        return

    dist.barrier()
    if is_rank_zero():
        state = build_checkpoint_state(
            model=model,
            engine=engine,
            epoch=epoch,
            batch_in_epoch=batch_idx + 1,
            global_step=engine.global_step,
            best_metric=best_mae,
            config=config,
            config_hash=config_hash,
            world_size=world_size,
            device=device,
            rng_states=None,
        )
        final_path = f"{checkpoint_dir}/latest.pth"
        save_checkpoint(state, final_path, do_verify=False)
        log.info(
            f"[Epoch {epoch + 1} Batch {batch_idx + 1}] "
            f"Checkpoint correctly promoted.",
        )
        write_checkpoint_manifest(
            checkpoint_dir,
            "latest.pth",
            final_path,
            extra={
                "epoch": epoch,
                "global_step": engine.global_step,
                "config_hash": config_hash,
            },
        )
        enqueue_hf_push(
            hf_pusher,
            final_path,
            name=f"{config.experiment_id}_latest.pth",
            extra_meta={
                "epoch": epoch,
                "global_step": engine.global_step,
                "config_hash": config_hash,
            },
        )
    dist.barrier()
