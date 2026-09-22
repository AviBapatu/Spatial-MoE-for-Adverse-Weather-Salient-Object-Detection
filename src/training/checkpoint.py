"""Checkpoint save / load / resume helpers.

Design rules:
    * Every new field in the checkpoint dict MUST be loaded via ``.get()``
      with a safe default so that old checkpoint files don't crash the loader.
    * ``torch.save`` / ``torch.load`` must never synchronously block the
      training hot-path for *verification*; integrity checks belong off the
      critical path (background HF-upload thread, separate step).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, Optional

import torch

CHECKPOINT_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    *,
    do_verify: bool = False,
) -> None:
    """Atomically save a checkpoint via tmp → replace.

    Parameters
    ----------
    state:
        Full checkpoint dict (model_state_dict, engine_state_dict, …).
    path:
        Final destination path (e.g. ``checkpoints/latest.pth``).
    do_verify:
        If True, load the tmp file back immediately for verification.
        **Disabled by default** — torch.save is reliable; verification
        belongs off the hot path (HF-sync background thread) or in tests.
    """
    tmp_path = path + ".tmp"
    backup_path = path.replace(".pth", "_prev.pth")

    torch.save(state, tmp_path)

    if do_verify:
        torch.load(tmp_path, map_location="cpu", weights_only=False)

    if os.path.exists(path):
        shutil.copy2(path, backup_path)

    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint dict from *path*.

    Raises ``FileNotFoundError`` if the path doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def validate_checkpoint_fields(
    ckpt: Dict[str, Any],
    *,
    expected_version: int,
    expected_world_size: int,
    expected_config_hash: str,
) -> None:
    """Strict field validation for checkpoint compatibility.

    Raises ``RuntimeError`` on any mismatch.
    """
    if ckpt.get("checkpoint_format_version") != expected_version:
        raise RuntimeError(
            f"Checkpoint version mismatch! Expected {expected_version}, "
            f"got {ckpt.get('checkpoint_format_version')}"
        )
    if ckpt.get("world_size") != expected_world_size:
        raise RuntimeError(
            f"World size mismatch! Checkpoint was {ckpt.get('world_size')}, "
            f"current is {expected_world_size}"
        )
    if ckpt.get("config_hash") != expected_config_hash:
        raise RuntimeError(
            "Configuration hash mismatch! Architecture or critical "
            "hyperparams changed."
        )


# ---------------------------------------------------------------------------
# Manifest + checkpoint-state helpers
# ---------------------------------------------------------------------------

def write_checkpoint_manifest(
    checkpoint_dir: str,
    filename: str,
    ckpt_path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a lightweight JSON manifest for a saved checkpoint."""
    manifest: Dict[str, Any] = {
        "checkpoint": filename,
        "size_bytes": os.path.getsize(ckpt_path),
        "timestamp": time.time(),
    }
    if extra:
        manifest.update(extra)
    manifest_path = os.path.join(checkpoint_dir, "checkpoint_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)


def build_checkpoint_state(
    *,
    model,
    engine,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    best_metric: float,
    config,
    config_hash: str,
    world_size: int,
    device: torch.device,
    rng_states: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct the canonical checkpoint dict."""
    state: Dict[str, Any] = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "model_state_dict": model.module.state_dict(),
        "engine_state_dict": engine.state_dict(),
        "rng_states": rng_states,
        "config": config.to_dict(),
        "config_hash": config_hash,
        "world_size": world_size,
        "timestamp": time.time(),
    }
    if extra:
        state.update(extra)
    return state


# ---------------------------------------------------------------------------
# Background HF-sync kickoff (delegates to src.hf_sync)
# ---------------------------------------------------------------------------

def enqueue_hf_push(
    hf_pusher: Any,
    local_path: str,
    name: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Non-blocking enqueue of a checkpoint to the HF background pusher.

    No-op when *hf_pusher* is ``None`` (e.g. local-only runs, preflight).
    """
    if hf_pusher is None:
        return
    hf_pusher.enqueue_checkpoint(local_path, name=name, extra_meta=extra_meta)
