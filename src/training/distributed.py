"""Distributed-training helpers: process-group lifecycle, rank utilities,
monitored barrier wrapper, and graceful-signal handling.
"""
from __future__ import annotations

import os
import signal
from datetime import timedelta
from typing import List

import torch
import torch.distributed as dist

from src.log import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Process group lifecycle
# ---------------------------------------------------------------------------

_monitor_group = None  # secondary GLOO group, used only for monitored_barrier


def init_process_group(
    backend: str = "nccl",
    timeout_minutes: int = 45,
) -> None:
    """Initialise the default process group with an explicit timeout.

    NCCL defaults to 600 s which is dangerously long — a rank-0 stall can
    silently wedge the whole job for 10 minutes before a cryptic SIGABRT.
    We use a 45-minute ceiling as a hard safety net; the real fix is making
    rank-0-only blocks provably short (see ``monitored_barrier``).

    ``dist.monitored_barrier`` is only implemented for the GLOO backend, so
    when the main backend is NCCL (the normal case for GPU training) we
    additionally stand up a small secondary GLOO group that exists purely
    to back ``monitored_barrier`` calls. The NCCL group remains the default
    for all actual tensor collectives (all-reduce, all-gather, DDP, etc.).

    ``device_id`` is passed explicitly so NCCL does not have to guess the
    GPU from the global rank.  Guessing is correct on standard single-node
    runs but emits a noisy warning and can hang on heterogeneous GPU
    assignments.  We read LOCAL_RANK from the environment (set by torchrun)
    and only pass ``device_id`` when CUDA is available.
    """
    if dist.is_initialized():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_id = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None

    dist.init_process_group(
        backend,
        init_method="env://",
        timeout=timedelta(minutes=timeout_minutes),
        device_id=device_id,
    )

    global _monitor_group
    if backend != "gloo":
        _monitor_group = dist.new_group(
            backend="gloo",
            timeout=timedelta(minutes=timeout_minutes),
        )


def destroy_process_group_safe() -> None:
    """Tear down the process group(s) if still live."""
    global _monitor_group
    if dist.is_initialized():
        dist.destroy_process_group()
    _monitor_group = None


# ---------------------------------------------------------------------------
# Rank helpers
# ---------------------------------------------------------------------------

def get_rank() -> int:
    """Return the global rank (0 when not distributed)."""
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


def get_local_rank() -> int:
    """Return the local rank within this node."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Return the total number of ranks."""
    return int(os.environ.get("WORLD_SIZE", 1))


def is_rank_zero() -> bool:
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Monitored barrier — catches rank-0-only stalls early
# ---------------------------------------------------------------------------

def monitored_barrier(timeout_minutes: int = 5) -> None:
    """All ranks call this immediately after a rank-0-only block.

    If rank 0 is stuck (e.g. doing I/O), the other ranks will time out
    here with a clear ``DistBackendError: Monitor trust barrier`` message
    instead of hanging silently until a later NCCL timeout fires.
    """
    if not dist.is_initialized():
        return
    dist.monitored_barrier(
        group=_monitor_group,
        timeout=timedelta(minutes=timeout_minutes),
    )


# ---------------------------------------------------------------------------
# Graceful-signal handling
# ---------------------------------------------------------------------------

_stop_flag: List[bool] = [False]


def _handle_signal(sig: int, _frame: object) -> None:
    rank = get_rank()
    log.warning(f"Rank {rank} received signal {sig}. Graceful shutdown initiated...")
    _stop_flag[0] = True


def install_signal_handlers() -> None:
    """Install SIGINT / SIGTERM handlers for graceful mid-epoch shutdown."""
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def should_stop() -> bool:
    """Check whether a termination signal was received."""
    return _stop_flag[0]


# ---------------------------------------------------------------------------
# RNG state save / restore (needed for exact-resume reproducibility)
# ---------------------------------------------------------------------------

def get_rng_states(device: torch.device) -> dict:
    """Snapshot Python, NumPy, and CUDA RNG states."""
    import random

    import numpy as np
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def set_rng_states(states: dict, device: torch.device) -> None:
    """Restore RNG states from a previous snapshot."""
    import random

    import numpy as np
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    torch.cuda.set_rng_state(states["torch_cuda"], device)
