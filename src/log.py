"""Centralized logging with rank-aware prefixing and filtering.

All modules should import ``get_logger`` from here and use it instead of
``print``.  Call :func:`setup_logging` once (from ``src.training.distributed``
or ``src.training.cli``) after the rank is known; then every ``get_logger``
call returns a logger that respects the configured rank.

Design:

* Only rank 0 logs at ``INFO`` level and above.  Other ranks suppress
  everything below ``WARNING``, keeping multi-GPU output readable.
* Messages are prefixed with ``[rank N]`` so that when a lower-level filter
  is temporarily loosened the source is still clear.
* The training hot path should never call ``log.info(...)`` on every rank
  per step; that would defeat the filter.  The ``rank==0`` guard is now
  handled centrally here instead of sprinkled across every call site.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

import numpy as np



class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and torch scalar types.

    Use as ``json.dump(data, f, cls=NumpyEncoder)`` wherever metric dicts
    (which contain ``numpy.float32`` / ``torch.Tensor`` values) are serialised.
    """

    def default(self, obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            import torch  # local import — avoids hard dep for non-GPU contexts
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


class _RankFilter(logging.Filter):
    """Suppress INFO/DEBUG on non-rank-0 ranks, and inject rank into records."""
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        if self.rank == 0:
            return True
        return record.levelno >= logging.WARNING


_QUIET_LOGGERS = (
    "timm",
    "huggingface_hub",
    "urllib3",
    "filelock",
    "PIL",
    "matplotlib",
    "torch.distributed",
    "torch.nn.parallel",
)

_rank: int = 0
_initialized: bool = False
_handler: Optional[logging.Handler] = None


class _NeatFormatter(logging.Formatter):
    """Short INFO lines; WARNING and above carry their level."""

    def __init__(self, show_rank: bool) -> None:
        super().__init__(datefmt="%H:%M:%S")
        prefix = "[r%(rank)s] " if show_rank else ""
        self._info = logging.Formatter(
            f"{prefix}%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
        self._loud = logging.Formatter(
            f"{prefix}%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "rank"):
            record.rank = _rank
        formatter = self._info if record.levelno == logging.INFO else self._loud
        return formatter.format(record)


def _env_rank() -> int:
    """Rank from the environment — valid before the process group exists."""
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def _env_world_size() -> int:
    """World size from the environment, defaulting to a single process."""
    value = os.environ.get("WORLD_SIZE")
    if not value:
        return 1
    try:
        return max(1, int(value))
    except ValueError:
        return 1


def setup_logging(rank: Optional[int] = None, level: int = logging.INFO) -> None:
    """Install the single rank-aware handler. Safe to call more than once.

    Parameters
    ----------
    rank:
        This process's global rank.  When omitted it is read from the
        environment, which is correct for ``torchrun``-launched processes.
    level:
        Root log level (default ``INFO``).
    """
    global _rank, _initialized, _handler
    _rank = _env_rank() if rank is None else rank
    _initialized = True

    if _handler is None:
        # stdout, not stderr: Kaggle renders stderr in red as if it were an error.
        _handler = logging.StreamHandler(stream=sys.stdout)
        _handler.setFormatter(_NeatFormatter(show_rank=_env_world_size() > 1))
        _handler.addFilter(_RankFilter(_rank))
    else:
        for f in _handler.filters:
            if isinstance(f, _RankFilter):
                f.rank = _rank

    root = logging.getLogger()
    root.handlers.clear()  # never stack handlers — never duplicate a line
    root.addHandler(_handler)
    root.setLevel(level)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a named logger, installing the shared handler if needed.

    Every module calls this at import time, so it must install the *same*
    rank-aware setup that :func:`setup_logging` does.  The previous version
    attached a second, bare-format handler instead, which is why every message
    appeared on every rank and in a different format in the Kaggle logs.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__``.
    level:
        Optional per-logger override (e.g. ``logging.DEBUG`` for noisy
        modules).
    """
    if not _initialized:
        setup_logging()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
