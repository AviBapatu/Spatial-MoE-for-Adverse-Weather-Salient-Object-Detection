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


_rank: int = 0
_initialized: bool = False


def setup_logging(rank: int = 0, level: int = logging.INFO) -> None:
    """Configure the root logger. Safe to call more than once — rank is
    always updated, even if a handler is already installed."""
    global _rank, _initialized
    _rank = rank
    if _initialized:
        root = logging.getLogger()
        for h in root.handlers:
            for f in h.filters:
                if isinstance(f, _RankFilter):
                    f.rank = rank
        return
    _initialized = True
    fmt = logging.Formatter(
        fmt="[rank %(rank)s] %(name)s: %(message)s",
        datefmt=None,
    )
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(fmt)
    handler.addFilter(_RankFilter(rank))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a named logger.

    Call ``setup_logging()`` first so that rank-aware filtering is active.
    If ``setup_logging`` was not called yet (e.g. standalone CLI tool or
    unit tests), a fallback handler is attached automatically so messages
    are never silently lost.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__``.
    level:
        Optional per-logger override (e.g. ``logging.DEBUG`` for noisy
        modules).
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    if not _initialized and not logging.getLogger().handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
    return logger
