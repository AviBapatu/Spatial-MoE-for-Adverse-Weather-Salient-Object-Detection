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

import logging
import sys
from typing import Optional


class _RankFilter(logging.Filter):
    """Suppress INFO/DEBUG on non-rank-0 ranks."""

    def __init__(self, rank: int) -> None:
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        if self.rank == 0:
            return True
        return record.levelno >= logging.WARNING


_rank: int = 0
_initialized: bool = False


def setup_logging(rank: int = 0, level: int = logging.INFO) -> None:
    """Configure the root logger exactly once.

    Parameters
    ----------
    rank:
        Global rank of this process.  Non-zero ranks suppress INFO.
    level:
        Minimum severity to emit (default ``INFO``).
    """
    global _rank, _initialized
    if _initialized:
        return
    _rank = rank
    _initialized = True

    fmt = logging.Formatter(
        fmt=f"[rank {rank}] %(name)s: %(message)s",
        datefmt=None,
    )
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.addFilter(_RankFilter(rank))
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
