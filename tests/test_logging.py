"""Logging setup: exactly one handler, rank-aware filtering, short INFO lines.

The bug these pin down: ``get_logger`` used to attach a second, unfiltered
handler at import time, so the same message was emitted once per rank, in a
different format from ``setup_logging``'s — which is what made the Kaggle logs
unreadable.
"""
import io
import logging
import sys

import pytest

import src.log as log_module
# Imported at collection time on purpose: importing it installs the shared handler,
# and doing that inside a test would bind it to the real stdout before the fixture runs.
from src.training import epoch as epoch_module


@pytest.fixture(autouse=True)
def clean_logging():
    """Start and end every test from an unconfigured logging module."""
    root = logging.getLogger()
    saved = root.handlers[:]
    log_module._initialized = False
    log_module._handler = None
    root.handlers.clear()
    yield
    root.handlers.clear()
    root.handlers.extend(saved)
    log_module._initialized = False
    log_module._handler = None


def _setup_capturing(monkeypatch, world: int, rank: int) -> io.StringIO:
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("RANK", str(rank))
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    log_module.setup_logging()
    return buf


def test_repeated_setup_keeps_exactly_one_handler(monkeypatch):
    _setup_capturing(monkeypatch, 1, 0)
    log_module.setup_logging()
    log_module.setup_logging(rank=0)
    assert len(logging.getLogger().handlers) == 1


def test_info_line_is_short_and_carries_no_rank_tag_when_single_process(monkeypatch):
    buf = _setup_capturing(monkeypatch, 1, 0)
    log_module.get_logger("src.training.setup").info("Architecture OK")
    line = buf.getvalue().strip()
    assert line.endswith("src.training.setup: Architecture OK")
    assert "[r" not in line and "INFO" not in line


def test_other_ranks_drop_info_but_keep_warnings(monkeypatch):
    buf = _setup_capturing(monkeypatch, 2, 1)
    logger = log_module.get_logger("src.training.epoch")
    logger.info("must not appear")
    logger.warning("must appear")
    out = buf.getvalue()
    assert "must not appear" not in out
    assert "must appear" in out


def test_rank_zero_of_two_shows_the_rank_tag(monkeypatch):
    buf = _setup_capturing(monkeypatch, 2, 0)
    log_module.get_logger("src.training.setup").info("hello")
    assert "[r0]" in buf.getvalue()


def test_get_logger_without_setup_uses_the_same_setup(monkeypatch):
    """The fallback path must not install a second, unfiltered handler."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    log_module.get_logger("some.module").info("filtered out")
    assert len(logging.getLogger().handlers) == 1
    assert "filtered out" not in buf.getvalue()


def test_chatty_third_party_loggers_are_pinned_to_warning(monkeypatch):
    _setup_capturing(monkeypatch, 1, 0)
    for name in log_module._QUIET_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


def test_amp_overflow_warns_once_per_epoch_then_summarises(monkeypatch):
    epoch_module._AMP_OVERFLOW_COUNTS.clear()
    epoch_module._AMP_LAST_EPOCH = -1

    buf = _setup_capturing(monkeypatch, 1, 0)
    for _ in range(4):
        epoch_module._note_amp_overflow(0)
    epoch_module._note_amp_overflow(1)

    out = buf.getvalue()
    # One warning per epoch (both epochs here), not one per skipped step.
    assert out.count("AMP overflow") == 2
    assert "AMP skipped 4 optimizer step(s)" in out
