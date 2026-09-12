"""Tests for the training.distributed module and dataset rank-awareness.

Run with:
    uv run pytest tests/test_training_distributed.py -v

These tests use the gloo backend on CPU so they work without GPUs.
"""
from __future__ import annotations

import os
import tempfile
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_gloo(rank: int, world_size: int) -> None:
    """Init a gloo process group for CPU-based distributed tests."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_find_free_port())
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Test 1: val_loader is rank-aware when distributed=True
# ---------------------------------------------------------------------------

def _make_mock_dataset(n: int = 20, split_boundary: int = 16) -> mock.MagicMock:
    """Create a mock WXSODDataset with the attributes get_dataloaders needs.

    Stems use ``sceneID_weather`` format.  Indices 0..split_boundary-1 belong
    to scene ``s0`` and indices split_boundary..n-1 belong to ``s1`` so that
    the train/val disjointness assertion passes for the mocked 80/20 split.
    """
    ds = mock.MagicMock()
    ds.__len__ = mock.Mock(return_value=n)
    ds.samples = [
        (f"/fake/img_{i}.png", f"/fake/gt_{i}.png",
         f"s0_w{i}" if i < split_boundary else f"s1_w{i}")
        for i in range(n)
    ]
    ds.set_samples = mock.Mock(side_effect=lambda s: setattr(ds, "samples", s))
    ds.set_epoch = mock.Mock()
    return ds


def test_val_loader_rank_aware() -> None:
    """get_dataloaders must attach DistributedSampler to all four loaders
    when distributed=True."""
    from torch.utils.data.distributed import DistributedSampler

    dummy_ds = _make_mock_dataset(20)

    with mock.patch("src.dataset.WXSODDataset", return_value=dummy_ds), \
         mock.patch("src.dataset.verify_weather_distribution"), \
         mock.patch("src.dataset.GroupShuffleSplit") as MockGSS:
        MockGSS.return_value.split.return_value = iter([
            (list(range(16)), list(range(16, 20)))
        ])

        from src.dataset import get_dataloaders
        train_loader, val_loader, test_synth_loader, test_real_loader = get_dataloaders(
            root_dir="/nonexistent",
            image_size=64,
            batch_size=4,
            num_workers=0,
            distributed=True,
            rank=0,
            world_size=2,
        )

    assert isinstance(train_loader.sampler, DistributedSampler), (
        "train_loader missing DistributedSampler"
    )
    assert isinstance(val_loader.sampler, DistributedSampler), (
        "val_loader missing DistributedSampler when distributed=True"
    )
    assert isinstance(test_synth_loader.sampler, DistributedSampler), (
        "test_synth_loader missing DistributedSampler when distributed=True"
    )
    assert isinstance(test_real_loader.sampler, DistributedSampler), (
        "test_real_loader missing DistributedSampler when distributed=True"
    )


def test_val_loader_no_sampler_when_not_distributed() -> None:
    """Non-distributed mode should NOT attach samplers."""
    dummy_ds = _make_mock_dataset(20)

    with mock.patch("src.dataset.WXSODDataset", return_value=dummy_ds), \
         mock.patch("src.dataset.verify_weather_distribution"), \
         mock.patch("src.dataset.GroupShuffleSplit") as MockGSS:
        MockGSS.return_value.split.return_value = iter([
            (list(range(16)), list(range(16, 20)))
        ])

        from src.dataset import get_dataloaders
        _, val_loader, _, _ = get_dataloaders(
            root_dir="/nonexistent",
            image_size=64,
            batch_size=4,
            num_workers=0,
            distributed=False,
            rank=0,
            world_size=1,
        )

    from torch.utils.data.distributed import DistributedSampler
    assert not isinstance(val_loader.sampler, DistributedSampler), (
        "val_loader should NOT have DistributedSampler when distributed=False"
    )


# ---------------------------------------------------------------------------
# Test 2: 2-rank all_gather_object round-trip via gloo
# ---------------------------------------------------------------------------

def _all_gather_worker(rank: int, world_size: int, port: int) -> None:
    """Worker function spawned by mp.spawn for the all_gather test."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        local_data = {"rank": rank, "values": [rank * 10, rank * 10 + 1]}
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_data)

        for i in range(world_size):
            assert gathered[i] == {"rank": i, "values": [i * 10, i * 10 + 1]}, (
                f"Rank {rank}: gathered[{i}] = {gathered[i]}, expected rank={i}"
            )
    finally:
        dist.destroy_process_group()


def test_all_gather_object_roundtrip() -> None:
    """Verify dist.all_gather_object round-trips correctly with 2 gloo ranks."""
    world_size = 2
    port = _find_free_port()
    mp.spawn(_all_gather_worker, args=(world_size, port), nprocs=world_size, join=True)


# ---------------------------------------------------------------------------
# Test 3: checkpoint save/load round-trip (no verification in hot path)
# ---------------------------------------------------------------------------

def test_checkpoint_save_load_roundtrip() -> None:
    """save_checkpoint should produce a loadable file; do_verify=False by default."""
    from src.training.checkpoint import save_checkpoint, load_checkpoint

    state = {
        "model_state_dict": {"layer.weight": torch.randn(4, 4)},
        "global_step": 42,
        "epoch": 3,
    }
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(state, path, do_verify=False)
        loaded = load_checkpoint(path)
        assert loaded["global_step"] == 42
        assert loaded["epoch"] == 3
        assert torch.equal(
            loaded["model_state_dict"]["layer.weight"],
            state["model_state_dict"]["layer.weight"],
        )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 4: validate_checkpoint_fields rejects mismatched version
# ---------------------------------------------------------------------------

def test_validate_checkpoint_fields_rejects_bad_version() -> None:
    from src.training.checkpoint import validate_checkpoint_fields

    with pytest.raises(RuntimeError, match="version mismatch"):
        validate_checkpoint_fields(
            {"checkpoint_format_version": 999, "world_size": 2, "config_hash": "abc"},
            expected_version=1,
            expected_world_size=2,
            expected_config_hash="abc",
        )


# ---------------------------------------------------------------------------
# Test 5: distributed.py helper functions
# ---------------------------------------------------------------------------

def test_is_rank_zero_without_dist() -> None:
    """is_rank_zero should return True when no RANK env var is set."""
    from src.training.distributed import is_rank_zero
    env = os.environ.copy()
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    with mock.patch.dict(os.environ, env, clear=True):
        assert is_rank_zero() is True


def test_monitored_barrier_noop_without_dist() -> None:
    """monitored_barrier should be a no-op when dist is not initialized."""
    from src.training.distributed import monitored_barrier
    # Should not raise
    monitored_barrier(timeout_minutes=1)
