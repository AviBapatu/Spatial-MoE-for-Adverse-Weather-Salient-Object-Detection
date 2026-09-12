#!/usr/bin/env python
"""Backward-compatible shim — ``torchrun -m src.train_ddp`` delegates here.

All real logic lives in src.training.cli; this file only exists so that
the canonical launch command ``torchrun -m src.train_ddp`` keeps working
after the modularisation.
"""
import copy
import hashlib

from src.training.checkpoint import CHECKPOINT_FORMAT_VERSION  # noqa: F401
from src.training.cli import main  # noqa: F401 — public entry point

# Re-export legacy symbols used by tests/test_moe_ddp.py
from src.training.distributed import get_rng_states, set_rng_states  # noqa: F401


def get_config_hash(config: dict, model_hash: str) -> str:
    """Canonical experiment identity hash (kept here for backward compat)."""
    core_config = copy.deepcopy({
        k: v for k, v in config.items()
        if k not in [
            "NUM_WORKERS", "CHECKPOINT_EVERY_N_STEPS",
            "experiment_id", "run_id", "batch_equivalence",
        ]
    })
    if "data" in core_config and "dataset_root" in core_config["data"]:
        del core_config["data"]["dataset_root"]
    hash_str = f"v{CHECKPOINT_FORMAT_VERSION}_{model_hash}_{sorted(core_config.items())}"
    return hashlib.md5(hash_str.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
