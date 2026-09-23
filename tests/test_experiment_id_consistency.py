"""Every rank must reach the same base_dir, so the experiment ID cannot be rank-0-only.

setup_experiment_run assigns config.experiment_id, but only rank 0 calls it. Rank 1
kept the empty string, making its base_dir the shared checkpoint root -- which is how
a run stopped on rank 1 with "Production checkpoints exist" while rank 0 proceeded,
and how a resume failed on rank 1 alone.

The ID is a pure function of the config, so the invariant to hold is that the value
assigned on rank 0 equals the value any other rank can compute for itself.
"""
from pathlib import Path

from src.config import ExperimentConfig
from src.experiment import generate_experiment_id, setup_experiment_run

CONFIG = "experiments/v_e8_repro_best.json"


def test_two_loads_of_the_same_config_agree():
    a = generate_experiment_id(ExperimentConfig.load(CONFIG))
    b = generate_experiment_id(ExperimentConfig.load(CONFIG))
    assert a and a == b


def test_the_rank_zero_assignment_matches_the_locally_derived_value(tmp_path: Path):
    cfg = ExperimentConfig.load(CONFIG)
    expected = generate_experiment_id(cfg)

    setup_experiment_run(cfg, base_dir=str(tmp_path), dry_run=True)

    assert cfg.experiment_id == expected, (
        "rank 0 would assign a different ID than the other ranks derive, so their "
        "base_dir would diverge"
    )
