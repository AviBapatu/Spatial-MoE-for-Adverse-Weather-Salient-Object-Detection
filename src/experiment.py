"""Experiment lifecycle: ID generation, run scaffolding, and registry management.

This module owns three clearly-separated responsibilities:

1. **Identity** — Git/code provenance (``get_git_identity``) and experiment-ID
   generation (``generate_experiment_id``).
2. **Scaffolding** — Directory creation, config/code-identity persistence, and
   split-manifest verification (``setup_experiment_run``).
3. **Registry** — CSV-based run registry for tracking experiment status
   (``update_registry_status``).

Public function names are stable — ``train.py``, ``src/training/setup.py``,
``src/training/cli.py``, and ``tests/test_experiment.py`` depend on them.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict

from src.config import ExperimentConfig
from src.log import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. Identity — git provenance and experiment-ID generation
# ---------------------------------------------------------------------------


def get_git_identity(workspace_root: str) -> Dict[str, Any]:
    """Return a dict describing the current git state of *workspace_root*.

    Keys: ``git_commit``, ``git_branch``, ``git_dirty``, ``warning``.
    Falls back to a static source hash when ``.git/`` is absent.
    """
    identity: Dict[str, Any] = {
        "git_commit": "UNKNOWN",
        "git_branch": "UNKNOWN",
        "git_dirty": False,
        "warning": None,
    }

    if os.path.exists(os.path.join(workspace_root, ".git")):
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=workspace_root
                )
                .decode("utf-8")
                .strip()
            )
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=workspace_root,
                )
                .decode("utf-8")
                .strip()
            )
            status = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=workspace_root
                )
                .decode("utf-8")
                .strip()
            )
            identity["git_commit"] = commit
            identity["git_branch"] = branch
            identity["git_dirty"] = bool(status)
            if identity["git_dirty"]:
                identity["warning"] = "WORKING TREE NOT CLEAN"
        except Exception:
            pass
    else:
        try:
            src_dir = os.path.join(workspace_root, "src")
            hasher = hashlib.sha256()
            for root, _, files in sorted(os.walk(src_dir)):
                for f in sorted(files):
                    if f.endswith(".py"):
                        with open(os.path.join(root, f), "rb") as fp:
                            hasher.update(fp.read())
            identity["static_src_hash"] = hasher.hexdigest()
        except Exception:
            pass

    return identity


def generate_experiment_id(config: ExperimentConfig) -> str:
    """Build a short human-readable experiment ID from *config*.

    Example output: ``EXP_B4_E8_K2_S32_R1_L1_M_SPARSE``.
    """
    bb = config.model.backbone.upper().replace("PVT_V2_", "")
    e = config.model.num_experts
    k = config.model.top_k
    s = config.opt.effective_global_batch

    r_abbrev = "R_DEF"
    if config.model.router_variant == "token_only":
        r_abbrev = "R1"
    elif config.model.router_variant == "token_local":
        r_abbrev = "R2"
    elif config.model.router_variant == "global":
        r_abbrev = "R3"

    moe_abbrev = "M_SPARSE"
    if config.model.moe_type == "none":
        moe_abbrev = "M_NONE"
    elif config.model.moe_type == "dense":
        moe_abbrev = "M_DENSE"

    l_abbrev = "L1"
    if config.loss.ssim_weight > 0:
        l_abbrev = "L2"
    if config.loss.boundary_weight > 0:
        l_abbrev = "L3"

    return f"EXP_{bb}_E{e}_K{k}_S{s}_{r_abbrev}_{l_abbrev}_{moe_abbrev}"


# ---------------------------------------------------------------------------
# 2. Scaffolding — directory creation, persistence, manifest verification
# ---------------------------------------------------------------------------


def verify_split_manifest(config: ExperimentConfig) -> None:
    """Check that the split manifest exists and matches its recorded hash.

    Raises:
        FileNotFoundError: If the manifest path does not exist.
        ValueError: If the SHA-256 hash does not match.
    """
    if not config.data.split_manifest_path:
        return

    if not os.path.exists(config.data.split_manifest_path):
        raise FileNotFoundError(
            f"Split manifest not found at {config.data.split_manifest_path}"
        )

    with open(config.data.split_manifest_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    if config.data.split_manifest_hash and file_hash != config.data.split_manifest_hash:
        raise ValueError(
            f"Split manifest hash mismatch! "
            f"Expected {config.data.split_manifest_hash}, got {file_hash}"
        )


def _write_code_identity(run_dir: str, workspace_root: str) -> Dict[str, Any]:
    """Write ``code_identity.json`` into *run_dir* and return the dict."""
    git_id = get_git_identity(workspace_root)
    with open(os.path.join(run_dir, "code_identity.json"), "w") as fh:
        json.dump(git_id, fh, indent=4)
    return git_id


def _write_registry_entry(
    base_dir: str,
    config: ExperimentConfig,
    git_id: Dict[str, Any],
    timestamp: str,
) -> None:
    """Append a new row to ``registry.csv`` (creating the header if needed)."""
    registry_path = os.path.join(base_dir, "registry.csv")
    file_exists = os.path.isfile(registry_path)

    fields = [
        "run_id", "experiment_id", "config_hash", "git_commit", "seed",
        "backbone", "experts", "top_k", "router_variant", "loss_variant",
        "moe_type", "batch_equivalence",
        "validation_primary_metric", "validation_primary_value", "best_epoch",
        "best_global_step", "status", "timestamp",
    ]

    with open(registry_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "run_id": config.run_id,
            "experiment_id": config.experiment_id,
            "config_hash": config.get_canonical_hash(),
            "git_commit": git_id.get("git_commit", ""),
            "seed": config.train.seed,
            "backbone": config.model.backbone,
            "experts": config.model.num_experts,
            "top_k": config.model.top_k,
            "router_variant": config.model.router_variant,
            "loss_variant": (
                f"BCE={config.loss.bce_weight},"
                f"IoU={config.loss.iou_weight},"
                f"SSIM={config.loss.ssim_weight},"
                f"BND={config.loss.boundary_weight}"
            ),
            "moe_type": config.model.moe_type,
            "batch_equivalence": config.batch_equivalence,
            "validation_primary_metric": config.eval.validation_metric_selection,
            "validation_primary_value": "",
            "best_epoch": "",
            "best_global_step": "",
            "status": "CREATED",
            "timestamp": timestamp,
        })


def _dry_run_report(
    config: ExperimentConfig,
    run_dir: str,
    workspace_root: str,
) -> None:
    """Print a dry-run summary to stdout."""
    log.info(f"\n[DRY RUN] Would create run directory: {run_dir}")
    log.info(f"[DRY RUN] Experiment ID: {config.experiment_id}")
    log.info(f"[DRY RUN] Run ID: {config.run_id}")
    log.info(f"[DRY RUN] Config Hash: {config.get_canonical_hash()}")
    log.info(f"[DRY RUN] Batch Equivalence: {config.batch_equivalence}")
    if config.batch_equivalence == "NON_MATCHED":
        log.info("[DRY RUN] WARNING: NON-IDENTICAL-BATCH experiment!")
    git_id = get_git_identity(workspace_root)
    log.info(f"[DRY RUN] Git/Code Identity: {git_id}")


def setup_experiment_run(
    config: ExperimentConfig,
    base_dir: str = "experiments",
    dry_run: bool = False,
) -> str:
    """Scaffold a new experiment run directory and persist metadata.

    This is the main entry point called by ``src/training/setup.py``.  It:

    1. Generates and assigns ``config.experiment_id`` and ``config.run_id``.
    2. Creates the run directory at
       ``<base_dir>/<experiment_id>/<run_id>/``.
    3. Writes ``config.json``, ``config_hash.txt``, and
       ``code_identity.json`` into the run directory.
    4. Appends a row to ``<base_dir>/registry.csv``.

    Args:
        config: The experiment configuration (mutated with generated IDs).
        base_dir: Root directory for experiment outputs.
        dry_run: If ``True``, print a summary without writing anything.

    Returns:
        The absolute path to the run directory.
    """
    config.experiment_id = generate_experiment_id(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    config.run_id = f"{config.experiment_id}__{timestamp}_{uid}"

    run_dir = os.path.join(base_dir, config.experiment_id, config.run_id)

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if dry_run:
        _dry_run_report(config, run_dir, workspace_root)
        return run_dir

    verify_split_manifest(config)

    os.makedirs(run_dir, exist_ok=False)

    git_id = _write_code_identity(run_dir, workspace_root)
    if git_id.get("git_dirty"):
        log.info("WARNING: WORKING TREE NOT CLEAN")

    config.save(os.path.join(run_dir, "config.json"))
    with open(os.path.join(run_dir, "config_hash.txt"), "w") as fh:
        fh.write(config.get_canonical_hash())

    _write_registry_entry(base_dir, config, git_id, timestamp)

    return run_dir


# ---------------------------------------------------------------------------
# 3. Registry — update rows in the CSV registry
# ---------------------------------------------------------------------------


def update_registry_status(
    base_dir: str,
    run_id: str,
    status: str,
    val_metric_val: str = "",
    best_epoch: str = "",
    best_step: str = "",
) -> None:
    """Update the status (and optional metrics) for *run_id* in the CSV registry.

    Args:
        base_dir: Directory containing ``registry.csv``.
        run_id: The run identifier to update.
        status: New status string (e.g. ``"RUNNING"``, ``"COMPLETED"``,
            ``"FAILED"``).
        val_metric_val: Optional validation metric value.
        best_epoch: Optional best-epoch string.
        best_step: Optional best-global-step string.
    """
    registry_path = os.path.join(base_dir, "registry.csv")
    if not os.path.exists(registry_path):
        return

    temp_path = registry_path + ".tmp"

    with open(registry_path, "r") as fin, open(temp_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)  # type: ignore[union-attr]
        writer.writeheader()

        for row in reader:
            if row["run_id"] == run_id:
                row["status"] = status
                if val_metric_val:
                    row["validation_primary_value"] = val_metric_val
                if best_epoch:
                    row["best_epoch"] = best_epoch
                if best_step:
                    row["best_global_step"] = best_step
            writer.writerow(row)

    os.replace(temp_path, registry_path)
