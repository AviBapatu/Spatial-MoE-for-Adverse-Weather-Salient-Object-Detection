import copy
import csv
import os
from typing import List

from src.config import ExperimentConfig


def load_baseline(path: str = "experiments/baseline_v1.json") -> ExperimentConfig:
    return ExperimentConfig.load(path)

def adjust_for_effective_batch(config: ExperimentConfig, target_effective: int) -> None:
    """Set grad_accum_steps so effective global batch size equals ``target_effective``.

    If ``target_effective`` is not a clean multiple of ``batch_per_gpu * 2``
    (assumes 2-GPU Kaggle T4 setup), ``batch_equivalence`` is set to
    ``"NON_MATCHED"`` and no adjustment is made.
    """
    config.opt.effective_global_batch = target_effective
    world_size_assumed = 2 # Usually 2 GPUs on Kaggle T4
    batch_gpu = config.opt.batch_per_gpu
    if target_effective % (batch_gpu * world_size_assumed) == 0:
        config.opt.grad_accum_steps = target_effective // (batch_gpu * world_size_assumed)
        config.batch_equivalence = "MATCHED"
    else:
        config.batch_equivalence = "NON_MATCHED"

def generate_architecture_matrix(baseline: ExperimentConfig) -> List[ExperimentConfig]:
    """Generate architecture ablations varying backbone, expert count and top_k."""
    matrix = []

    # A: PVT-B2, 8 experts, K=2
    c_a = copy.deepcopy(baseline)
    c_a.model.backbone = "pvt_v2_b2"
    c_a.model.num_experts = 8
    c_a.model.top_k = 2
    c_a.opt.batch_per_gpu = 2
    adjust_for_effective_batch(c_a, baseline.opt.effective_global_batch)
    matrix.append(c_a)

    # B: PVT-B4, 8 experts, K=2 (baseline essentially)
    c_b = copy.deepcopy(baseline)
    c_b.model.backbone = "pvt_v2_b4"
    c_b.model.num_experts = 8
    c_b.model.top_k = 2
    c_b.opt.batch_per_gpu = 1
    adjust_for_effective_batch(c_b, baseline.opt.effective_global_batch)
    matrix.append(c_b)

    # C: PVT-B4, 6 experts, K=2
    c_c = copy.deepcopy(baseline)
    c_c.model.backbone = "pvt_v2_b4"
    c_c.model.num_experts = 6
    c_c.model.top_k = 2
    c_c.opt.batch_per_gpu = 1
    adjust_for_effective_batch(c_c, baseline.opt.effective_global_batch)
    matrix.append(c_c)

    # D: PVT-B4, 8 experts, K=1
    c_d = copy.deepcopy(baseline)
    c_d.model.backbone = "pvt_v2_b4"
    c_d.model.num_experts = 8
    c_d.model.top_k = 1
    c_d.opt.batch_per_gpu = 1
    adjust_for_effective_batch(c_d, baseline.opt.effective_global_batch)
    matrix.append(c_d)

    # E: PVT-B4, 8 experts, K=3
    c_e = copy.deepcopy(baseline)
    c_e.model.backbone = "pvt_v2_b4"
    c_e.model.num_experts = 8
    c_e.model.top_k = 3
    c_e.opt.batch_per_gpu = 1
    adjust_for_effective_batch(c_e, baseline.opt.effective_global_batch)
    matrix.append(c_e)

    return matrix

def generate_moe_ladder(baseline: ExperimentConfig) -> List[ExperimentConfig]:
    """Generate a no-MoE / dense-MoE / sparse-MoE ablation ladder."""
    matrix = []

    # No-MoE (shared block)
    c_no = copy.deepcopy(baseline)
    c_no.model.moe_type = "none"
    adjust_for_effective_batch(c_no, baseline.opt.effective_global_batch)
    matrix.append(c_no)

    # Dense-MoE
    c_dense = copy.deepcopy(baseline)
    c_dense.model.moe_type = "dense"
    adjust_for_effective_batch(c_dense, baseline.opt.effective_global_batch)
    matrix.append(c_dense)

    # Sparse-MoE
    c_sparse = copy.deepcopy(baseline)
    c_sparse.model.moe_type = "sparse"
    adjust_for_effective_batch(c_sparse, baseline.opt.effective_global_batch)
    matrix.append(c_sparse)

    return matrix

def generate_loss_matrix(baseline: ExperimentConfig) -> List[ExperimentConfig]:
    """Generate loss ablations toggling SSIM and boundary term weights."""
    matrix = []

    # L1: BCE + IoU
    c_l1 = copy.deepcopy(baseline)
    c_l1.loss.ssim_weight = 0.0
    c_l1.loss.boundary_weight = 0.0
    adjust_for_effective_batch(c_l1, baseline.opt.effective_global_batch)
    matrix.append(c_l1)

    # L2: BCE + IoU + SSIM
    c_l2 = copy.deepcopy(baseline)
    c_l2.loss.ssim_weight = 1.0
    c_l2.loss.boundary_weight = 0.0
    adjust_for_effective_batch(c_l2, baseline.opt.effective_global_batch)
    matrix.append(c_l2)

    # L3: BCE + IoU + SSIM + Boundary
    c_l3 = copy.deepcopy(baseline)
    c_l3.loss.ssim_weight = 1.0
    c_l3.loss.boundary_weight = 1.0
    adjust_for_effective_batch(c_l3, baseline.opt.effective_global_batch)
    matrix.append(c_l3)

    return matrix

def generate_router_matrix(baseline: ExperimentConfig) -> List[ExperimentConfig]:
    """Generate router-variant ablations (token-only, token-local, global)."""
    matrix = []

    # R1: Token-only
    c_r1 = copy.deepcopy(baseline)
    c_r1.model.router_variant = "token_only"
    adjust_for_effective_batch(c_r1, baseline.opt.effective_global_batch)
    matrix.append(c_r1)

    # R2: Token + local DWConv context
    c_r2 = copy.deepcopy(baseline)
    c_r2.model.router_variant = "token_local"
    adjust_for_effective_batch(c_r2, baseline.opt.effective_global_batch)
    matrix.append(c_r2)

    # R3: Global/image-level router
    c_r3 = copy.deepcopy(baseline)
    c_r3.model.router_variant = "global"
    adjust_for_effective_batch(c_r3, baseline.opt.effective_global_batch)
    matrix.append(c_r3)

    return matrix

def export_ablation_tables(
    registry_path: str = "experiments/registry.csv", out_dir: str = "experiments"
) -> None:
    """Flatten completed, batch-matched registry runs into ``ablation_results.csv``.

    Args:
        registry_path: path to the experiment registry CSV (``experiments/registry.csv``).
        out_dir: directory to write ``ablation_results.csv`` into; no-op when the
            registry file does not exist.
    """
    if not os.path.exists(registry_path):
        return

    with open(registry_path, "r") as f:
        reader = csv.DictReader(f)
        runs = [row for row in reader if row["status"] == "COMPLETED" and row["batch_equivalence"] == "MATCHED"]

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "ablation_results.csv")
    if runs:
        with open(out_file, "w", newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=list(runs[0].keys()))
            writer.writeheader()
            for r in runs:
                writer.writerow(r)
