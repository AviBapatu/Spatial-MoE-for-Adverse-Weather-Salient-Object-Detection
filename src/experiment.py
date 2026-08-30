import os
import csv
import json
import uuid
import subprocess
from datetime import datetime
from src.config import ExperimentConfig

def get_git_identity(workspace_root: str) -> dict:
    from typing import Dict, Any
    identity: Dict[str, Any] = {
        "git_commit": "UNKNOWN",
        "git_branch": "UNKNOWN",
        "git_dirty": False,
        "warning": None
    }
    
    if os.path.exists(os.path.join(workspace_root, ".git")):
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace_root).decode("utf-8").strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=workspace_root).decode("utf-8").strip()
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=workspace_root).decode("utf-8").strip()
            
            identity["git_commit"] = commit
            identity["git_branch"] = branch
            identity["git_dirty"] = bool(status)
            
            if identity["git_dirty"]:
                identity["warning"] = "WORKING TREE NOT CLEAN"
                
        except Exception:
            pass
    else:
        # Static hash fallback
        try:
            import hashlib
            src_dir = os.path.join(workspace_root, "src")
            hasher = hashlib.sha256()
            for root, _, files in sorted(os.walk(src_dir)):
                for f in sorted(files):
                    if f.endswith('.py'):
                        with open(os.path.join(root, f), 'rb') as fp:
                            hasher.update(fp.read())
            identity["static_src_hash"] = hasher.hexdigest()
        except Exception:
            pass
            
    return identity

def generate_experiment_id(config: ExperimentConfig) -> str:
    # E.g., EXP_B4_E8_K2_S32_R1_L1_M_SPARSE
    bb = config.model.backbone.upper().replace('PVT_V2_', '')
    e = config.model.num_experts
    k = config.model.top_k
    s = config.opt.effective_global_batch
    
    # Simple abbreviations for variants
    r_abbrev = "R_DEF"
    if config.model.router_variant == "token_only": r_abbrev = "R1"
    elif config.model.router_variant == "token_local": r_abbrev = "R2"
    elif config.model.router_variant == "global": r_abbrev = "R3"
    
    moe_abbrev = "M_SPARSE"
    if config.model.moe_type == "none": moe_abbrev = "M_NONE"
    elif config.model.moe_type == "dense": moe_abbrev = "M_DENSE"
    
    # Loss variants approx
    l_abbrev = "L1"
    if config.loss.ssim_weight > 0: l_abbrev = "L2"
    if config.loss.boundary_weight > 0: l_abbrev = "L3"
    
    return f"EXP_{bb}_E{e}_K{k}_S{s}_{r_abbrev}_{l_abbrev}_{moe_abbrev}"

def verify_split_manifest(config: ExperimentConfig):
    if not config.data.split_manifest_path:
        return
        
    if not os.path.exists(config.data.split_manifest_path):
        raise FileNotFoundError(f"Split manifest not found at {config.data.split_manifest_path}")
        
    import hashlib
    with open(config.data.split_manifest_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
        
    if config.data.split_manifest_hash and file_hash != config.data.split_manifest_hash:
        raise ValueError(f"Split manifest hash mismatch! Expected {config.data.split_manifest_hash}, got {file_hash}")

def setup_experiment_run(config: ExperimentConfig, base_dir: str = "experiments", dry_run: bool = False):
    config.experiment_id = generate_experiment_id(config)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    config.run_id = f"{config.experiment_id}__{timestamp}_{uid}"
    
    run_dir = os.path.join(base_dir, config.experiment_id, config.run_id)
    
    if dry_run:
        print(f"\n[DRY RUN] Would create run directory: {run_dir}")
        print(f"[DRY RUN] Experiment ID: {config.experiment_id}")
        print(f"[DRY RUN] Run ID: {config.run_id}")
        print(f"[DRY RUN] Config Hash: {config.get_canonical_hash()}")
        print(f"[DRY RUN] Batch Equivalence: {config.batch_equivalence}")
        if config.batch_equivalence == "NON_MATCHED":
            print("[DRY RUN] WARNING: NON-IDENTICAL-BATCH experiment!")
        
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        git_id = get_git_identity(workspace_root)
        print(f"[DRY RUN] Git/Code Identity: {git_id}")
        return run_dir
        
    verify_split_manifest(config)
        
    os.makedirs(run_dir, exist_ok=False)
    
    # Save code identity
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_id = get_git_identity(workspace_root)
    with open(os.path.join(run_dir, "code_identity.json"), "w") as f:
        json.dump(git_id, f, indent=4)
        
    if git_id.get("git_dirty"):
        print(f"WARNING: WORKING TREE NOT CLEAN")
        
    # Save config
    config.save(os.path.join(run_dir, "config.json"))
    with open(os.path.join(run_dir, "config_hash.txt"), "w") as f:
        f.write(config.get_canonical_hash())
        
    # Init registry
    registry_path = os.path.join(base_dir, "registry.csv")
    file_exists = os.path.isfile(registry_path)
    
    fields = [
        "run_id", "experiment_id", "config_hash", "git_commit", "seed", 
        "backbone", "experts", "top_k", "router_variant", "loss_variant", "moe_type", "batch_equivalence",
        "validation_primary_metric", "validation_primary_value", "best_epoch", 
        "best_global_step", "status", "timestamp"
    ]
    
    with open(registry_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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
            "loss_variant": f"BCE={config.loss.bce_weight},IoU={config.loss.iou_weight},SSIM={config.loss.ssim_weight},BND={config.loss.boundary_weight}",
            "moe_type": config.model.moe_type,
            "batch_equivalence": config.batch_equivalence,
            "validation_primary_metric": config.eval.validation_metric_selection,
            "validation_primary_value": "",
            "best_epoch": "",
            "best_global_step": "",
            "status": "CREATED",
            "timestamp": timestamp
        })
        
    return run_dir

def update_registry_status(base_dir: str, run_id: str, status: str, val_metric_val: str = "", best_epoch: str = "", best_step: str = ""):
    registry_path = os.path.join(base_dir, "registry.csv")
    if not os.path.exists(registry_path):
        return
        
    temp_path = registry_path + ".tmp"
    
    with open(registry_path, "r") as fin, open(temp_path, "w", newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames) # type: ignore
        writer.writeheader()
        
        for row in reader:
            if row["run_id"] == run_id:
                row["status"] = status
                if val_metric_val: row["validation_primary_value"] = val_metric_val
                if best_epoch: row["best_epoch"] = best_epoch
                if best_step: row["best_global_step"] = best_step
            writer.writerow(row)
            
    os.replace(temp_path, registry_path)
