"""account A — 2-expert, top_k=1 run.

Regenerates ``moe-of-sod__2expert_accountA.ipynb`` and is the editable source of truth for it — change a
cell here and re-run this script to rebuild the notebook. The generate_notebook_accountB.py
generator holds the same cells for the other account (the two differ only in
ACTIVE_CONFIG_PATH), so a shared cell change has to be applied to
both files.

Usage:
    python generate_notebook_accountA.py
"""
import json

OUTPUT_NOTEBOOK = "moe-of-sod__2expert_accountA.ipynb"


def create_notebook() -> None:
    """Build the notebook cell list and write it to ``OUTPUT_NOTEBOOK``."""
    cells = []

    def add_markdown(text: str) -> None:
        """Append a markdown cell holding *text*."""
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        })

    def add_code(text: str) -> None:
        """Append a code cell holding *text*."""
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        })

    add_markdown(
        r"""## 00_hf_sync
""")

    add_code(
        r"""# uncomment and run this if you want to manually push your latest checkpoints to Hugging Face
# import os
# import sys
# import subprocess
# from kaggle_secrets import UserSecretsClient
# from huggingface_hub import hf_hub_download
#
# # 1. Set credentials safely (bypassing Cell 01)
# os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
# os.environ["HF_REPO_ID"] = "Avi2006/spatial-moe-results"
# PROJECT_ROOT = "/kaggle/working/spatial_moe_sod"
#
# # 2. Download ONLY the latest code (bypasses Checkpoint downloads)
# print("Downloading latest code...")
# zip_path = hf_hub_download(
#     repo_id=os.environ["HF_REPO_ID"], repo_type="dataset",
#     filename="code/spatial_moe_sod_code.zip", token=os.environ["HF_TOKEN"],
# )
# subprocess.run(["unzip", "-q", "-o", zip_path, "-d", PROJECT_ROOT], check=True)
# print("Code updated!")
#
# # 3. Now run the push script
# sys.path.insert(0, PROJECT_ROOT)
# from src.config import ExperimentConfig
# from src.experiment import generate_experiment_id
# from src.hf_sync import push_checkpoint
#
# # Namespace both the local directory and the remote filename by this run's
# # experiment ID, so pushing from one account can never overwrite the other
# # account's checkpoints.
# ACTIVE_CONFIG_PATH = "experiments/v_2expert_gatedense.json"
# experiment_id = generate_experiment_id(
#     ExperimentConfig.load(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH))
# )
# ckpt_dir = os.path.join("/kaggle/working/WXSOD_Checkpoints", experiment_id)
# for name in ["best.pth", "latest.pth", "training_complete.json", "latest_prev.pth"]:
#     path = os.path.join(ckpt_dir, name)
#     if os.path.exists(path):
#         print(f"Uploading {name}...")
#         push_checkpoint(path, name=f"{experiment_id}_{name}")
# print("Done!")

""")

    add_markdown(
        r"""## 01_config
""")

    add_code(
        r"""import os
import json
import shutil
import hashlib

# Configuration & Modes
# RUN_MODE strictly governs the allowed execution path.
# Allowed: "VALIDATE", "TRAIN", "RESUME", "EVALUATE"
RUN_MODE = "EVALUATE"
ACTIVE_CONFIG_PATH = "experiments/v_2expert_gatedense.json"
# Analysis results are uploaded under the experiment ID derived from
# ACTIVE_CONFIG_PATH (see the upload cell), so there is no separate label here
# that could drift out of sync with the checkpoints.

DATA_SOURCE = "GOOGLE_DRIVE"
DATA_FILE_ID = "1SSELvRYI-cwd9mzA8dWLbv4o1IffjkoW"

# --- Hugging Face Hub is now the single source of truth for code + checkpoints ---
# HF_TOKEN is read from a Kaggle Secret (Add-ons > Secrets > add "HF_TOKEN"),
# never hardcoded. HF_REPO_ID is the one repo holding both code/ and checkpoints/.
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
HF_REPO_ID = "Avi2006/spatial-moe-results"
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_REPO_ID"] = HF_REPO_ID

PROJECT_ROOT = "/kaggle/working/spatial_moe_sod"
CHECKPOINT_ROOT = "/kaggle/working/WXSOD_Checkpoints"
PREFLIGHT_ROOT = "/kaggle/working/WXSOD_Preflight"
RESUME_TEST_ROOT = "/kaggle/working/WXSOD_ResumeTest"

NUM_GPUS = 2
FINAL_EPOCHS = 50

# These get exported so train_ddp.py picks them up
os.environ["CHECKPOINT_ROOT"] = CHECKPOINT_ROOT
os.environ["PREFLIGHT_ROOT"] = PREFLIGHT_ROOT

# Global state for dynamic final audit gate
GATES = {
    "ENV_CHECK": "NOT_RUN",
    "GPU_CHECK": "NOT_RUN",
    "DATA_CHECK": "NOT_RUN",
    "PROJECT_CHECK": "NOT_RUN",
    "DEPENDENCY_CHECK": "NOT_RUN",
    "PVT_CHECK": "NOT_RUN",
    "STATIC_CHECK": "NOT_RUN",
    "TRANSFORM_CHECK": "NOT_RUN",
    "MODEL_CHECK": "NOT_RUN",
    "MOE_CHECK": "NOT_RUN",
    "LOSS_CHECK": "NOT_RUN",
    "OPT_CHECK": "NOT_RUN",
    "DDP_CHECK": "NOT_RUN",
    "MEMORY_CHECK": "NOT_RUN",
    "CHECKPOINT_CHECK": "NOT_RUN",
    "RESUME_CHECK": "NOT_RUN",
    "PREFLIGHT_DRY_RUN_CHECK": "NOT_RUN",
    "TORCHRUN_1GPU_CHECK": "NOT_RUN",
    "RESUME_PREFLIGHT_CHECK": "NOT_RUN"
}

def mark_gate(gate, status, msg="", evidence="", run_id=None, config_hash=None):
    if GATES.get(gate) == "FAIL" and status == "PASS":
        print(f"[{gate}] FAIL -> PASS transition authorized (run_id: {run_id}, config_hash: {config_hash})")
    print(f"[{gate}] -> {status} {msg}")
    GATES[gate] = status

# --- Auto-fetch checkpoint from Hugging Face Hub (replaces the old Google Drive flow) ---
def _sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def fetch_checkpoint_from_hf(repo_id, token, dest_root):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    os.makedirs(dest_root, exist_ok=True)

    try:
        manifest_path = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename="checkpoints/checkpoint_manifest.json", token=token,
        )
    except EntryNotFoundError:
        mark_gate("CHECKPOINT_CHECK", "NOT_RUN", msg="No checkpoint manifest on HF yet — training from scratch.")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)
    files_meta = manifest.get("files", {})

    fetched_any = False
    for name in ("latest.pth", "best.pth"):
        meta = files_meta.get(name)
        if meta is None:
            continue
        local_path = os.path.join(dest_root, name)
        if os.path.exists(local_path) and _sha256_file(local_path) == meta.get("sha256"):
            print(f"{name}: local copy already matches HF (sha256 match), skipping download.")
            fetched_any = True
            continue
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id, repo_type="dataset",
                filename=f"checkpoints/{name}", token=token,
            )
        except EntryNotFoundError:
            continue
        shutil.copy2(downloaded, local_path)
        actual = _sha256_file(local_path)
        if meta.get("sha256") and actual != meta["sha256"]:
            mark_gate("CHECKPOINT_CHECK", "FAIL", msg=f"{name}: sha256 mismatch after download")
            raise RuntimeError(f"{name}: downloaded sha256 {actual} != manifest sha256 {meta['sha256']}")
        print(f"{name}: pulled from HF -> {local_path}")
        fetched_any = True

    if fetched_any:
        shutil.copy2(manifest_path, os.path.join(dest_root, "checkpoint_manifest.json"))
        mark_gate("CHECKPOINT_CHECK", "PASS", msg=f"Fetched checkpoint(s) from {repo_id} -> {dest_root}")
    else:
        mark_gate("CHECKPOINT_CHECK", "NOT_RUN", msg="Manifest present but no checkpoint files found.")
    return fetched_any


CHECKPOINT_AVAILABLE = fetch_checkpoint_from_hf(HF_REPO_ID, HF_TOKEN, CHECKPOINT_ROOT)

if CHECKPOINT_AVAILABLE:
    if RUN_MODE == "TRAIN":
        print("⚠️  WARNING: Checkpoint(s) found on HF but RUN_MODE is TRAIN — NOT switching to RESUME.")
        print("   Old checkpoints have been downloaded locally but will NOT be loaded.")
        print(f"   Training will start from scratch with {ACTIVE_CONFIG_PATH}.")
        print("   If you want to resume instead, manually set RUN_MODE = 'RESUME' at the top of this cell.")
else:
    print("No checkpoint found on HF — training from scratch.")

""")

    add_markdown(
        r"""## 02_environment
""")

    add_code(
        r"""import torch
import os
import shutil

print("1. Environment & GPU Info")
print("PyTorch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
num_gpus = torch.cuda.device_count()
print("GPU Count:", num_gpus)
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Disk space check
total, used, free = shutil.disk_usage("/kaggle/working")
free_gb = free / (1024**3)
print(f"Available Disk Space: {free_gb:.2f} GB")
if free_gb < 5.0:
    mark_gate("ENV_CHECK", "FAIL", "Insufficient disk space.")
    raise RuntimeError("At least 5 GB required in /kaggle/working.")

if num_gpus < NUM_GPUS and RUN_MODE in ["VALIDATE", "TRAIN", "RESUME"]:
    mark_gate("GPU_CHECK", "FAIL", f"Expected {NUM_GPUS} GPUs")
    raise RuntimeError(f"Expected {NUM_GPUS} GPUs")
else:
    mark_gate("GPU_CHECK", "PASS")
    mark_gate("ENV_CHECK", "PASS")

""")

    add_markdown(
        r"""## 03_dataset_acquire
""")

    add_code(
        r"""import subprocess

def is_dataset_valid(path):
    def check_root(r):
        reqs = ["train_sys/input", "train_sys/gt", "test_sys/input", "test_sys/gt", "test_real/input", "test_real/gt"]
        return all(os.path.isdir(os.path.join(r, req)) for req in reqs)
        
    if check_root(path):
        return path
    for root, dirs, _ in os.walk(path):
        if check_root(root):
            return root
    return None

EXTRACT_PATH = "/kaggle/working/WXSDO_data"

valid_root = is_dataset_valid(EXTRACT_PATH)
if valid_root is None:
    print("Dataset not found locally, downloading...")
    ZIP_PATH = "/kaggle/working/dataset.zip"
    if not os.path.exists(ZIP_PATH):
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
        subprocess.run(["gdown", DATA_FILE_ID, "-O", ZIP_PATH], check=True)
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", ZIP_PATH, "-d", EXTRACT_PATH], check=True)
    valid_root = is_dataset_valid(EXTRACT_PATH)

if valid_root is None:
    mark_gate("DATA_CHECK", "FAIL")
    raise RuntimeError("Dataset extraction completed but valid WXSOD root was not found.")
else:
    print("Dataset successfully extracted and validated at:", valid_root)

""")

    add_markdown(
        r"""## 04_dataset_validate
""")

    add_code(
        r"""import glob

def check_stems(split_path):
    input_files = glob.glob(os.path.join(split_path, "input", "*.*"))
    gt_files = glob.glob(os.path.join(split_path, "gt", "*.*"))
    
    valid_exts = {".png", ".jpg", ".jpeg"}
    input_files = [f for f in input_files if os.path.splitext(f)[1].lower() in valid_exts]
    gt_files = [f for f in gt_files if os.path.splitext(f)[1].lower() in valid_exts]
    
    input_stems = {os.path.splitext(os.path.basename(f))[0] for f in input_files}
    gt_stems = {os.path.splitext(os.path.basename(f))[0] for f in gt_files}
    
    missing_gt = input_stems - gt_stems
    missing_input = gt_stems - input_stems
    
    if missing_gt or missing_input:
        return False, len(missing_gt), len(missing_input)
    return True, len(input_stems), len(gt_stems)

splits = {"train_sys": 12891, "test_sys": 1500, "test_real": 554}
all_valid = True
for split, req_count in splits.items():
    split_path = os.path.join(valid_root, split)
    if not os.path.exists(split_path):
        print(f"{split} missing!")
        all_valid = False
        continue
    valid, m_gt, m_in = check_stems(split_path)
    if not valid:
        print(f"{split} stem mismatch: {m_gt} missing GT, {m_in} missing Inputs")
        all_valid = False
    else:
        print(f"{split}: {m_in} pairs verified perfectly.")
        if RUN_MODE in ["VALIDATE", "TRAIN", "EVALUATE"] and m_in != req_count:
            print(f"{split} count mismatch. Expected {req_count}, got {m_in}.")
            all_valid = False

if all_valid:
    mark_gate("DATA_CHECK", "PASS")
else:
    mark_gate("DATA_CHECK", "FAIL")
    raise RuntimeError("Dataset integrity check failed.")

""")

    add_markdown(
        r"""## 05_project_deploy
""")

    add_code(
        r"""import sys
import json
import hashlib
import shutil
import os
import subprocess
from huggingface_hub import hf_hub_download

os.makedirs(PROJECT_ROOT, exist_ok=True)
print(f"Downloading code from {os.environ['HF_REPO_ID']}...")

manifest_path = hf_hub_download(
    repo_id=os.environ["HF_REPO_ID"], repo_type="dataset",
    filename="code/project_manifest.json", token=os.environ["HF_TOKEN"],
)
zip_path = hf_hub_download(
    repo_id=os.environ["HF_REPO_ID"], repo_type="dataset",
    filename="code/spatial_moe_sod_code.zip", token=os.environ["HF_TOKEN"],
)

with open(manifest_path, "r") as f:
    manifest = json.load(f)

# Verify ZIP
sha256 = hashlib.sha256()
with open(zip_path, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
        sha256.update(chunk)
        
if sha256.hexdigest() != manifest["archive_sha256"]:
    mark_gate("PROJECT_CHECK", "FAIL", "ZIP SHA256 mismatch!")
    raise RuntimeError("Project source archive corrupted or outdated.")

subprocess.run(["unzip", "-q", "-o", zip_path, "-d", PROJECT_ROOT], check=True)

# Post-unzip existence assertions
required_files = ["src/train_ddp.py", "src/smoke_test.py", "src/model.py", "src/optimization.py", "src/evaluate.py"]
for f in required_files:
    if not os.path.exists(os.path.join(PROJECT_ROOT, f)):
        mark_gate("PROJECT_CHECK", "FAIL")
        raise RuntimeError(f"Payload corrupted: missing {f}")

print("Project source code deployed to:", PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
mark_gate("PROJECT_CHECK", "PASS")

""")

    add_markdown(
        r"""## 06_dependencies
""")

    add_code(
        r"""import importlib
import subprocess
deps = {
    "torch": "torch", 
    "torchvision": "torchvision", 
    "timm": "timm", 
    "albumentations": "albumentations", 
    "opencv-python-headless": "cv2", 
    "numpy": "numpy", 
    "pysodmetrics": "py_sod_metrics"
}
env_data = {}
all_passed = True
for pip_name, mod_name in deps.items():
    try:
        mod = importlib.import_module(mod_name)
        env_data[pip_name] = getattr(mod, "__version__", "unknown")
    except ImportError:
        print(f"Missing dependency: {pip_name}. Installing...")
        subprocess.run(["pip", "install", "-q", pip_name], check=True)
        try:
            mod = importlib.import_module(mod_name)
            env_data[pip_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            all_passed = False

if all_passed:
    with open(os.path.join(PROJECT_ROOT, "environment.json"), "w") as f:
        json.dump(env_data, f)
    mark_gate("DEPENDENCY_CHECK", "PASS")
else:
    mark_gate("DEPENDENCY_CHECK", "FAIL")
    raise RuntimeError("Failed to resolve dependencies.")

""")

    add_markdown(
        r"""## 07_pretrained_b4
""")

    add_code(
        r"""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    from src.backbone import MultiScaleBackbone
    try:
        model1 = MultiScaleBackbone(model_name='pvt_v2_b4', pretrained=True, d=256)
        model2 = MultiScaleBackbone(model_name='pvt_v2_b4', pretrained=True, d=256)
        
        dummy_input = torch.randn(1, 3, 384, 384)
        feats = model1(dummy_input)
        
        assert torch.isfinite(feats['res_4']).all()
        assert torch.isfinite(feats['res_8']).all()
        assert torch.isfinite(feats['res_16']).all()
        assert feats['res_4'].shape == (1, 256, 96, 96), "Y4 shape incorrect"
        assert feats['res_8'].shape == (1, 256, 48, 48), "Y8 shape incorrect"
        assert feats['res_16'].shape == (1, 256, 24, 24), "Y16 shape incorrect"
        
        del model1, model2
        mark_gate("PVT_CHECK", "PASS")
    except Exception as e:
        mark_gate("PVT_CHECK", "FAIL")
        raise RuntimeError(f"PVT Instantiation failed: {e}")

""")

    add_markdown(
        r"""## 08_static_checks
""")

    add_code(
        r"""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    try:
        import src.model
        import src.train_ddp
        import src.loss
        mark_gate("STATIC_CHECK", "PASS")
    except ImportError as e:
        mark_gate("STATIC_CHECK", "FAIL")
        raise e

""")

    add_markdown(
        r"""## 09_data_transform
""")

    add_code(
        r"""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    from src.dataset import WXSODDataset
    try:
        ds = WXSODDataset(valid_root, split="train")
        
        # Test synthetic geometric mask alignment
        import numpy as np
        
        test_shapes = [(800, 200), (200, 800), (400, 400), (400, 287), (287, 400), (320, 350)]
        for h, w in test_shapes:
            syn_img = np.zeros((h, w, 3), dtype=np.uint8)
            syn_mask = np.zeros((h, w), dtype=np.uint8)
            if w == 800: syn_mask[:, :400] = 255
            
            # Use the dataset's internal methods to mimic __getitem__
            img_pad, mask_pad, _, _ = ds._aspect_preserving_resize_pad(syn_img, syn_mask)
            edge_pad = ds._compute_edge_map(mask_pad)
            
            assert img_pad.shape == (384, 384, 3)
            assert mask_pad.shape == (384, 384)
            assert edge_pad.shape == (384, 384)
            
            if w == 800 and h == 200:
                assert (mask_pad[:, 192:] == 0).all(), "Mask alignment failed during resize/pad"
        
        for idx in [0, len(ds)//2, len(ds)-1]:
            samp = ds[idx]
            assert samp['image'].shape == (3, 384, 384)
            assert samp['mask'].shape == (1, 384, 384)
            
        # Check split identity for leakage guard
        assert ds.split == "train", "Leakage guard: Expected train split identity."
        mark_gate("TRANSFORM_CHECK", "PASS")
    except Exception as e:
        mark_gate("TRANSFORM_CHECK", "FAIL")
        raise e

""")

    add_markdown(
        r"""## 10_model_shapes
""")

    add_code(
        r"""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    from src.model import SpatialMoESODNet
    try:
        model = SpatialMoESODNet(dim=256)
        dummy_input = torch.randn(1, 3, 384, 384)
        out, moe_outputs = model(dummy_input)
        
        assert torch.isfinite(out.saliency_logits).all(), "saliency_logits not finite"
        assert torch.isfinite(out.boundary_logits).all(), "boundary_logits not finite"
        for m_out in moe_outputs:
            assert torch.isfinite(m_out.features).all(), "routed features not finite"
            assert torch.isfinite(m_out.routing_probs).all(), "router probabilities not finite"
            assert torch.isfinite(m_out.entropy).all(), "entropy not finite"
            
        # The true model architecture expects Y4, Y8, Y16 inside backbone forward
        feats = model.backbone(dummy_input)
        assert feats['res_4'].shape == (1, 256, 96, 96), "Y4 shape incorrect"
        assert feats['res_8'].shape == (1, 256, 48, 48), "Y8 shape incorrect"
        assert feats['res_16'].shape == (1, 256, 24, 24), "Y16 shape incorrect"
        mark_gate("MODEL_CHECK", "PASS")
    except Exception as e:
        mark_gate("MODEL_CHECK", "FAIL")
        raise e

""")

    add_markdown(
        r"""## 13_empty_batch_check
""")

    add_code(
        r"""# if RUN_MODE in ["VALIDATE", "TRAIN"]:
#     print("Running Empty Batch Check...")
#     with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
#         canonical_cfg = json.load(f)
#     from src.config import ExperimentConfig
#     canonical_cfg = ExperimentConfig.from_dict(canonical_cfg).to_dict()
#     from src.train_ddp import get_config_hash
#     CURRENT_CONFIG_HASH = get_config_hash(canonical_cfg, "model_config_hash")
#     CURRENT_RUN_ID = canonical_cfg.get("run_id") or "test_run_id"

#     res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "empty_batch", "--data_root", valid_root, "--result_file", "test_empty_batch.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
#     with open(os.path.join(PROJECT_ROOT, "test_empty_batch.json"), "r") as f:
#         data = json.load(f)
#         if data.get("status") == "PASS":
#             mark_gate("EMPTY_BATCH_CHECK", "PASS")
#         else:
#             mark_gate("EMPTY_BATCH_CHECK", "FAIL")
#             raise RuntimeError(f"Empty Batch Check Failed: {data}")

""")

    add_markdown(
        r"""## 14_smoke_2gpu
""")

    add_code(
        r"""# if RUN_MODE in ["VALIDATE", "TRAIN"]:
#     with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
#         canonical_cfg = json.load(f)
#     from src.config import ExperimentConfig
#     canonical_cfg = ExperimentConfig.from_dict(canonical_cfg).to_dict()
#     from src.train_ddp import get_config_hash
#     CURRENT_CONFIG_HASH = get_config_hash(canonical_cfg, "model_config_hash")
#     CURRENT_RUN_ID = canonical_cfg.get("run_id") or "test_run_id"

#     print("Running 2-GPU DDP Smoke Test...")
#     res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "ddp", "--data_root", valid_root, "--result_file", "test_2gpu.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
#     with open(os.path.join(PROJECT_ROOT, "test_2gpu.json"), "r") as f:
#         data = json.load(f)
#         if data.get("status") == "PASS" and data.get("world_size") == 2 and set(data.get("ranks_completed", [])) == {0, 1}:
#             mark_gate("DDP_CHECK", "PASS")
#             mark_gate("MOE_CHECK", data.get("moe_check", "FAIL"), "Sparsity/Token Identity")
#             mark_gate("LOSS_CHECK", data.get("loss_check", "FAIL"), "Loss Gradients/Behavior")
#             mark_gate("OPT_CHECK", data.get("optimizer_check", "FAIL"), "Optimizer Groups/Behavior")
#         else:
#             mark_gate("DDP_CHECK", "FAIL")
#             raise RuntimeError(f"DDP Smoke Failed: {data}")

""")

    add_markdown(
        r"""## 15_production_calibration
""")

    add_code(
        r"""# if RUN_MODE in ["VALIDATE", "TRAIN"]:
#     print("Running 2-GPU Production Memory Calibration...")
#     res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "memory", "--data_root", valid_root, "--result_file", "test_mem.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
#     with open(os.path.join(PROJECT_ROOT, "test_mem.json"), "r") as f:
#         data = json.load(f)
#         if data.get("status") == "PASS":
#             mark_gate("MEMORY_CHECK", "PASS", f"Peak: {data.get('peak_allocated_gb')} GB")
#         else:
#             mark_gate("MEMORY_CHECK", "FAIL")
#             raise RuntimeError(data.get("reason"))

""")

    add_markdown(
        r"""## 16_checkpoint_test
""")

    add_code(
        r"""# if RUN_MODE in ["VALIDATE", "TRAIN"]:
#     print("Running Checkpoint Write Test...")
#     res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "resume_a", "--data_root", valid_root, "--result_file", "test_ckpt.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
#     with open(os.path.join(PROJECT_ROOT, "test_ckpt.json"), "r") as f:
#         if json.load(f).get("status") == "PASS":
#             mark_gate("CHECKPOINT_CHECK", "PASS")
#         else:
#             mark_gate("CHECKPOINT_CHECK", "FAIL")

""")

    add_markdown(
        r"""## 17_resume_test
""")

    add_code(
        r"""# if RUN_MODE in ["VALIDATE", "TRAIN"]:
#     print("Running Checkpoint Resume Test...")
#     res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "resume_b", "--data_root", valid_root, "--result_file", "test_res.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
#     with open(os.path.join(PROJECT_ROOT, "test_res.json"), "r") as f:
#         if json.load(f).get("status") == "PASS":
#             mark_gate("RESUME_CHECK", "PASS")
#         else:
#             mark_gate("RESUME_CHECK", "FAIL")

""")

    add_markdown(
        r"""## 18_final_gate
""")

    add_code(
        r"""# if RUN_MODE == "TRAIN":
#     print("Running Preflight Dry Run (2-5 steps)...")
#     with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
#         runtime_cfg = json.load(f)
        
#     runtime_cfg["data"]["dataset_root"] = valid_root
    
#     RUNTIME_CONFIG = os.path.join(PROJECT_ROOT, "experiments", "kaggle_runtime.json")
    
#     with open(RUNTIME_CONFIG, "w") as f:
#         json.dump(runtime_cfg, f, indent=4)
        
#     print("Runtime dataset root:")
#     print(runtime_cfg["data"]["dataset_root"])
#     print("Runtime config:")
#     print(RUNTIME_CONFIG)
    
#     res = subprocess.run([
#         "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp",
#         "--config", RUNTIME_CONFIG,
#         "--preflight", "--max_optimizer_steps", "5"
#     ], cwd=PROJECT_ROOT, check=True)
    
# _pf_exp_id = runtime_cfg.get("experiment_id", "")
# with open(os.path.join(PREFLIGHT_ROOT, _pf_exp_id, "preflight_results.json"), "r") as f:
#         pf_data = json.load(f)
#     with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
#         canonical_cfg = json.load(f)
#     from src.config import ExperimentConfig
#     canonical_cfg = ExperimentConfig.from_dict(canonical_cfg).to_dict()
#     from src.train_ddp import get_config_hash
#     canonical_hash = get_config_hash(canonical_cfg, "model_config_hash")
    
#     if pf_data.get("status") == "PASS":
#         mark_gate("PREFLIGHT_DRY_RUN_CHECK", "PASS", run_id=pf_data.get("run_id"), config_hash=pf_data.get("config_hash"))
#     else:
#         mark_gate("PREFLIGHT_DRY_RUN_CHECK", "FAIL")
#         raise RuntimeError(f"Preflight validation failed: {pf_data}")

# FINAL_STATUS = "PASS"
# # Some gates are only for train/validate
# if RUN_MODE in ["VALIDATE", "TRAIN"]:
#     with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
#         canonical_cfg = json.load(f)
#     from src.config import ExperimentConfig
#     canonical_cfg = ExperimentConfig.from_dict(canonical_cfg).to_dict()
#     from src.train_ddp import get_config_hash
#     CURRENT_CONFIG_HASH = get_config_hash(canonical_cfg, "model_config_hash")
#     CURRENT_RUN_ID = canonical_cfg.get("run_id") or "test_run_id"

#     # 9. Current-run evidence validation
#     import time
#     required_files = [
#         "test_empty_batch.json", "test_2gpu.json", "test_mem.json", "test_ckpt.json", "test_res.json"
#     ]
#     for filename in required_files:
#         fpath = os.path.join(PROJECT_ROOT, filename)
#         if not os.path.exists(fpath):
#             print(f"Missing required test result: {filename}")
#             FINAL_STATUS = "FAIL"
#             continue
#         with open(fpath, "r") as f:
#             data = json.load(f)
#             if data.get("status") != "PASS":
#                 print(f"Test result not PASS: {filename}")
#                 FINAL_STATUS = "FAIL"
#             if data.get("run_id") != CURRENT_RUN_ID or data.get("config_hash") != CURRENT_CONFIG_HASH:
#                 print(f"Evidence mismatch (run_id/config_hash) in {filename}")
#                 FINAL_STATUS = "FAIL"
        
#         # Freshness check: file must be modified recently (within the last hour)
#         mtime = os.path.getmtime(fpath)
#         if time.time() - mtime > 3600:
#             print(f"Stale test result (timestamp): {filename}")
#             FINAL_STATUS = "FAIL"
            
#     if RUN_MODE == "TRAIN":
# _pf_exp_id = runtime_cfg.get("experiment_id", "")
# with open(os.path.join(PREFLIGHT_ROOT, _pf_exp_id, "preflight_results.json"), "r") as f:
#         if os.path.exists(fpath):
#             with open(fpath, "r") as f:
#                 pf_data = json.load(f)
#                 # Note: run_id is always a fresh timestamped value (e.g. EXP_...timestamp...),
#                 # so we only validate config_hash here. The PREFLIGHT_DRY_RUN_CHECK gate
#                 # already enforces that the preflight actually passed.
#                 if pf_data.get("config_hash") != CURRENT_CONFIG_HASH:
#                     print(f"Evidence mismatch (config_hash) in preflight_results.json: "
#                           f"expected {CURRENT_CONFIG_HASH}, got {pf_data.get('config_hash')}")
#                     FINAL_STATUS = "FAIL"
#             mtime = os.path.getmtime(fpath)
#             if time.time() - mtime > 3600:
#                 print(f"Stale preflight result (timestamp).")
#                 FINAL_STATUS = "FAIL"

#     MODE_REQUIRED_GATES = {
#         "VALIDATE": [
#             "ENV_CHECK", "GPU_CHECK", "DATA_CHECK", "PROJECT_CHECK", "DEPENDENCY_CHECK",
#             "PVT_CHECK", "STATIC_CHECK", "TRANSFORM_CHECK", "MODEL_CHECK", "MOE_CHECK",
#             "LOSS_CHECK", "OPT_CHECK", "DDP_CHECK", "MEMORY_CHECK",
#             "CHECKPOINT_CHECK", "RESUME_CHECK", "EMPTY_BATCH_CHECK"
#         ],
#         "TRAIN": [
#             "ENV_CHECK", "GPU_CHECK", "DATA_CHECK", "PROJECT_CHECK", "DEPENDENCY_CHECK",
#             "PVT_CHECK", "STATIC_CHECK", "TRANSFORM_CHECK", "MODEL_CHECK", "MOE_CHECK",
#             "LOSS_CHECK", "OPT_CHECK", "DDP_CHECK", "MEMORY_CHECK",
#             "CHECKPOINT_CHECK", "RESUME_CHECK", "PREFLIGHT_DRY_RUN_CHECK", "EMPTY_BATCH_CHECK"
#         ],
#         "RESUME": [
#             "ENV_CHECK", "GPU_CHECK", "DATA_CHECK", "PROJECT_CHECK", "DEPENDENCY_CHECK",
#             "STATIC_CHECK", "RESUME_PREFLIGHT_CHECK"
#         ]
#     }
    
#     required = MODE_REQUIRED_GATES.get(RUN_MODE, [])
#     for k in required:
#         v = GATES.get(k)
#         if v != "PASS":
#             print(f"GATE {k} FAILED or NOT RUN.")
#             FINAL_STATUS = "FAIL"

#     with open(os.path.join(PROJECT_ROOT, "final_audit.json"), "w") as f:
#         json.dump({
#             "timestamp": time.time(),
#             "gates": GATES,
#             "final_status": FINAL_STATUS
#         }, f, indent=4)

# print(f"FINAL AUDIT STATUS: {FINAL_STATUS}")

""")

    add_markdown(
        r"""## 18b_smoke_100_images (optional, commented out)

Quick sanity check before committing to the full 50-epoch run: trains a few real epochs (train -> val -> checkpoint -> diagnostics) on a 100-image slice of the data, so you can watch each epoch complete cleanly while it's still fast.

Uses `--preflight`, so it writes to `WXSOD_Preflight` only — it never touches your production `WXSOD_Checkpoints` and never pushes anything to Hugging Face. Safe to re-run as many times as you like.

To run it: select all lines in the cell below and toggle comments off (Edit > Toggle Comment, or Ctrl+/), then run the cell.
""")

    add_code(
        r"""# with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
#     smoke_cfg = json.load(f)

# smoke_cfg["data"]["dataset_root"] = valid_root
# smoke_cfg["data"]["max_samples"] = 100
# smoke_cfg["train"]["epochs"] = 3
# smoke_cfg["train"]["num_workers"] = 2

# SMOKE_CONFIG = os.path.join(PROJECT_ROOT, "experiments", "smoke_100_config.json")
# with open(SMOKE_CONFIG, "w") as f:
#     json.dump(smoke_cfg, f, indent=4)

# print(f"Running 100-image smoke check across {smoke_cfg['train']['epochs']} epochs...")
# subprocess.run([
#     "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp",
#     "--config", SMOKE_CONFIG,
#     "--preflight",
#     "--max_epochs", str(smoke_cfg["train"]["epochs"]),
#     "--max_optimizer_steps", "100000",
# ], cwd=PROJECT_ROOT, check=True)

# _pf_exp_id = runtime_cfg.get("experiment_id", "")
# with open(os.path.join(PREFLIGHT_ROOT, _pf_exp_id, "preflight_results.json"), "r") as f:
#     smoke_result = json.load(f)
# print(json.dumps(smoke_result, indent=2))
# if smoke_result.get("status") != "PASS":
#     raise RuntimeError(f"Smoke check failed: {smoke_result}")
# print(f"Smoke check PASSED - {smoke_cfg['train']['epochs']} epochs completed cleanly on 100 images.")
""")

    add_markdown(
        r"""## 19_train
""")

    add_code(
        r"""# if RUN_MODE == "TRAIN":
#     if FINAL_STATUS != "PASS" or GATES.get("PREFLIGHT_DRY_RUN_CHECK") != "PASS":
#         raise RuntimeError("Refusing to train: Not all gates passed.")
        
#     with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
#         canonical_cfg = json.load(f)
#     from src.config import ExperimentConfig
#     canonical_cfg = ExperimentConfig.from_dict(canonical_cfg).to_dict()
#     from src.train_ddp import get_config_hash
#     canonical_hash = get_config_hash(canonical_cfg, "model_config_hash")
    
# _pf_exp_id = canonical_cfg.get("experiment_id", "")
# with open(os.path.join(PREFLIGHT_ROOT, _pf_exp_id, "preflight_results.json"), "r") as f:
#         pf_data = json.load(f)
#     actual_hash = pf_data.get("config_hash")
    
#     print(f"Validated config hash: {canonical_hash}")
#     print(f"Training config hash:  {actual_hash}")
#     if canonical_hash != actual_hash:
#         raise RuntimeError("Config hash mismatch between canonical config and actual runtime config!")
#     print("MATCH")
    
#     print("Launching final 50-epoch training...")
#     RUNTIME_CONFIG = os.path.join(PROJECT_ROOT, "experiments", "kaggle_runtime.json")
#     subprocess.run([
#         "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp",
#         "--config", RUNTIME_CONFIG,
#         "--overwrite"
#     ], cwd=PROJECT_ROOT, check=True)

""")

    add_markdown(
        r"""## 20_status
""")

    add_code(
        r"""# if RUN_MODE == "TRAIN":
#     if os.path.exists(os.path.join(CHECKPOINT_ROOT, "training_complete.json")):
#         print("Training successfully reached completion state.")
#     else:
#         print("Training did not produce completion marker.")

""")

    add_markdown(
        r"""## 21_resume
""")

    add_code(
        r"""if RUN_MODE == "RESUME":
    print("Initiating Resume Recovery Sequence...")

    # The trainer names every checkpoint after the run's experiment ID, which it
    # derives from the config, and pushes it as "<experiment_id>_latest.pth".
    # Recompute the same ID here so we pull this experiment's checkpoint and
    # never the other account's.
    from src.config import ExperimentConfig
    from src.experiment import generate_experiment_id
    from src.hf_sync import pull_checkpoint

    experiment_id = generate_experiment_id(
        ExperimentConfig.load(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH))
    )
    local_dir = os.path.join(CHECKPOINT_ROOT, experiment_id)
    os.makedirs(local_dir, exist_ok=True)

    if os.path.exists(os.path.join(local_dir, "training_complete.json")):
        raise RuntimeError("TRAINING ALREADY COMPLETE. Cannot resume.")

    # Kaggle wipes /kaggle/working between sessions, so the checkpoint has to
    # come back from the Hub before `--resume latest` can find it.
    # pull_checkpoint verifies the sha256 against the manifest as it downloads.
    local_latest = os.path.join(local_dir, "latest.pth")
    if not pull_checkpoint(f"{experiment_id}_latest.pth", local_latest, force=True):
        raise RuntimeError(
            f"{experiment_id}_latest.pth is not on Hugging Face — nothing to resume from."
        )
    print(f"Resume checkpoint ready: {local_latest}")

    with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
        runtime_cfg = json.load(f)

    runtime_cfg["data"]["dataset_root"] = valid_root
    RUNTIME_CONFIG = os.path.join(PROJECT_ROOT, "experiments", "kaggle_runtime.json")

    with open(RUNTIME_CONFIG, "w") as f:
        json.dump(runtime_cfg, f, indent=4)

    print("Executing Torchrun Resume...")
    subprocess.run([
        "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp",
        "--config", RUNTIME_CONFIG,
        "--resume", "latest"
    ], cwd=PROJECT_ROOT, check=True)

""")

    add_markdown(
        r"""## 22_evaluate
""")

    add_code(
        r"""import os, sys, subprocess
from src.config import ExperimentConfig
from src.experiment import generate_experiment_id
from src.hf_sync import pull_checkpoint

if RUN_MODE == "EVALUATE":
    print("Evaluating Best Checkpoint...")
    # The trainer writes to WXSOD_Checkpoints/<experiment_id>/ and derives that
    # id from the config at startup, so recompute it here rather than guessing a
    # path — and pull it from the Hub when this session is fresh.
    experiment_id = generate_experiment_id(
        ExperimentConfig.load(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH))
    )
    ckpt_dir = os.path.join(CHECKPOINT_ROOT, experiment_id)
    best_ckpt = os.path.join(ckpt_dir, "best.pth")
    if not os.path.exists(best_ckpt):
        os.makedirs(ckpt_dir, exist_ok=True)
        pull_checkpoint(f"{experiment_id}_best.pth", best_ckpt)
    assert os.path.exists(best_ckpt), f"No checkpoint at {best_ckpt}"

    data_dir = valid_root  # use the dynamically-validated dataset root, not a guessed path
    eval_out_dir = "/kaggle/working/WXSOD_EvalResults"
    os.makedirs(eval_out_dir, exist_ok=True)
    print(f"Checkpoint: {best_ckpt}")

    process = subprocess.Popen(
        [
            "python", "-u", "-m", "src.evaluate",
            "--checkpoint", best_ckpt,
            "--dataset", "both",
            "--data_dir", data_dir,
            "--out_dir", eval_out_dir,
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    while True:
        chunk = process.stdout.read(1)
        if not chunk and process.poll() is not None:
            break
        if chunk:
            sys.stdout.write(chunk.decode(errors="replace"))
            sys.stdout.flush()

    retcode = process.wait()
    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, process.args)

    print("\n--- Files in eval_out_dir ---")
    for f in os.listdir(eval_out_dir):
        print(f)
""")

    add_markdown(
        r"""## 23_load_model_for_analysis
""")

    add_code(
        r"""import os, json, torch
import numpy as np
from src.model import SpatialMoESODNet
from src.dataset import get_dataloaders
from src.config import ExperimentConfig
from src.experiment import generate_experiment_id


def load_model_from_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get('config', {}).get('model', {})
    assert 'num_experts' in model_cfg, 'checkpoint config missing num_experts'
    # Every architecture-defining field must come from the checkpoint config, or
    # evaluation silently runs a different model than the one that was trained.
    model = SpatialMoESODNet(
        dim=model_cfg.get('working_dim', 256),
        num_experts=model_cfg.get('num_experts', 8),
        k=model_cfg.get('top_k', 2),
        gate_mode=model_cfg.get('gate_mode', 'renormalized'),
        window_size=model_cfg.get('window_size', 7),
        use_deep_supervision=model_cfg.get('deep_supervision', False)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded: {ckpt_path}")
    print(f"  Epoch {checkpoint.get('epoch','?')+1}  Best MAE {checkpoint.get('best_metric','?')}")
    return model, model_cfg, checkpoint


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NpEncoder)
    print(f"Saved {path}")


if RUN_MODE == "EVALUATE":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_id = generate_experiment_id(
        ExperimentConfig.load(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH))
    )
    ckpt_dir = os.path.join(CHECKPOINT_ROOT, experiment_id)
    best_ckpt = os.path.join(ckpt_dir, 'best.pth')
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(ckpt_dir, 'latest.pth')
    assert os.path.exists(best_ckpt), f"No checkpoint found at {best_ckpt}"

    model, model_cfg, checkpoint = load_model_from_checkpoint(best_ckpt, device)

    _, _, test_sys_loader, test_real_loader = get_dataloaders(
        root_dir=valid_root, image_size=384, batch_size=1, num_workers=2
    )

    RESULTS_ROOT = '/kaggle/working/analysis_results'
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    for sub in ['compute_cost', 'proxy_ablations', 'routing_entropy', 'qualitative', 'diagnostics']:
        os.makedirs(os.path.join(RESULTS_ROOT, sub), exist_ok=True)
""")

    add_markdown(
        r"""## 23a_compute_cost
""")

    add_code(
        r"""# !pip install thop --quiet
# import time
# from thop import profile

# dummy = torch.randn(1, 3, 384, 384).to(device)
# macs, params = profile(model, inputs=(dummy,), verbose=False)

# with torch.no_grad():
#     for _ in range(10):
#         model(dummy)  # warmup
#     if device.type == 'cuda': torch.cuda.synchronize()
#     start = time.time()
#     N = 50
#     for _ in range(N):
#         model(dummy)
#     if device.type == 'cuda': torch.cuda.synchronize()
#     elapsed = time.time() - start
# fps = N / elapsed

# compute_cost = {'params_M': round(params/1e6, 2), 'macs_G': round(macs/1e9, 2), 'fps': round(fps, 2)}
# print(compute_cost)
# save_json(os.path.join(RESULTS_ROOT, 'compute_cost', 'compute_cost.json'), compute_cost)
""")

    add_markdown(
        r"""## 23b_proxy_ablations
""")

    add_code(
        r"""if RUN_MODE == "EVALUATE":
    from datetime import datetime, timezone
    from src.evaluate import evaluate

    ablation_runs = [
        ('full_model',           None),
        ('no_entropy_fusion',    {'disable_entropy': True}),
        ('forced_expert_scale4', {'scale': 4, 'expert_id': 0}),
        ('random_routing',       {'random_routing': True}),
    ]

    ablation_results = {}
    for name, cfg in ablation_runs:
        print(f"\n{'='*60}")
        print(f"Ablation: {name}   cfg={cfg}")
        if name == 'random_routing':
            # Multi-seed variance for random routing
            seeds = [42, 1, 7, 123]
            seed_results = []
            for seed in seeds:
                torch.manual_seed(seed)
                out_dir = os.path.join(RESULTS_ROOT, 'proxy_ablations', f"{name}_seed{seed}")
                results = evaluate(model, test_real_loader, out_dir, ablation_cfg=cfg)
                mae = results['global']['MAE']
                sm  = results['global']['S_measure']
                print(f"  [Seed {seed}] MAE={mae:.4f}   S_measure={sm:.4f}")
                seed_results.append({'seed': seed, 'MAE': mae, 'S_measure': sm})
            ablation_results[name] = {'random_routing_seeds': seed_results}
        else:
            out_dir = os.path.join(RESULTS_ROOT, 'proxy_ablations', name)
            results = evaluate(model, test_real_loader, out_dir, ablation_cfg=cfg)
            mae = results['global']['MAE']
            sm  = results['global']['S_measure']
            print(f"  MAE={mae:.4f}   S_measure={sm:.4f}")
            ablation_results[name] = {'MAE': mae, 'S_measure': sm}

    output = {
        'checkpoint': best_ckpt,
        'dataset': 'test_real',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'ablations': ablation_results,
    }
    json_out = os.path.join(RESULTS_ROOT, 'proxy_ablations', 'proxy_ablation_results.json')
    save_json(json_out, output)
""")

    add_markdown(
        r"""## 23c_forced_expert_sanity_check
""")

    add_code(
        r"""# # Targets stride/8 (scale=8), the stage with confirmed DEAD_EXPERT [0, 5], as opposed to 23b's stride/4 (scale=4) check.
# print("\nForced Expert Sanity Check (Scale 8, Expert 0):")
# forced_results = evaluate(
#     model,
#     test_real_loader,
#     output_dir=os.path.join(RESULTS_ROOT, 'proxy_ablations', 'force_expert_scale8_out'),
#     use_tta=False,
#     ablation_cfg={'scale': 8, 'expert_id': 0}
# )
# save_json(os.path.join(RESULTS_ROOT, 'proxy_ablations', 'force_expert_scale8_result.json'), forced_results)
""")

    add_markdown(
        r"""## 23d_routing_entropy
""")

    add_code(
        r"""if RUN_MODE == "EVALUATE":
    from collections import defaultdict

    def collect_entropy_stats(model, dataloader, device, max_batches=None):
        model.eval()
        scale_entropies = defaultdict(list)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches and i >= max_batches: break
                images = batch['image'].to(device)
                _, moe_outputs = model(images)
                for scale_name, out in zip(['scale_4', 'scale_8', 'scale_16'], moe_outputs):
                    scale_entropies[scale_name].append(out.entropy.mean().item())
        return {k: (sum(v)/len(v)) for k, v in scale_entropies.items()}

    sys_entropy = collect_entropy_stats(model, test_sys_loader, device)
    real_entropy = collect_entropy_stats(model, test_real_loader, device)

    entropy_comparison = {'synthetic': sys_entropy, 'real': real_entropy}
    print(entropy_comparison)
    save_json(os.path.join(RESULTS_ROOT, 'routing_entropy', 'entropy_comparison.json'), entropy_comparison)
""")

    add_markdown(
        r"""## 23e_qualitative_figure
""")

    add_code(
        r"""if RUN_MODE == "EVALUATE":
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    test_real_ds = test_real_loader.dataset
    snow_idx, rain_or_fog_idx, light_idx = -1, -1, -1

    # TODO(Task 4): replace with canonical weather parser
    for idx in range(len(test_real_ds)):
        stem = test_real_ds.samples[idx][2]
        weather = stem.split('-')[0].lower()
        if 'snow' in weather or 'snow' in stem.lower(): snow_idx = idx
        elif 'rain' in weather or 'rain' in stem.lower() or 'fog' in weather or 'fog' in stem.lower(): rain_or_fog_idx = idx
        elif 'light' in weather or 'light' in stem.lower() or 'sun' in stem.lower(): light_idx = idx
        if snow_idx != -1 and rain_or_fog_idx != -1 and light_idx != -1: break

    if snow_idx == -1: snow_idx = 0
    if rain_or_fog_idx == -1: rain_or_fog_idx = 1
    if light_idx == -1: light_idx = 2

    def make_qualitative_grid(model, dataset, indices, device, save_path):
        fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3*len(indices)))
        model.eval()
        with torch.no_grad():
            for row, idx in enumerate(indices):
                sample = dataset[idx]
                image = sample['image'].unsqueeze(0).to(device)
                gt_path = sample['meta']['gt_path']
                out, _ = model(image)
                pred = torch.sigmoid(out.saliency_logits)[0,0].cpu().numpy()
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
                img_disp = sample['image'].permute(1,2,0).cpu().numpy()
                img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
            
                axes[row,0].imshow(img_disp); axes[row,0].set_title(f"Input (idx {idx})")
                axes[row,1].imshow(gt, cmap='gray'); axes[row,1].set_title("GT")
                axes[row,2].imshow(pred, cmap='gray'); axes[row,2].set_title("Prediction")
                for ax in axes[row]: ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"Saved to {save_path}")

    make_qualitative_grid(model, test_real_ds, [snow_idx, rain_or_fog_idx, light_idx], device, os.path.join(RESULTS_ROOT, 'qualitative', 'qualitative_grid.png'))
""")

    add_markdown(
        r"""## 23f_diagnostics_and_upload
""")

    add_code(
        r"""if RUN_MODE == "EVALUATE":
    import subprocess
    import shutil
    import glob
    from huggingface_hub import login, HfApi, create_repo
    from kaggle_secrets import UserSecretsClient

    print("[1/3] Running Padding Audit...")
    subprocess.run([
        "python", "-m", "src.diagnose_padding",
        "--data_dir", valid_root,
        "--out_dir", os.path.join(RESULTS_ROOT, 'diagnostics')
    ], cwd=PROJECT_ROOT, check=True)

    print("\n[2/3] Running MoE Diagnostics (Routing & Expert Similarity)...")
    subprocess.run([
        "python", "-m", "src.run_moe_diagnostics",
        "--checkpoint", best_ckpt,
        "--data_dir", valid_root,
        "--out_dir", os.path.join(RESULTS_ROOT, 'diagnostics')
    ], cwd=PROJECT_ROOT, check=True)


""")

    add_markdown(
        r"""## 23g_ablation_screening
""")

    add_code(
        r"""if RUN_MODE == "TRAIN":
    import subprocess
    import os

    PROJECT_ROOT = "/kaggle/working/spatial_moe_sod"

    subprocess.run([
        "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp",
        "--config", ACTIVE_CONFIG_PATH, "--overwrite"
    ], cwd=PROJECT_ROOT, check=True)
""")

    add_markdown(
        r"""# Campare Full and moe16 dense Ablation""")

    add_code(
        r"""# Skipped for this run: run_ablation_comparison.py hardcodes
# experiments = ["ablation_full_moe", "ablation_moe16_dense"], which does not
# include this run's checkpoint dir. Re-enable / rewrite once the num_experts
# ablation has its own comparison logic.
# subprocess.run([
#     "python", "-m", "src.run_ablation_comparison"
# ], cwd=PROJECT_ROOT, check=True)
""")

    add_markdown(
        r"""# 24_Upload_To_Hugging_Face""")

    add_code(
        r"""RESULTS_ROOT = '/kaggle/working/analysis_results'
if RUN_MODE == "EVALUATE":
    import glob
    import shutil
    from huggingface_hub import login, HfApi, create_repo
    from kaggle_secrets import UserSecretsClient
    from src.config import ExperimentConfig
    from src.experiment import generate_experiment_id

    # Name the HF folder after the run being evaluated, derived from the active
    # config exactly as the trainer derives it.  The analysis results then sit
    # under the same identifier as the checkpoints and diagnostics they describe,
    # instead of a hand-typed label that can drift out of sync.
    RUN_LABEL = generate_experiment_id(
        ExperimentConfig.load(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH))
    )
    print(f"\n[3/3] Uploading results for {RUN_LABEL} to Hugging Face...")
    # Copy the 22_evaluate outputs into RESULTS_ROOT so everything is in one tree
    eval_dir = "/kaggle/working/WXSOD_EvalResults"
    if os.path.exists(eval_dir):
        for ext in ["*.txt", "*.json"]:
            for filepath in glob.glob(f"{eval_dir}/**/{ext}", recursive=True):
                dest = os.path.join(RESULTS_ROOT, 'eval_results', os.path.basename(filepath))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(filepath, dest)

    try:
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        login(token=hf_token)
        REPO_ID = os.environ.get("HF_REPO_ID", "Avi2006/spatial-moe-results")
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=True)

        api = HfApi()
        # Upload the ENTIRE RESULTS_ROOT folder
        uploaded_url = api.upload_folder(
            folder_path=RESULTS_ROOT,
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo=f"analysis_results/{RUN_LABEL}"
        )
        print(f"\nUpload successful!")
        print(f"Destination: {uploaded_url}")
        print("\nManifest of uploaded files (local paths in RESULTS_ROOT):")
        for root, dirs, files in os.walk(RESULTS_ROOT):
            for f in files:
                print(os.path.relpath(os.path.join(root, f), RESULTS_ROOT))
    except Exception as e:
        print(f"Failed to upload to Hugging Face: {e}")
""")

    notebook = {
        "cells": cells,
        "metadata": {
        "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
        },
        "language_info": {
                "name": "python",
                "version": "3.12.13",
                "mimetype": "text/x-python",
                "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                },
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py"
        }
},
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(OUTPUT_NOTEBOOK, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"{OUTPUT_NOTEBOOK} generated successfully ({len(cells)} cells).")


if __name__ == "__main__":
    create_notebook()
