import os
import shutil
import json
import subprocess

def create_notebook():
    cells = []
    
    def add_markdown(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")]
        })

    # CELL 00: 00_hf_sync
    add_markdown("""## 00_hf_sync""")
    add_code("""# uncomment and run this if you want to manually push your latest checkpoints to Hugging Face
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
# from src.hf_sync import push_checkpoint
#
# ckpt_dir = "/kaggle/working/WXSOD_Checkpoints"
# for name in ["best.pth", "latest.pth", "training_complete.json", "latest_prev.pth"]:
#     path = os.path.join(ckpt_dir, name)
#     if os.path.exists(path):
#         print(f"Uploading {name}...")
#         push_checkpoint(path, name=name)
# print("Done!")
""")

    # CELL 02: 01_config
    add_markdown("""## 01_config""")
    add_code("""import os
import json
import shutil
import hashlib

# Configuration & Modes
# RUN_MODE strictly governs the allowed execution path.
# Allowed: "VALIDATE", "TRAIN", "RESUME", "EVALUATE"
RUN_MODE = "TRAIN"
ACTIVE_CONFIG_PATH = "experiments/v4_noise_floor.json"

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

    # CELL 04: 02_environment
    add_markdown("""## 02_environment""")
    add_code("""import torch
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

    # CELL 06: 03_dataset_acquire
    add_markdown("""## 03_dataset_acquire""")
    add_code("""import subprocess

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

    # CELL 08: 04_dataset_validate
    add_markdown("""## 04_dataset_validate""")
    add_code("""import glob

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

    # CELL 10: 05_project_deploy
    add_markdown("""## 05_project_deploy""")
    add_code("""import sys
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

    # CELL 12: 06_dependencies
    add_markdown("""## 06_dependencies""")
    add_code("""import importlib
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

    # CELL 14: 07_pretrained_b4
    add_markdown("""## 07_pretrained_b4""")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
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

    # CELL 16: 08_static_checks
    add_markdown("""## 08_static_checks""")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    try:
        import src.model
        import src.train_ddp
        import src.loss
        mark_gate("STATIC_CHECK", "PASS")
    except ImportError as e:
        mark_gate("STATIC_CHECK", "FAIL")
        raise e
""")

    # CELL 18: 09_data_transform
    add_markdown("""## 09_data_transform""")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
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

    # CELL 20: 10_model_shapes
    add_markdown("""## 10_model_shapes""")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
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

    # CELL 22: 13_empty_batch_check
    add_markdown("""## 13_empty_batch_check""")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    print("Running Empty Batch Check...")
    with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
        canonical_cfg = json.load(f)
    from src.config import ExperimentConfig
    canonical_cfg = ExperimentConfig.from_dict(canonical_cfg).to_dict()
    from src.train_ddp import get_config_hash
    CURRENT_CONFIG_HASH = get_config_hash(canonical_cfg, "model_config_hash")
    CURRENT_RUN_ID = canonical_cfg.get("run_id") or "test_run_id"

    res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "empty_batch", "--data_root", valid_root, "--result_file", "test_empty_batch.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
    with open(os.path.join(PROJECT_ROOT, "test_empty_batch.json"), "r") as f:
        data = json.load(f)
        if data.get("status") == "PASS":
            mark_gate("EMPTY_BATCH_CHECK", "PASS")
        else:
            mark_gate("EMPTY_BATCH_CHECK", "FAIL")
            raise RuntimeError(f"Empty Batch Check Failed: {data}")
""")

    # CELL 24: 14_smoke_2gpu
    add_markdown("""## 14_smoke_2gpu""")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
        canonical_cfg = json.load(f)
    from src.config import ExperimentConfig
    canonical_cfg = ExperimentConfig.from_dict(canonical_cfg).to_dict()
    from src.train_ddp import get_config_hash
    CURRENT_CONFIG_HASH = get_config_hash(canonical_cfg, "model_config_hash")
    CURRENT_RUN_ID = canonical_cfg.get("run_id") or "test_run_id"

    print("Running 2-GPU DDP Smoke Test...")
    res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "ddp", "--data_root", valid_root, "--result_file", "test_2gpu.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
    with open(os.path.join(PROJECT_ROOT, "test_2gpu.json"), "r") as f:
        data = json.load(f)
        if data.get("status") == "PASS" and data.get("world_size") == 2 and set(data.get("ranks_completed", [])) == {0, 1}:
            mark_gate("DDP_CHECK", "PASS")
            mark_gate("MOE_CHECK", data.get("moe_check", "FAIL"), "Sparsity/Token Identity")
            mark_gate("LOSS_CHECK", data.get("loss_check", "FAIL"), "Loss Gradients/Behavior")
            mark_gate("OPT_CHECK", data.get("optimizer_check", "FAIL"), "Optimizer Groups/Behavior")
        else:
            mark_gate("DDP_CHECK", "FAIL")
            raise RuntimeError(f"DDP Smoke Failed: {data}")
""")

    # CELL 26: 15_production_calibration
    add_markdown("""## 15_production_calibration""")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    print("Running 2-GPU Production Memory Calibration...")
    res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "memory", "--data_root", valid_root, "--result_file", "test_mem.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
    with open(os.path.join(PROJECT_ROOT, "test_mem.json"), "r") as f:
        data = json.load(f)
        if data.get("status") == "PASS":
            mark_gate("MEMORY_CHECK", "PASS", f"Peak: {data.get('peak_allocated_gb')} GB")
        else:
            mark_gate("MEMORY_CHECK", "FAIL")
            raise RuntimeError(data.get("reason"))
""")

    # CELL 28: 16_checkpoint_test
    add_markdown("""## 16_checkpoint_test""")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    print("Running Checkpoint Write Test...")
    res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "resume_a", "--data_root", valid_root, "--result_file", "test_ckpt.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
    with open(os.path.join(PROJECT_ROOT, "test_ckpt.json"), "r") as f:
        if json.load(f).get("status") == "PASS":
            mark_gate("CHECKPOINT_CHECK", "PASS")
        else:
            mark_gate("CHECKPOINT_CHECK", "FAIL")
""")

    # CELL 30: 17_resume_test
    add_markdown("""## 17_resume_test""")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    print("Running Checkpoint Resume Test...")
    res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "resume_b", "--data_root", valid_root, "--result_file", "test_res.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
    with open(os.path.join(PROJECT_ROOT, "test_res.json"), "r") as f:
        if json.load(f).get("status") == "PASS":
            mark_gate("RESUME_CHECK", "PASS")
        else:
            mark_gate("RESUME_CHECK", "FAIL")
""")

    # CELL 32: 18_final_gate
    add_markdown("""## 18_final_gate""")
    add_code("""if RUN_MODE == "TRAIN":
    print("Running Preflight Dry Run (2-5 steps)...")
    with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
        runtime_cfg = json.load(f)
        
    runtime_cfg["data"]["dataset_root"] = valid_root
    
    RUNTIME_CONFIG = os.path.join(PROJECT_ROOT, "experiments", "kaggle_runtime.json")
    
    with open(RUNTIME_CONFIG, "w") as f:
        json.dump(runtime_cfg, f, indent=4)
        
    print("Runtime dataset root:")
    print(runtime_cfg["data"]["dataset_root"])
    print("Runtime config:")
    print(RUNTIME_CONFIG)
    
    res = subprocess.run([
        "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp",
        "--config", RUNTIME_CONFIG,
        "--preflight", "--max_optimizer_steps", "5"
    ], cwd=PROJECT_ROOT, check=True)
    
    with open(os.path.join(PREFLIGHT_ROOT, "preflight_results.json"), "r") as f:
        pf_data = json.load(f)
    with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
        canonical_cfg = json.load(f)
    from src.config import ExperimentConfig
    canonical_cfg = ExperimentConfig.from_dict(canonical_cfg).to_dict()
    from src.train_ddp import get_config_hash
    canonical_hash = get_config_hash(canonical_cfg, "model_config_hash")
    
    if pf_data.get("status") == "PASS":
        mark_gate("PREFLIGHT_DRY_RUN_CHECK", "PASS", run_id=pf_data.get("run_id"), config_hash=pf_data.get("config_hash"))
    else:
        mark_gate("PREFLIGHT_DRY_RUN_CHECK", "FAIL")
        raise RuntimeError(f"Preflight validation failed: {pf_data}")

FINAL_STATUS = "PASS"
# Some gates are only for train/validate
if RUN_MODE in ["VALIDATE", "TRAIN"]:
    with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
        canonical_cfg = json.load(f)
    from src.config import ExperimentConfig
    canonical_cfg = ExperimentConfig.from_dict(canonical_cfg).to_dict()
    from src.train_ddp import get_config_hash
    CURRENT_CONFIG_HASH = get_config_hash(canonical_cfg, "model_config_hash")
    CURRENT_RUN_ID = canonical_cfg.get("run_id") or "test_run_id"

    # 9. Current-run evidence validation
    import time
    required_files = [
        "test_empty_batch.json", "test_2gpu.json", "test_mem.json", "test_ckpt.json", "test_res.json"
    ]
    for filename in required_files:
        fpath = os.path.join(PROJECT_ROOT, filename)
        if not os.path.exists(fpath):
            print(f"Missing required test result: {filename}")
            FINAL_STATUS = "FAIL"
            continue
        with open(fpath, "r") as f:
            data = json.load(f)
            if data.get("status") != "PASS":
                print(f"Test result not PASS: {filename}")
                FINAL_STATUS = "FAIL"
            if data.get("run_id") != CURRENT_RUN_ID or data.get("config_hash") != CURRENT_CONFIG_HASH:
                print(f"Evidence mismatch (run_id/config_hash) in {filename}")
                FINAL_STATUS = "FAIL"
        
        # Freshness check: file must be modified recently (within the last hour)
        mtime = os.path.getmtime(fpath)
        if time.time() - mtime > 3600:
            print(f"Stale test result (timestamp): {filename}")
            FINAL_STATUS = "FAIL"
            
    if RUN_MODE == "TRAIN":
        fpath = os.path.join(PREFLIGHT_ROOT, "preflight_results.json")
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                pf_data = json.load(f)
                # Note: run_id is always a fresh timestamped value (e.g. EXP_...timestamp...),
                # so we only validate config_hash here. The PREFLIGHT_DRY_RUN_CHECK gate
                # already enforces that the preflight actually passed.
                if pf_data.get("config_hash") != CURRENT_CONFIG_HASH:
                    print(f"Evidence mismatch (config_hash) in preflight_results.json: "
                          f"expected {CURRENT_CONFIG_HASH}, got {pf_data.get('config_hash')}")
                    FINAL_STATUS = "FAIL"
            mtime = os.path.getmtime(fpath)
            if time.time() - mtime > 3600:
                print(f"Stale preflight result (timestamp).")
                FINAL_STATUS = "FAIL"

    MODE_REQUIRED_GATES = {
        "VALIDATE": [
            "ENV_CHECK", "GPU_CHECK", "DATA_CHECK", "PROJECT_CHECK", "DEPENDENCY_CHECK",
            "PVT_CHECK", "STATIC_CHECK", "TRANSFORM_CHECK", "MODEL_CHECK", "MOE_CHECK",
            "LOSS_CHECK", "OPT_CHECK", "DDP_CHECK", "MEMORY_CHECK",
            "CHECKPOINT_CHECK", "RESUME_CHECK", "EMPTY_BATCH_CHECK"
        ],
        "TRAIN": [
            "ENV_CHECK", "GPU_CHECK", "DATA_CHECK", "PROJECT_CHECK", "DEPENDENCY_CHECK",
            "PVT_CHECK", "STATIC_CHECK", "TRANSFORM_CHECK", "MODEL_CHECK", "MOE_CHECK",
            "LOSS_CHECK", "OPT_CHECK", "DDP_CHECK", "MEMORY_CHECK",
            "CHECKPOINT_CHECK", "RESUME_CHECK", "PREFLIGHT_DRY_RUN_CHECK", "EMPTY_BATCH_CHECK"
        ],
        "RESUME": [
            "ENV_CHECK", "GPU_CHECK", "DATA_CHECK", "PROJECT_CHECK", "DEPENDENCY_CHECK",
            "STATIC_CHECK", "RESUME_PREFLIGHT_CHECK"
        ]
    }
    
    required = MODE_REQUIRED_GATES.get(RUN_MODE, [])
    for k in required:
        v = GATES.get(k)
        if v != "PASS":
            print(f"GATE {k} FAILED or NOT RUN.")
            FINAL_STATUS = "FAIL"

    with open(os.path.join(PROJECT_ROOT, "final_audit.json"), "w") as f:
        json.dump({
            "timestamp": time.time(),
            "gates": GATES,
            "final_status": FINAL_STATUS
        }, f, indent=4)

print(f"FINAL AUDIT STATUS: {FINAL_STATUS}")
""")

    # CELL 34: 18b_smoke_100_images (optional, commented out)
    add_markdown("""## 18b_smoke_100_images (optional, commented out)

Quick sanity check before committing to the full 50-epoch run: trains a few real epochs (train -> val -> checkpoint -> diagnostics) on a 100-image slice of the data, so you can watch each epoch complete cleanly while it's still fast.

Uses `--preflight`, so it writes to `WXSOD_Preflight` only — it never touches your production `WXSOD_Checkpoints` and never pushes anything to Hugging Face. Safe to re-run as many times as you like.

To run it: select all lines in the cell below and toggle comments off (Edit > Toggle Comment, or Ctrl+/), then run the cell.""")
    add_code("""# with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
#     smoke_cfg = json.load(f)
#
# smoke_cfg["data"]["dataset_root"] = valid_root
# smoke_cfg["data"]["max_samples"] = 100
# smoke_cfg["train"]["epochs"] = 3
# smoke_cfg["train"]["num_workers"] = 2
#
# SMOKE_CONFIG = os.path.join(PROJECT_ROOT, "experiments", "smoke_100_config.json")
# with open(SMOKE_CONFIG, "w") as f:
#     json.dump(smoke_cfg, f, indent=4)
#
# print(f"Running 100-image smoke check across {smoke_cfg['train']['epochs']} epochs...")
# subprocess.run([
#     "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp",
#     "--config", SMOKE_CONFIG,
#     "--preflight",
#     "--max_epochs", str(smoke_cfg["train"]["epochs"]),
#     "--max_optimizer_steps", "100000",
# ], cwd=PROJECT_ROOT, check=True)
#
# with open(os.path.join(PREFLIGHT_ROOT, "preflight_results.json"), "r") as f:
#     smoke_result = json.load(f)
# print(json.dumps(smoke_result, indent=2))
# if smoke_result.get("status") != "PASS":
#     raise RuntimeError(f"Smoke check failed: {smoke_result}")
# print(f"Smoke check PASSED - {smoke_cfg['train']['epochs']} epochs completed cleanly on 100 images.")""")

    # CELL 36: 19_train
    add_markdown("""## 19_train""")
    add_code("""if RUN_MODE == "TRAIN":
    if FINAL_STATUS != "PASS" or GATES.get("PREFLIGHT_DRY_RUN_CHECK") != "PASS":
        raise RuntimeError("Refusing to train: Not all gates passed.")
        
    with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
        canonical_cfg = json.load(f)
    from src.config import ExperimentConfig
    canonical_cfg = ExperimentConfig.from_dict(canonical_cfg).to_dict()
    from src.train_ddp import get_config_hash
    canonical_hash = get_config_hash(canonical_cfg, "model_config_hash")
    
    with open(os.path.join(PREFLIGHT_ROOT, "preflight_results.json"), "r") as f:
        pf_data = json.load(f)
    actual_hash = pf_data.get("config_hash")
    
    print(f"Validated config hash: {canonical_hash}")
    print(f"Training config hash:  {actual_hash}")
    if canonical_hash != actual_hash:
        raise RuntimeError("Config hash mismatch between canonical config and actual runtime config!")
    print("MATCH")
    
    print("Launching final 50-epoch training...")
    RUNTIME_CONFIG = os.path.join(PROJECT_ROOT, "experiments", "kaggle_runtime.json")
    subprocess.run([
        "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp",
        "--config", RUNTIME_CONFIG,
        "--overwrite"
    ], cwd=PROJECT_ROOT, check=True)
""")

    # CELL 38: 20_status
    add_markdown("""## 20_status""")
    add_code("""if RUN_MODE == "TRAIN":
    if os.path.exists(os.path.join(CHECKPOINT_ROOT, "training_complete.json")):
        print("Training successfully reached completion state.")
    else:
        print("Training did not produce completion marker.")
""")

    # CELL 40: 21_resume
    add_markdown("""## 21_resume""")
    add_code("""if RUN_MODE == "RESUME":
    print("Initiating Resume Recovery Sequence...")
    local_latest = os.path.join(CHECKPOINT_ROOT, "latest.pth")
    if os.path.exists(os.path.join(CHECKPOINT_ROOT, "training_complete.json")):
        raise RuntimeError("TRAINING ALREADY COMPLETE. Cannot resume.")
        
    if not os.path.exists(local_latest):
        raise RuntimeError("latest.pth not found in /kaggle/working/WXSOD_Checkpoints and could not be fetched from Hugging Face.")
            
    with open(os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH), "r") as f:
        runtime_cfg = json.load(f)
        
    runtime_cfg["data"]["dataset_root"] = valid_root
    RUNTIME_CONFIG = os.path.join(PROJECT_ROOT, "experiments", "kaggle_runtime.json")
    
    with open(RUNTIME_CONFIG, "w") as f:
        json.dump(runtime_cfg, f, indent=4)
            
    print("Running Resume Preflight Dry Run...")
    subprocess.run([
        "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp", 
        "--config", RUNTIME_CONFIG,
        "--resume", "latest", "--preflight", "--max_optimizer_steps", "5"
    ], cwd=PROJECT_ROOT, check=True)
    with open(os.path.join(PREFLIGHT_ROOT, "preflight_results.json"), "r") as f:
        pf_data = json.load(f)
    if pf_data.get("status") == "PASS":
        mark_gate("RESUME_PREFLIGHT_CHECK", "PASS", run_id=pf_data.get("run_id"), config_hash=pf_data.get("config_hash"))
    else:
        mark_gate("RESUME_PREFLIGHT_CHECK", "FAIL")
        raise RuntimeError("Resume preflight failed.")
            
    print("Executing Torchrun Resume...")
    subprocess.run([
        "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp", 
        "--config", RUNTIME_CONFIG,
        "--resume", "latest"
    ], cwd=PROJECT_ROOT, check=True)
""")

    # CELL 42: 22_evaluate
    add_markdown("""## 22_evaluate""")
    add_code("""import sys

if RUN_MODE in ["TRAIN", "EVALUATE"]:
    print("Evaluating Best Checkpoint...")
    best_ckpt = os.path.join(CHECKPOINT_ROOT, "best.pth")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(CHECKPOINT_ROOT, "latest.pth")
    data_dir = valid_root  # use the dynamically-validated dataset root, not a guessed path
    eval_out_dir = "/kaggle/working/WXSOD_EvalResults"
    os.makedirs(eval_out_dir, exist_ok=True)
    print("Started the Process")

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

    print("\\n--- Files in eval_out_dir ---")
    for f in os.listdir(eval_out_dir):
        print(f)
""")

    # CELL 44: 23_proxy_ablations
    add_markdown("""## 23_proxy_ablations""")
    add_code("""# ── 23_proxy_ablations ─────────────────────────────────────────────────
# Self-contained: runs 4 inference-time ablations on the frozen best
# checkpoint against test_real, assembles one JSON, uploads to HF.
# Re-runnable without re-executing earlier cells.
# ───────────────────────────────────────────────────────────────────────
import os, sys, json
os.environ["TQDM_DISABLE"] = "1"  # suppress per-batch progress bars
from datetime import datetime, timezone

import torch

from src.model import SpatialMoESODNet
from src.evaluate import evaluate
from src.dataset import get_dataloaders

# ── Paths ───────────────────────────────────────────────────────────────
best_ckpt = os.path.join(CHECKPOINT_ROOT, "best.pth")
if not os.path.exists(best_ckpt):
    best_ckpt = os.path.join(CHECKPOINT_ROOT, "latest.pth")
assert os.path.exists(best_ckpt), f"No checkpoint found at {best_ckpt}"

DATA_DIR    = valid_root  # use the dynamically-validated dataset root, not a guessed path
ABLATION_OUT = "/kaggle/working/proxy_ablations"
JSON_OUT     = "/kaggle/working/proxy_ablation_results.json"

# ── Load model once ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)

model_cfg = checkpoint.get("config", {}).get("model", {})
print(f"num_experts from checkpoint config: {model_cfg.get('num_experts', 'MISSING - defaulting to 6!')}")
assert "num_experts" in model_cfg, "checkpoint config missing num_experts — verify before trusting results"

model = SpatialMoESODNet(
    use_deep_supervision=model_cfg.get("deep_supervision", False),
    num_experts=model_cfg.get("num_experts", 6),
    window_size=model_cfg.get("window_size", 8),
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded: {best_ckpt}")
print(f"  Epoch {checkpoint.get('epoch','?')+1}  Best MAE {checkpoint.get('best_metric','?')}")

# ── Build test_real dataloader ───────────────────────────────────────────
_, _, _, test_real_loader = get_dataloaders(
    root_dir=DATA_DIR, batch_size=1, num_workers=2, distributed=False
)
print(f"test_real samples: {len(test_real_loader.dataset)}")
assert len(test_real_loader.dataset) > 0, "test_real dataloader is empty!"

# ── Ablation run definitions ─────────────────────────────────────────────
ablation_runs = [
    ("full_model",           None),
    ("no_entropy_fusion",    {"disable_entropy": True}),
    ("random_routing",       {"random_routing": True}),
    ("forced_expert_scale4", {"scale": 4, "expert_id": 0}),
]

# ── Run evaluations ──────────────────────────────────────────────────────
ablation_results = {}
for name, cfg in ablation_runs:
    print(f"\\n{'='*60}")
    print(f"Ablation: {name}   cfg={cfg}")
    if name == "random_routing":
        torch.manual_seed(42)
    out_dir = os.path.join(ABLATION_OUT, name)
    results = evaluate(model, test_real_loader, out_dir, ablation_cfg=cfg)
    mae = results["global"]["MAE"]
    sm  = results["global"]["S_measure"]
    print(f"  MAE={mae:.4f}   S_measure={sm:.4f}")
    ablation_results[name] = {"MAE": mae, "S_measure": sm}

# ── Assemble + write JSON ────────────────────────────────────────────────
output = {
    "checkpoint": best_ckpt,
    "dataset": "test_real",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "ablations": ablation_results,
}
with open(JSON_OUT, "w") as f:
    json.dump(output, f, indent=2)
print(f"\\nWritten: {JSON_OUT}")

# ── Upload to Hugging Face ───────────────────────────────────────────────
from huggingface_hub import HfApi
HfApi(token=HF_TOKEN).upload_file(
    path_or_fileobj=JSON_OUT,
    path_in_repo="results/proxy_ablation_results.json",
    repo_id=HF_REPO_ID,
    repo_type="dataset",
)
print(f"Uploaded -> {HF_REPO_ID}/results/proxy_ablation_results.json")

# ── Print full results ───────────────────────────────────────────────────
print("\\n" + "="*60)
print("PROXY ABLATION RESULTS")
print("="*60)
print(json.dumps(output, indent=2))
""")

    # CELL 46: 23_compute_cost (3a)
    add_markdown("""## 23_compute_cost (3a)""")
    add_code("""!pip install thop huggingface_hub --quiet

import os, json, torch
from src.model import SpatialMoESODNet
from src.dataset import get_dataloaders
import time
from thop import profile

RESULTS_DIR = "/kaggle/working/WXSOD_EvalResults"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("/kaggle/working/WXSOD_Checkpoints/best.pth", map_location=device, weights_only=False)

# Reconstruct the model from the checkpoint's own saved config (not hardcoded
# values) so the architecture always matches what was actually trained —
# mirrors the pattern used in 23_proxy_ablations and src/evaluate.py.
model_cfg = checkpoint.get("config", {}).get("model", {})
print(f"num_experts from checkpoint config: {model_cfg.get('num_experts', 'MISSING - defaulting to 6!')}")
assert "num_experts" in model_cfg, "checkpoint config missing num_experts — verify before trusting results"

model = SpatialMoESODNet(
    dim=256,
    num_experts=model_cfg.get("num_experts", 6),
    k=model_cfg.get("top_k", 2),
    window_size=model_cfg.get("window_size", 8),
    use_deep_supervision=model_cfg.get("deep_supervision", False),
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

train_loader, val_loader, test_sys_loader, test_real_loader = get_dataloaders(
    root_dir=valid_root,
    image_size=384, batch_size=1, num_workers=2
)

def save_json(name, data):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")

dummy = torch.randn(1, 3, 384, 384).to(device)
macs, params = profile(model, inputs=(dummy,), verbose=False)

with torch.no_grad():
    for _ in range(10):
        model(dummy)  # warmup
    if device == "cuda": torch.cuda.synchronize()
    start = time.time()
    N = 50
    for _ in range(N):
        model(dummy)
    if device == "cuda": torch.cuda.synchronize()
    elapsed = time.time() - start
fps = N / elapsed

compute_cost = {"params_M": round(params/1e6, 2), "macs_G": round(macs/1e9, 2), "fps": round(fps, 2)}
print(compute_cost)
save_json("compute_cost.json", compute_cost)
""")

    # CELL 48: 24_routing_entropy (3b)
    add_markdown("""## 24_routing_entropy (3b)""")
    add_code("""from collections import defaultdict

def collect_entropy_stats(model, dataloader, device, max_batches=None):
    model.eval()
    scale_entropies = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            images = batch['image'].to(device)
            _, moe_outputs = model(images)  # [out_4, out_8, out_16]
            for scale_name, out in zip(["scale_4", "scale_8", "scale_16"], moe_outputs):
                scale_entropies[scale_name].append(out.entropy.mean().item())
    return {k: (sum(v)/len(v)) for k, v in scale_entropies.items()}

sys_entropy = collect_entropy_stats(model, test_sys_loader, device)
real_entropy = collect_entropy_stats(model, test_real_loader, device)

entropy_comparison = {"synthetic": sys_entropy, "real": real_entropy}
print(entropy_comparison)
save_json("entropy_comparison.json", entropy_comparison)
""")

    # CELL 50: 25_qualitative_figure (3c)
    add_markdown("""## 25_qualitative_figure (3c)""")
    add_code("""# Peek at a few real-test samples and their weather labels to pick from
test_real_ds = test_real_loader.dataset

snow_idx, rain_or_fog_idx, light_idx = -1, -1, -1

# Simple logic to find examples (you can adjust this as needed based on actual stems/labels)
for idx in range(len(test_real_ds)):
    stem = test_real_ds.samples[idx][2]
    weather = stem.split('-')[0].lower() # Assuming format like snow-001 or similar
    
    # Just grab some generic ones if parsing is hard
    if 'snow' in weather or 'snow' in stem.lower():
        snow_idx = idx
    elif 'rain' in weather or 'rain' in stem.lower() or 'fog' in weather or 'fog' in stem.lower():
        rain_or_fog_idx = idx
    elif 'light' in weather or 'light' in stem.lower() or 'sun' in stem.lower():
        light_idx = idx
        
    if snow_idx != -1 and rain_or_fog_idx != -1 and light_idx != -1:
        break

# Fallback
if snow_idx == -1: snow_idx = 0
if rain_or_fog_idx == -1: rain_or_fog_idx = 1
if light_idx == -1: light_idx = 2

import matplotlib.pyplot as plt
import cv2
import numpy as np

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
            
            # De-normalize image for display
            img_disp = sample['image'].permute(1,2,0).cpu().numpy()
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
            
            axes[row,0].imshow(img_disp); axes[row,0].set_title(f"Input (idx {idx})")
            axes[row,1].imshow(gt, cmap='gray'); axes[row,1].set_title("GT")
            axes[row,2].imshow(pred, cmap='gray'); axes[row,2].set_title("Prediction")
            for ax in axes[row]: ax.axis('off')
    plt.tight_layout()
    save_path_full = os.path.join(RESULTS_DIR, save_path)
    plt.savefig(save_path_full, dpi=200)
    print(f"Saved to {save_path_full}")

make_qualitative_grid(model, test_real_ds, [snow_idx, rain_or_fog_idx, light_idx], device, "qualitative_grid.png")
""")

    # CELL 52: 26_sanity_ablation (3d)
    add_markdown("""## 26_sanity_ablation (3d)""")
    add_code("""from src.evaluate import evaluate as eval_fn

# Run evaluate with forced expert id
forced_results = eval_fn(
    model, 
    test_real_loader, 
    output_dir=os.path.join(RESULTS_DIR, "force_expert_out"),
    use_tta=False, 
    ablation_cfg={'scale': 8, 'expert_id': 0}
)
print("Forced Expert Results:")
print(forced_results)
save_json("force_expert_ablation.json", forced_results)
""")

    # CELL 54: 27_upload_results (3e)
    add_markdown("""## 27_upload_results (3e)""")
    add_code("""import os
import shutil
import glob
from huggingface_hub import login, HfApi, create_repo
from kaggle_secrets import UserSecretsClient

# Copy evaluation metrics and summaries to RESULTS_DIR before uploading
eval_dir = "/kaggle/working/spatial_moe_sod/evaluation"
if os.path.exists(eval_dir):
    for ext in ["*.txt", "*.json"]:
        for filepath in glob.glob(f"{eval_dir}/**/{ext}", recursive=True):
            dest = os.path.join(RESULTS_DIR, os.path.basename(filepath))
            shutil.copy(filepath, dest)
            print(f"Copied {os.path.basename(filepath)} to RESULTS_DIR")

try:
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    login(token=hf_token)

    # Use the single central repository defined earlier
    REPO_ID = os.environ.get("HF_REPO_ID", "Avi2006/spatial-moe-results")

    create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=True)

    api = HfApi()
    api.upload_folder(
        folder_path=RESULTS_DIR,
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo="evaluation_results"
    )
    print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}/tree/main/evaluation_results")
except Exception as e:
    print(f"Failed to upload to Hugging Face: {e}")
""")

    # CELL 56: 24_diagnostics_and_upload
    add_markdown("""## 24_diagnostics_and_upload
Run MoE Diagnostics, Padding Audit, and HF Upload""")
    add_code("""import subprocess
import os
import sys

print("=== Starting MoE Diagnostics, Padding Audit, and HF Upload ===\\n")

# 1. Run Padding Audit
print("[1/3] Running Padding Audit...")
subprocess.run([
    "python", "-m", "src.diagnose_padding", 
    "--data_dir", valid_root
], cwd=PROJECT_ROOT, check=True)

# 2. Run MoE Diagnostics
print("\\n[2/3] Running MoE Diagnostics (Routing & Expert Similarity)...")
best_ckpt = os.path.join(CHECKPOINT_ROOT, "best.pth")
if not os.path.exists(best_ckpt):
    best_ckpt = os.path.join(CHECKPOINT_ROOT, "latest.pth")
subprocess.run([
    "python", "-m", "src.run_moe_diagnostics", 
    "--checkpoint", best_ckpt,
    "--data_dir", valid_root
], cwd=PROJECT_ROOT, check=True)

# 3. Upload to HF
print("\\n[3/3] Uploading Diagnostics to Hugging Face...")
subprocess.run([
    "python", "-m", "src.upload_diagnostics"
], cwd=PROJECT_ROOT, check=True)

print("\\n=== All diagnostic tasks completed successfully ===")
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
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("moe-for-sod-final.ipynb", "w") as f:
        json.dump(notebook, f, indent=2)
    print("moe-for-sod-final.ipynb generated successfully.")

if __name__ == '__main__':
    create_notebook()
