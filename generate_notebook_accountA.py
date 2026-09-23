"""account A — 2-expert, top_k=1 run.

Regenerates ``moe-of-sod__accountA__v_e8_repro_best.ipynb`` and is the editable source of truth for it — change a
cell here and re-run this script to rebuild the notebook. The generate_notebook_accountB.py
generator holds the same cells for the other account (the two differ only in
ACTIVE_CONFIG_PATH), so a shared cell change has to be applied to
both files.

Usage:
    python generate_notebook_accountA.py
"""
import json

DEFAULT_CONFIG_PATH = "experiments/v_e8_repro_best.json"
OUTPUT_NOTEBOOK = "moe-of-sod__accountA__v_e8_repro_best.ipynb"


def create_notebook(output_notebook: str = OUTPUT_NOTEBOOK,
                    config_path: str = DEFAULT_CONFIG_PATH) -> None:
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
RUN_MODE = "TRAIN"
# Pre-run checks (architecture agreement + a 100-image end-to-end run).
# Set False to skip the "## 11_preflight_checks" cell entirely.
RUN_PREFLIGHT = True
# Training refuses to touch a run whose checkpoints already exist, so a
# name can never silently overwrite finished work. Set True to wipe and redo.
ALLOW_OVERWRITE = False
# Evaluation only: "none" or "hflip" (flip-average TTA). The model is trained with
# HorizontalFlip and the padding is centre + reflect, so the flipped forward pass
# is in-distribution. Every model in a comparison table must use the same setting;
# hflip results land under their own subfolder, so both can coexist.
EVAL_TTA = "hflip"
ACTIVE_CONFIG_PATH = "experiments/v_e8_repro_best.json"
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
# Kaggle logs: unbuffered so lines appear as they happen, and no tqdm
# redraws (each redraw becomes its own line in the log pane).
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TQDM_DISABLE"] = "1"

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
        r"""## 02_environment_info
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
        r"""## 07_backbone_check
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
        r"""## 08_static_import_check
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
        r"""## 09_dataset_alignment_check
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
        r"""## 10_model_shape_check
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
        r"""## 11_preflight_checks

Runs before training and stops it if anything is wrong.  About 8 minutes, two stages:

1. **Static check** -- rebuilds the model from the active config on CPU and asserts that
   every architecture field actually reached the model (expert count, top-k, gate mode,
   MoE type).  This is the bug class where a config field is validated but never
   forwarded, so the run silently trains a different model than the config describes.

2. **End-to-end mini run** -- 100 images under 2-GPU DDP: training steps, the epoch-end
   validation, the routing diagnostics, a checkpoint write, and a resume from that
   checkpoint.  Those epoch-boundary paths are where an unattended run would otherwise
   die at the end of epoch 1, four hours in.

Everything is written to ``WXSOD_Preflight/<experiment_id>/``, never to the real
checkpoint directory, and no Hugging Face upload happens.  Set ``RUN_PREFLIGHT = False``
in the config cell to skip this cell.
""")
    add_code(
        r"""if RUN_MODE == "TRAIN" and RUN_PREFLIGHT:
    import json
    import os
    import subprocess
    import sys

    PROJECT_ROOT = "/kaggle/working/spatial_moe_sod"
    CONFIG_SRC = os.path.join(PROJECT_ROOT, ACTIVE_CONFIG_PATH)

    # --- 1. the config must describe the model that actually gets built ---------
    sys.path.insert(0, PROJECT_ROOT)
    from src.config import ExperimentConfig
    from src.experiment import generate_experiment_id
    from src.model import SpatialMoESODNet, assert_model_matches_config

    experiment = ExperimentConfig.load(CONFIG_SRC)
    probe = SpatialMoESODNet(
        use_deep_supervision=experiment.model.deep_supervision,
        num_experts=experiment.model.num_experts,
        k=experiment.model.top_k,
        window_size=experiment.model.window_size,
        router_noise_enabled=experiment.model.router_noise_enabled,
        router_noise_scale=experiment.model.router_noise_scale,
        router_noise_min_std=experiment.model.router_noise_min_std,
        moe_16_mode=experiment.model.moe_16_mode,
        gate_mode=experiment.model.gate_mode,
        moe_type=experiment.model.moe_type,
        pretrained_backbone=False,  # weights are irrelevant to an architecture check
    )
    assert_model_matches_config(probe, experiment)
    del probe
    m = experiment.model
    print(f"\u2713 config and model agree: {m.num_experts} experts, k={m.top_k}, "
          f"gate_mode={m.gate_mode}, moe_type={m.moe_type}, moe_16_mode={m.moe_16_mode}")

    # --- 2. one real epoch on a small subset, then a real resume ----------------
    with open(CONFIG_SRC) as f:
        tiny = json.load(f)
    assert {"data", "train"} <= set(tiny), f"unexpected config layout: {list(tiny)}"
    # epochs=2 so the resume below genuinely continues the schedule instead of
    # resuming into a run that has already finished.
    tiny["data"] = dict(tiny["data"], max_samples=100)
    # epochs=2 so the resume below continues the schedule rather than resuming
    # into a finished run; checkpoint_every_n_steps=1 because the mini epoch is
    # ~3 optimizer steps, far short of the production cadence of 200, so
    # otherwise no checkpoint would exist to resume from.
    tiny["train"] = dict(tiny["train"], epochs=2, checkpoint_every_n_steps=1)
    TINY_CONFIG = "/kaggle/working/preflight_config.json"
    with open(TINY_CONFIG, "w") as f:
        json.dump(tiny, f, indent=2)

    # The same experiment ID the trainer derives, so the assertions below read
    # exactly the files it writes.  Preflight artifacts never touch the real
    # checkpoint directory.
    preflight_dir = os.path.join(PREFLIGHT_ROOT, generate_experiment_id(experiment))
    checkpoint = os.path.join(preflight_dir, "latest.pth")
    common = ["torchrun", "--nproc_per_node=2", "-m", "src.train_ddp",
              "--config", TINY_CONFIG, "--preflight",
              # Without an explicit cap, --preflight defaults to 5 optimizer steps.
              # That would end the epoch early and skip exactly the epoch-boundary
              # paths (validation, diagnostics, checkpoint) this run exists to test.
              "--max_optimizer_steps", "100000"]

    print("\n" + "=" * 70)
    print("PREFLIGHT 1/2  100 images: train -> validate -> diagnostics -> checkpoint")
    print("=" * 70)
    subprocess.run(common + ["--max_epochs", "1"], cwd=PROJECT_ROOT, check=True)
    assert os.path.exists(checkpoint), (
        f"epoch finished but wrote no checkpoint: {checkpoint} "
        f"(checkpoint_every_n_steps={tiny['train']['checkpoint_every_n_steps']})"
    )
    written_at = os.path.getmtime(checkpoint)

    print("\n" + "=" * 70)
    print("PREFLIGHT 2/2  resume from the checkpoint that run wrote")
    print("=" * 70)
    subprocess.run(common + ["--resume", "latest", "--max_epochs", "2"],
                   cwd=PROJECT_ROOT, check=True)
    # A resume that silently does nothing also exits 0, so require new work.
    assert os.path.getmtime(checkpoint) > written_at, (
        "resume exited cleanly but never rewrote the checkpoint -- it did no work"
    )

    print("\n\u2713 preflight passed -- the next cell starts the real run")
""")
    add_markdown(
        r"""## 12_train
""")
    add_code(
        r"""if RUN_MODE == "TRAIN":
    import subprocess
    import os

    PROJECT_ROOT = "/kaggle/working/spatial_moe_sod"

    train_cmd = [
        "torchrun", "--nproc_per_node=2", "-m", "src.train_ddp",
        "--config", ACTIVE_CONFIG_PATH,
    ]
    if ALLOW_OVERWRITE:
        train_cmd.append("--overwrite")
    subprocess.run(train_cmd, cwd=PROJECT_ROOT, check=True)
""")
    add_markdown(
        r"""## 13_resume
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
        r"""## 14_evaluate
""")
    add_code(
        r"""import os, sys, subprocess
from src.config import ExperimentConfig
from src.experiment import generate_experiment_id
from src.hf_sync import pull_checkpoint

if RUN_MODE == "EVALUATE":
    print("Evaluating Best Checkpoint...")

    # These two paths are shared by every config within a session, so a second
    # evaluation would silently mix its results into this run's upload.
    for _shared in ("/kaggle/working/WXSOD_EvalResults", "/kaggle/working/analysis_results"):
        if os.path.isdir(_shared) and os.listdir(_shared):
            raise RuntimeError(
                f"{_shared} already holds results from another config -- delete it or "
                "restart the session before evaluating a different config."
            )
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
            "--tta", EVAL_TTA,
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Line-based, not byte-based: one write per line instead of per character.
    for line in process.stdout:
        sys.stdout.write(line)

    retcode = process.wait()
    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, process.args)

    print("\n--- Files in eval_out_dir ---")
    for f in os.listdir(eval_out_dir):
        print(f)
""")
    add_markdown(
        r"""## 15_load_model_for_analysis
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
        r"""## 16_proxy_ablations
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
        r"""## 17_routing_entropy
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
        r"""## 18_qualitative_figure
""")
    add_code(
        r"""if RUN_MODE == "EVALUATE":
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    from src.dataset import get_weather_type
    from src.evaluate import reverse_geometry

    test_real_ds = test_real_loader.dataset

    # One image per condition, using the canonical parser so each row really is
    # the weather it is labelled with.
    wanted = ("snow", "fog", "light")
    picked = {}
    for idx in range(len(test_real_ds)):
        weather = get_weather_type(test_real_ds.samples[idx][2])
        if weather in wanted and weather not in picked:
            picked[weather] = idx
        if len(picked) == len(wanted):
            break

    def make_qualitative_grid(model, dataset, picks, device, save_path):
        rows = list(picks.items())
        fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
        model.eval()
        with torch.no_grad():
            for row, (weather, idx) in enumerate(rows):
                sample = dataset[idx]
                meta = {k: v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in sample['meta'].items()}
                out, _ = model(sample['image'].unsqueeze(0).to(device))
                pred = torch.sigmoid(out.saliency_logits)

                # The prediction lives in the padded/resized model frame.  Undo that
                # exactly as the evaluator does, so all three columns are in ORIGINAL
                # image coordinates and are directly comparable.
                pred_orig = reverse_geometry(pred[:1], meta)
                gt_orig = cv2.imread(str(meta['gt_path']), cv2.IMREAD_GRAYSCALE)
                img_orig = cv2.cvtColor(cv2.imread(str(dataset.samples[idx][0])),
                                        cv2.COLOR_BGR2RGB)

                panels = (
                    (img_orig, f"Input ({weather})", {}),
                    (gt_orig, "Ground truth", {"cmap": "gray"}),
                    (pred_orig, "Prediction", {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}),
                )
                for col, (panel, title, imshow_kwargs) in enumerate(panels):
                    axes[row, col].imshow(panel, **imshow_kwargs)
                    axes[row, col].set_title(title)
                    axes[row, col].axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"Saved to {save_path}")

    make_qualitative_grid(model, test_real_ds, picked, device,
                          os.path.join(RESULTS_ROOT, 'qualitative', 'qualitative_grid.png'))
""")
    add_markdown(
        r"""## 19_diagnostics
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
        r"""## 20_compute_cost (optional, needs `pip install thop`)
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
        r"""## 21_forced_expert_check (optional)
""")
    add_code(
        r"""# # Targets stride/8 (scale=8), the stage with confirmed DEAD_EXPERT [0, 5], as opposed to the scale=4 check in "## 16_proxy_ablations".
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
        r"""## 22_upload_results""")
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
    # Copy the "## 14_evaluate" outputs into RESULTS_ROOT so everything is in one tree
    eval_dir = "/kaggle/working/WXSOD_EvalResults"
    if os.path.exists(eval_dir):
        for ext in ["*.txt", "*.json"]:
            for filepath in glob.glob(f"{eval_dir}/**/{ext}", recursive=True):
                # Keep the dataset/tta subdirectory in the path so a hflip run
                # never overwrites the plain one.
                rel = os.path.relpath(filepath, eval_dir)
                dest = os.path.join(RESULTS_ROOT, 'eval_results', rel)
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
    add_markdown(
        r"""## 23_optional_pre_run_gates

The notebook used to carry these as commented-out cells; they duplicate
`src.smoke_test`, so they are kept here as commands instead.  Run whichever you
need -- they fail fast, which is cheaper than discovering the problem four hours
into a run.

```bash
torchrun --nproc_per_node=2 -m src.smoke_test --mode empty_batch --data_root <root> --result_file /tmp/eb.json
torchrun --nproc_per_node=2 -m src.smoke_test --mode ddp         --data_root <root> --result_file /tmp/ddp.json
torchrun --nproc_per_node=2 -m src.smoke_test --mode memory      --data_root <root> --result_file /tmp/mem.json
torchrun --nproc_per_node=2 -m src.smoke_test --mode resume_a    --data_root <root> --result_file /tmp/r1.json
torchrun --nproc_per_node=2 -m src.smoke_test --mode resume_b    --data_root <root> --result_file /tmp/r2.json
```

Other one-off commands that were in the notebook as dead cells:

```bash
# 5-step training sanity check on a new config
torchrun --nproc_per_node=2 -m src.train_ddp --config <cfg> --preflight --max_optimizer_steps 5
# manual checkpoint push (equivalent of the old 00_hf_sync cell)
python -m src.hf_sync push-checkpoint <file> --name <remote-name>
```
""")

    # The config path is the only thing that distinguishes the arms, so swap it into
    # whichever cell holds it instead of duplicating this whole file per variant.
    _swapped = 0
    for _cell in cells:
        _joined = "".join(_cell["source"])
        if DEFAULT_CONFIG_PATH in _joined:
            _cell["source"] = _joined.replace(
                DEFAULT_CONFIG_PATH, config_path).splitlines(keepends=True)
            _swapped += 1
    assert _swapped == 1, f"config path appears in {_swapped} cells, expected 1"

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

    with open(output_notebook, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"{output_notebook} generated successfully ({len(cells)} cells).")


if __name__ == "__main__":
    create_notebook()
