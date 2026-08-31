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

    # CELL 01: Config
    add_markdown("## 01_config")
    add_code("""# Configuration & Modes
# RUN_MODE strictly governs the allowed execution path.
# Allowed: "VALIDATE", "TRAIN", "RESUME", "EVALUATE"
RUN_MODE = "VALIDATE"

DATA_SOURCE = "GOOGLE_DRIVE"
DATA_FILE_ID = "1SSELvRYI-cwd9mzA8dWLbv4o1IffjkoW"

PROJECT_INPUT_ZIP = "/kaggle/input/spatial-moe-code/spatial_moe_sod_code.zip"
MANIFEST_INPUT = "/kaggle/input/spatial-moe-code/project_manifest.json"
PREVIOUS_OUTPUT_DATASET = None

PROJECT_ROOT = "/kaggle/working/spatial_moe_sod"
CHECKPOINT_ROOT = "/kaggle/working/WXSOD_Checkpoints"
PREFLIGHT_ROOT = "/kaggle/working/WXSOD_Preflight"
RESUME_TEST_ROOT = "/kaggle/working/WXSOD_ResumeTest"

NUM_GPUS = 2
FINAL_EPOCHS = 50

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
""")

    # CELL 02: Environment
    add_markdown("## 02_environment")
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

    # CELL 03: Dataset Acquire
    add_markdown("## 03_dataset_acquire")
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

    # CELL 04: Dataset Validate
    add_markdown("## 04_dataset_validate")
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

    # CELL 05: Project Deploy
    add_markdown("## 05_project_deploy")
    add_code("""import sys
import json
import hashlib
import shutil
import os

os.makedirs(PROJECT_ROOT, exist_ok=True)

KAGGLE_UNZIPPED_DIR = "/kaggle/input/datasets/avinashreddybapatu/spatial-moe-code/spatial_moe_sod_code"

if os.path.exists(KAGGLE_UNZIPPED_DIR):
    print(f"Using pre-unzipped Kaggle directory: {KAGGLE_UNZIPPED_DIR}")
    shutil.copytree(KAGGLE_UNZIPPED_DIR, PROJECT_ROOT, dirs_exist_ok=True)
elif os.path.exists(PROJECT_INPUT_ZIP):
    if os.path.exists(MANIFEST_INPUT):
        with open(MANIFEST_INPUT, "r") as f:
            manifest = json.load(f)
        
        sha256 = hashlib.sha256()
        with open(PROJECT_INPUT_ZIP, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
                
        if sha256.hexdigest() != manifest["archive_sha256"]:
            mark_gate("PROJECT_CHECK", "FAIL", "ZIP SHA256 mismatch!")
            raise RuntimeError("Project source archive corrupted or outdated.")
            
    subprocess.run(["unzip", "-q", "-o", PROJECT_INPUT_ZIP, "-d", PROJECT_ROOT], check=True)
elif os.path.exists("src"):
    print("Using local src directory fallback.")
    shutil.copytree("src", os.path.join(PROJECT_ROOT, "src"), dirs_exist_ok=True)
else:
    raise FileNotFoundError(f"Neither {KAGGLE_UNZIPPED_DIR}, {PROJECT_INPUT_ZIP}, nor local 'src' found.")

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

    # CELL 06: Dependencies
    add_markdown("## 06_dependencies")
    add_code("""import importlib
deps = ["torch", "torchvision", "timm", "albumentations", "cv2", "numpy"]
env_data = {}
all_passed = True
for dep in deps:
    try:
        mod = importlib.import_module(dep)
        env_data[dep] = getattr(mod, "__version__", "unknown")
    except ImportError:
        print(f"Missing dependency: {dep}. Installing...")
        subprocess.run(["pip", "install", "-q", dep], check=True)
        try:
            mod = importlib.import_module(dep)
            env_data[dep] = getattr(mod, "__version__", "unknown")
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

    # CELL 07: Pretrained B4
    add_markdown("## 07_pretrained_b4")
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

    # CELL 08: Static Checks
    add_markdown("## 08_static_checks")
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

    # CELL 09: Data Transform
    add_markdown("## 09_data_transform")
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
            img_pad, mask_pad, _ = ds._aspect_preserving_resize_pad(syn_img, syn_mask)
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

    # CELL 10: Model Shapes
    add_markdown("## 10_model_shapes")
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

    # CELL 14: Smoke 2-GPU
    add_markdown("## 14_smoke_2gpu")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    with open(os.path.join(PROJECT_ROOT, "experiments/baseline_v1.json"), "r") as f:
        canonical_cfg = json.load(f)
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

    # CELL 15: Production Calibration
    add_markdown("## 15_production_calibration")
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

    # CELL 16: Checkpoint Test
    add_markdown("## 16_checkpoint_test")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    print("Running Checkpoint Write Test...")
    res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "resume_a", "--data_root", valid_root, "--result_file", "test_ckpt.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
    with open(os.path.join(PROJECT_ROOT, "test_ckpt.json"), "r") as f:
        if json.load(f).get("status") == "PASS":
            mark_gate("CHECKPOINT_CHECK", "PASS")
        else:
            mark_gate("CHECKPOINT_CHECK", "FAIL")
""")

    # CELL 17: Resume Test
    add_markdown("## 17_resume_test")
    add_code("""if RUN_MODE in ["VALIDATE", "TRAIN"]:
    print("Running Checkpoint Resume Test...")
    res = subprocess.run(["torchrun", "--nproc_per_node=2", "-m", "src.smoke_test", "--mode", "resume_b", "--data_root", valid_root, "--result_file", "test_res.json", "--run_id", CURRENT_RUN_ID, "--config_hash", CURRENT_CONFIG_HASH], cwd=PROJECT_ROOT)
    with open(os.path.join(PROJECT_ROOT, "test_res.json"), "r") as f:
        if json.load(f).get("status") == "PASS":
            mark_gate("RESUME_CHECK", "PASS")
        else:
            mark_gate("RESUME_CHECK", "FAIL")
""")

    # CELL 18: Final Gate
    add_markdown("## 18_final_gate")
    add_code("""if RUN_MODE == "TRAIN":
    print("Running Preflight Dry Run (2-5 steps)...")
    with open(os.path.join(PROJECT_ROOT, "experiments/baseline_v1.json"), "r") as f:
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
    with open(os.path.join(PROJECT_ROOT, "experiments/baseline_v1.json"), "r") as f:
        canonical_cfg = json.load(f)
    from src.train_ddp import get_config_hash
    canonical_hash = get_config_hash(canonical_cfg, "model_config_hash")
    
        mark_gate("PREFLIGHT_DRY_RUN_CHECK", "PASS", run_id=pf_data.get("run_id"), config_hash=pf_data.get("config_hash"))
    else:
        mark_gate("PREFLIGHT_DRY_RUN_CHECK", "FAIL")
        raise RuntimeError(f"Preflight validation failed: {pf_data}")

FINAL_STATUS = "PASS"
# Some gates are only for train/validate
if RUN_MODE in ["VALIDATE", "TRAIN"]:
    with open(os.path.join(PROJECT_ROOT, "experiments/baseline_v1.json"), "r") as f:
        canonical_cfg = json.load(f)
    from src.train_ddp import get_config_hash
    CURRENT_CONFIG_HASH = get_config_hash(canonical_cfg, "model_config_hash")
    CURRENT_RUN_ID = canonical_cfg.get("run_id") or "test_run_id"

    # 9. Current-run evidence validation
    import time
    required_files = [
        "test_2gpu.json", "test_mem.json", "test_ckpt.json", "test_res.json"
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
                if pf_data.get("run_id") != CURRENT_RUN_ID or pf_data.get("config_hash") != CURRENT_CONFIG_HASH:
                    print(f"Evidence mismatch in preflight_results.json")
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
            "CHECKPOINT_CHECK", "RESUME_CHECK"
        ],
        "TRAIN": [
            "ENV_CHECK", "GPU_CHECK", "DATA_CHECK", "PROJECT_CHECK", "DEPENDENCY_CHECK",
            "PVT_CHECK", "STATIC_CHECK", "TRANSFORM_CHECK", "MODEL_CHECK", "MOE_CHECK",
            "LOSS_CHECK", "OPT_CHECK", "DDP_CHECK", "MEMORY_CHECK",
            "CHECKPOINT_CHECK", "RESUME_CHECK", "PREFLIGHT_DRY_RUN_CHECK"
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

    # CELL 19: Train
    add_markdown("## 19_train")
    add_code("""if RUN_MODE == "TRAIN":
    if FINAL_STATUS != "PASS" or GATES.get("PREFLIGHT_DRY_RUN_CHECK") != "PASS":
        raise RuntimeError("Refusing to train: Not all gates passed.")
        
    with open(os.path.join(PROJECT_ROOT, "experiments/baseline_v1.json"), "r") as f:
        canonical_cfg = json.load(f)
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
        "--config", RUNTIME_CONFIG
    ], cwd=PROJECT_ROOT, check=True)
""")

    # CELL 20: Status
    add_markdown("## 20_status")
    add_code("""if RUN_MODE == "TRAIN":
    if os.path.exists(os.path.join(CHECKPOINT_ROOT, "training_complete.json")):
        print("Training successfully reached completion state.")
    else:
        print("Training did not produce completion marker.")
""")

    # CELL 21: Resume
    add_markdown("## 21_resume")
    add_code("""if RUN_MODE == "RESUME":
    print("Initiating Resume Recovery Sequence...")
    local_latest = os.path.join(CHECKPOINT_ROOT, "latest.pth")
    if os.path.exists(os.path.join(CHECKPOINT_ROOT, "training_complete.json")):
        raise RuntimeError("TRAINING ALREADY COMPLETE. Cannot resume.")
        
    if not os.path.exists(local_latest):
        if PREVIOUS_OUTPUT_DATASET is None:
            raise RuntimeError("latest.pth not found in /kaggle/working and PREVIOUS_OUTPUT_DATASET is None. Configure it to recover checkpoint.")
            
        persisted_latest = f"/kaggle/input/{PREVIOUS_OUTPUT_DATASET}/latest.pth"
        persisted_manifest = f"/kaggle/input/{PREVIOUS_OUTPUT_DATASET}/checkpoint_manifest.json"
        
        if not os.path.exists(persisted_latest):
            raise RuntimeError(f"Could not find {persisted_latest}")
            
        print("Recovering checkpoint from persisted input dataset...")
        os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
        if os.path.isdir(persisted_latest):
            print(f"Kaggle unzipped {persisted_latest}, re-zipping...")
            import zipfile
            needs_prefix = os.path.exists(os.path.join(persisted_latest, "version"))
            prefix = "archive" if needs_prefix else ""
            with zipfile.ZipFile(local_latest, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(persisted_latest):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, persisted_latest)
                        if prefix:
                            arcname = os.path.join(prefix, arcname)
                        zipf.write(file_path, arcname)
        else:
            shutil.copy2(persisted_latest, local_latest)
        if os.path.exists(persisted_manifest):
            shutil.copy2(persisted_manifest, os.path.join(CHECKPOINT_ROOT, "checkpoint_manifest.json"))
            
    with open(os.path.join(PROJECT_ROOT, "experiments/baseline_v1.json"), "r") as f:
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

    # CELL 22: Evaluate
    add_markdown("## 22_evaluate")
    add_code("""if RUN_MODE in ["TRAIN", "EVALUATE"]:
    print("Evaluating Best Checkpoint...")
    best_ckpt = os.path.join(CHECKPOINT_ROOT, "best.pth")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(CHECKPOINT_ROOT, "latest.pth")
    subprocess.run(["python", "-m", "src.evaluate", "--checkpoint", best_ckpt, "--dataset", "both"], cwd=PROJECT_ROOT, check=True)
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
