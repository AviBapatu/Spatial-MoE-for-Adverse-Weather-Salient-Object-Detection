"""Generate the legacy-checkpoint evaluation notebook.

Regenerates ``moe-of-sod__legacy_eval.ipynb`` and is the editable source of truth for it.

One job the run notebooks cannot do: evaluate a checkpoint that is not tied to an experiment
ID, specifically the legacy 8-expert model behind the 0.0168 result. The run notebooks resolve
their checkpoint from the active config's experiment ID, so a legacy file has to arrive out of
band - from Google Drive by file ID, using the same gdown pattern the dataset cell uses.

Like the run notebooks it uses both T4s and uploads its results to the Hub. ``src.evaluate`` is
single-process (there is no data-parallel evaluation), so the two GPUs are used by evaluating
the two test splits concurrently, one per card.

Usage:
    python generate_notebook_legacy_eval.py
"""
import json

OUTPUT_NOTEBOOK = "moe-of-sod__legacy_eval.ipynb"

LEGACY_DRIVE_ID = "1GblynkVLciAy2OATB00naeEEgWnDOH9P"
DATA_FILE_ID = "1SSELvRYI-cwd9mzA8dWLbv4o1IffjkoW"


def create_notebook() -> None:
    """Build the notebook cell list and write it to ``OUTPUT_NOTEBOOK``."""
    cells = []

    def add_markdown(text: str) -> None:
        """Append a markdown cell holding *text*."""
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": text.splitlines(keepends=True)})

    def add_code(text: str) -> None:
        """Append a code cell holding *text*."""
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": text.splitlines(keepends=True)})

    add_markdown(f"""# Legacy checkpoint evaluation

Evaluates the legacy 8-expert checkpoint - the model behind the 0.0168 / 0.9151 result - with the
**current** evaluation code. This is a diagnostic, not a result:

- reads **~0.0168** -> the current pipeline reproduces the old number, so the gap to the new
  runs' ~0.0195 is a training-side difference;
- reads **~0.019** -> the evaluation protocol changed, and the current numbers are the
  consistent ones.

The checkpoint's Google Drive file ID is already set (`{LEGACY_DRIVE_ID}`). The file must be
shared as "anyone with the link", or gdown cannot fetch it.

Unlike the run notebooks there is no experiment ID to key on, so results are uploaded under
`analysis_results/legacy_8expert_best/`.
""")

    add_code(r'''import os
import json
import hashlib
import shutil
import subprocess
import sys

# The legacy checkpoint is not tied to an experiment ID, so it arrives from Drive.
# For a link like  https://drive.google.com/file/d/<ID>/view?usp=sharing  this is <ID>.
LEGACY_DRIVE_ID = "1GblynkVLciAy2OATB00naeEEgWnDOH9P"

# Uploaded under this label: there is no config-derived experiment ID for a legacy file.
LEGACY_LABEL = "legacy_8expert_best"

# Both TTA settings, evaluated concurrently across the two GPUs.
TTA_MODES = ["none", "hflip"]

from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
HF_REPO_ID = "Avi2006/spatial-moe-results"
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_REPO_ID"] = HF_REPO_ID

PROJECT_ROOT = "/kaggle/working/spatial_moe_sod"
CHECKPOINT_ROOT = "/kaggle/working/WXSOD_Checkpoints"
LEGACY_PATH = os.path.join(CHECKPOINT_ROOT, "legacy_best.pth")
LEGACY_OUT = os.path.join("/kaggle/working/legacy_eval", LEGACY_LABEL)
DATA_FILE_ID = "1SSELvRYI-cwd9mzA8dWLbv4o1IffjkoW"
EXTRACT_PATH = "/kaggle/working/WXSDO_data"

NUM_GPUS = 2
os.environ["CHECKPOINT_ROOT"] = CHECKPOINT_ROOT
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

assert LEGACY_DRIVE_ID != "PASTE_THE_GOOGLE_DRIVE_FILE_ID_HERE", (
    "Paste the Google Drive file ID of the legacy checkpoint into LEGACY_DRIVE_ID."
)

print("Configuration")
print("  legacy checkpoint ->", LEGACY_PATH)
print("  upload label      ->", LEGACY_LABEL)
print("  output root       ->", LEGACY_OUT)
print("  TTA settings      ->", TTA_MODES)
''')

    add_markdown(r"""## 02_environment

Two GPUs are required: the two test splits are evaluated concurrently, one per card.
""")

    add_code(r'''import torch

num_gpus = torch.cuda.device_count()
print("GPU count:", num_gpus)
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
assert num_gpus >= NUM_GPUS, f"Expected {NUM_GPUS} GPUs, found {num_gpus}"
''')

    add_markdown(r"""## 03_dataset_acquire

Same acquisition path as the run notebooks: a Google Drive archive extracted locally, then the
real WXSOD root resolved by structure rather than by a hard-coded path.
""")

    add_code(r'''def is_dataset_valid(path):
    def check_root(r):
        reqs = ["train_sys/input", "train_sys/gt", "test_sys/input", "test_sys/gt",
                "test_real/input", "test_real/gt"]
        return all(os.path.isdir(os.path.join(r, req)) for req in reqs)

    if check_root(path):
        return path
    for root, dirs, _ in os.walk(path):
        if check_root(root):
            return root
    return None


valid_root = is_dataset_valid(EXTRACT_PATH)
if valid_root is None:
    print("Dataset not found locally, downloading from Google Drive...")
    ZIP_PATH = "/kaggle/working/dataset.zip"
    if not os.path.exists(ZIP_PATH):
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
        subprocess.run(["gdown", DATA_FILE_ID, "-O", ZIP_PATH], check=True)
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", ZIP_PATH, "-d", EXTRACT_PATH], check=True)
    valid_root = is_dataset_valid(EXTRACT_PATH)

assert valid_root is not None, "Dataset extraction completed but no valid WXSOD root was found."
print("Dataset root:", valid_root)
''')

    add_markdown(r"""## 04_project_deploy

The code comes from the Hub as a verified archive - the same route the run notebooks use, so
this evaluates the current implementation rather than whatever is already on the box.
""")

    add_code(r'''from huggingface_hub import hf_hub_download

os.makedirs(PROJECT_ROOT, exist_ok=True)
print(f"Downloading code from {os.environ['HF_REPO_ID']} ...")

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

sha256 = hashlib.sha256()
with open(zip_path, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
        sha256.update(chunk)

assert sha256.hexdigest() == manifest["archive_sha256"], (
    "Project source archive corrupted or stale - repackage and re-upload."
)

subprocess.run(["unzip", "-q", "-o", zip_path, "-d", PROJECT_ROOT], check=True)

required = ["src/train_ddp.py", "src/evaluate.py", "src/model.py", "src/experiment.py"]
for f in required:
    assert os.path.exists(os.path.join(PROJECT_ROOT, f)), f"Payload corrupted: missing {f}"

sys.path.insert(0, PROJECT_ROOT)
print("Project source deployed to:", PROJECT_ROOT)
''')

    add_markdown(r"""## 05_dependencies

`src/evaluate.py` needs torch, torchvision, timm, albumentations, OpenCV, numpy and
`py_sod_metrics`. Missing ones are installed rather than assumed.
""")

    add_code(r'''import importlib

deps = {
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "albumentations": "albumentations",
    "opencv-python-headless": "cv2",
    "numpy": "numpy",
    "pysodmetrics": "py_sod_metrics",
}
for pip_name, mod_name in deps.items():
    try:
        mod = importlib.import_module(mod_name)
        print(f"  {pip_name:28s} {getattr(mod, '__version__', 'unknown')}")
    except ImportError:
        print(f"  {pip_name:28s} installing ...")
        subprocess.run(["pip", "install", "-q", pip_name], check=True)
print("Dependencies resolved.")
''')

    add_markdown(r"""## 06_legacy_checkpoint

The legacy checkpoint is not tied to an experiment ID, so the run notebooks cannot resolve it.
It comes from Drive and is checked before use.
""")

    add_code(r'''EXPECTED_SHA256 = "8b7cac641afb1ff409c6e3db7caf2b826455bd1077158fd227b822e50d934ce0"

if not os.path.exists(LEGACY_PATH):
    print("Downloading the legacy checkpoint from Google Drive ...")
    subprocess.run(["pip", "install", "-q", "gdown"], check=True)
    subprocess.run(["gdown", LEGACY_DRIVE_ID, "-O", LEGACY_PATH], check=True)

assert os.path.exists(LEGACY_PATH), f"Legacy checkpoint missing: {LEGACY_PATH}"

h = hashlib.sha256()
with open(LEGACY_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
actual = h.hexdigest()
size_mb = os.path.getsize(LEGACY_PATH) / 1024 ** 2
print(f"  {LEGACY_PATH}  ({size_mb:.0f} MB)")
print(f"  sha256 {actual}")
if actual != EXPECTED_SHA256:
    print("  WARNING: sha256 does not match the value recorded on the Hub.")
    print("           Expected 8b7cac641afb1ff409c6e3db7caf2b826455bd1077158fd227b822e50d934ce0")
    print("           Continuing, but record which file was actually scored.")
else:
    print("  sha256 matches the value recorded on the Hub.")
''')

    add_markdown(r"""## 07_identity

Prints the parameter-name fingerprint of the architecture this checkpoint creates. The 0.0168
evaluation recorded `2cd252ad3a3581e1c65961b828857d70`, so a different value means the file is
not provably that save - worth knowing when reading the metrics below.
""")

    add_code(r'''import torch
from src.evaluate import verify_parameter_consistency
from src.model import SpatialMoESODNet

ckpt = torch.load(LEGACY_PATH, map_location="cpu", weights_only=False)
print(f"  epoch={ckpt.get('epoch')}  global_step={ckpt.get('global_step')}  "
      f"best_metric={ckpt.get('best_metric')}")

mc = (ckpt.get("config") or {}).get("model") or {}
print(f"  config: experts={mc.get('num_experts')} k={mc.get('top_k')} "
      f"gate={mc.get('gate_mode', 'renormalized')} moe_type={mc.get('moe_type', 'sparse')} "
      f"deep_supervision={mc.get('deep_supervision')}")

probe = SpatialMoESODNet(
    dim=mc.get("working_dim", 256),
    num_experts=mc.get("num_experts", 8),
    k=mc.get("top_k", 2),
    gate_mode=mc.get("gate_mode", "renormalized"),
    window_size=mc.get("window_size", 8),
    moe_16_mode=mc.get("moe_16_mode", "sparse"),
    moe_type=mc.get("moe_type", "sparse"),
    use_deep_supervision=mc.get("deep_supervision", False),
    pretrained_backbone=False,
)
fingerprint = verify_parameter_consistency(probe)
print(f"  parameter-name fingerprint: {fingerprint}")
print("  recorded for the 0.0168 model: 2cd252ad3a3581e1c65961b828857d70")
print(f"  identical: {fingerprint == '2cd252ad3a3581e1c65961b828857d70'}")
del probe
''')

    add_markdown(r"""## 08_evaluation

Both test splits are evaluated concurrently, one per GPU, under each TTA setting. `none` is what
the legacy numbers were produced with; `hflip` is what the new runs use, so this run gives both
sides of the comparison.
""")

    add_code(r'''def run_eval(split: str, tta: str, gpu: int):
    """Launch one evaluation on one GPU, as a separate process."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    print(f"  [gpu{gpu}] split={split} tta={tta}")
    return subprocess.Popen([
        "python", "-u", "-m", "src.evaluate",
        "--checkpoint", LEGACY_PATH,
        "--dataset", split,
        "--data_dir", valid_root,
        "--out_dir", os.path.join(LEGACY_OUT, tta),
        "--tta", tta,
    ], cwd=PROJECT_ROOT, env=env)


for tta in TTA_MODES:
    print("\n" + "=" * 70)
    print(f"TTA={tta}: both splits, one per GPU")
    print("=" * 70)
    procs = [run_eval("test_real", tta, 0), run_eval("test_sys", tta, 1)]
    codes = [p.wait() for p in procs]
    assert all(c == 0 for c in codes), f"evaluation failed: exit codes {codes}"
print("\nAll evaluations finished.")
''')

    add_markdown(r"""## 09_results

Prints every metric produced, and restates the reference numbers so the diagnostic is readable
from the output alone.
""")

    add_code(r'''import glob

rows = []
for path in sorted(glob.glob(os.path.join(LEGACY_OUT, "**", "metrics_*.json"), recursive=True)):
    with open(path) as f:
        d = json.load(f)
    g = d.get("global", d)
    split = "test_real" if "test_real" in path else "test_sys"
    tta = "hflip" if "/hflip/" in path else "none"
    rows.append((tta, split, g.get("MAE"), g.get("S_measure"), g.get("E_adaptive"),
                 g.get("F_adaptive"), g.get("sample_count")))

print(f"  {'tta':6s} {'split':10s} {'MAE':>8s} {'S':>8s} {'E':>8s} {'F':>8s} {'n':>6s}")
for tta, split, mae, s, e, f, n in sorted(rows):
    print(f"  {tta:6s} {split:10s} {mae:8.4f} {s:8.4f} {e:8.4f} {f:8.4f} {n:6d}")

print("\nReference values (plain TTA, legacy evaluation):")
print("  test_real  MAE 0.0168  S 0.9151   (554 images)")
print("  test_sys   MAE 0.0192  S 0.9139   (1500 images)")
print("\nComparison values (hflip TTA, new runs, test_real):")
print("  REPRO 0.0195 | REPRO_SEED43 0.0199 | REPRO_SEED44 0.0191 | REPRODENSE 0.0199")
print("\nIf the plain-TTA row lands near 0.0168 the pipeline reproduces the legacy number;")
print("if it lands near 0.019 the evaluation protocol has changed since that measurement.")
''')

    add_markdown(r"""## 10_upload_results

Uploads the evaluation to `analysis_results/legacy_8expert_best/` on the Hub, the same way the
run notebooks upload under their experiment ID.
""")

    add_code(r'''from huggingface_hub import login, HfApi, create_repo

print(f"Uploading results for {LEGACY_LABEL} ...")
login(token=HF_TOKEN)
REPO_ID = os.environ.get("HF_REPO_ID", "Avi2006/spatial-moe-results")
create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=True)

api = HfApi()
uploaded_url = api.upload_folder(
    folder_path=LEGACY_OUT,
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo=f"analysis_results/{LEGACY_LABEL}",
)
print("Upload successful")
print("Destination:", uploaded_url)
print("Files uploaded:")
for root, _, files in os.walk(LEGACY_OUT):
    for f in files:
        print("  ", os.path.relpath(os.path.join(root, f), LEGACY_OUT))
''')

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.13", "mimetype": "text/x-python",
                              "codemirror_mode": {"name": "ipython", "version": 3},
                              "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                              "file_extension": ".py"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(OUTPUT_NOTEBOOK, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"{OUTPUT_NOTEBOOK} generated successfully ({len(cells)} cells).")


if __name__ == "__main__":
    create_notebook()
