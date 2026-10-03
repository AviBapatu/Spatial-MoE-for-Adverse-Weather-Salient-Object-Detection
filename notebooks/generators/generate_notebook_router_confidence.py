"""Generate the router-confidence training notebook (review item 3).

One (optionally two) 8-epoch run(s) on the ablation-family recipe:
the confidence arm, and a matched control that differs only in
`entropy_confidence_weight`.

Run:  python3 generate_notebook_router_confidence.py
Writes: notebooks/training/moe-of-sod__router_confidence_arm.ipynb
"""
import json, pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "training" / \
      "moe-of-sod__router_confidence_arm.ipynb"

MD = lambda s: ("md", s.strip("\n"))
CO = lambda s: ("code", s.strip("\n"))

CELLS = [
MD(r"""
# Router-confidence arm

**Review item 3.** *"You have a router-confidence loss implemented and disabled, and
your conclusion lists 'a routing objective that induces specialisation' as future work.
Run it. A null result is only convincing if the MoE was given a fair chance."*

The loss term already exists — `src/loss.py:484` adds
`cfg.entropy_confidence_weight * l_routing_conf`, and `src/config.py:92` defines the
field with a default of `0.0`. The config `experiments/v_4expert_entropyconf.json`
already sets it to `0.2` at the ablation-family recipe (E=4, k=2, 8 epochs); it has
simply never been run, which is why it is absent from `analysis_results/` on the Hub.

**What runs.** Two arms at the family recipe, differing in exactly one field:

| | `entropy_confidence_weight` |
|---|---|
| `CONF02` | 0.2 |
| `CONF00` (matched control) | 0.0 |

The matched control exists because a confidence arm is only interpretable against a
run that differs in nothing else. Set `RUN_CONTROL = False` at the top if GPU time is
short — the existing E4 k2 arms can serve as an approximate control, but they are not
a matched one.

**Either outcome helps.** If routing starts to bite, you have a mechanism result and
the paper's thesis changes. If it does not, *"we tried to induce specialisation and
could not"* is a materially stronger claim than the current *"we did not try."*

Everything (checkpoint, metrics, per-image predictions, routing diagnostics) is
uploaded to the Hub.
"""),

MD("## 1 · Setup"),

CO(r"""
!pip install -q -U "huggingface_hub>=0.23" gdown 2>/dev/null | tail -1
"""),

CO(r"""
import json, os, re, shutil, subprocess, sys, zipfile
from pathlib import Path

from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
REPO     = "Avi2006/spatial-moe-results"
api      = HfApi(token=HF_TOKEN)
api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=True)

WORK = Path("/kaggle/working")
print("hub:", REPO)
"""),

CO(r"""
# how many GPUs did Kaggle give us?
import torch
NGPU = torch.cuda.device_count()
for i in range(NGPU):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory/2**30:.1f} GiB")
print("=> torchrun --nproc_per_node =", max(1, NGPU))
"""),

MD(r"""
## 1.1 · Dependencies

`src/training/loop.py` imports `src.metrics`, which imports `py_sod_metrics` at module
load — so the **pip name is `pysodmetrics`** (imported as `py_sod_metrics`) and it must
be present before `src.train_ddp` can even be imported. This is the one dependency that
the inference-only notebooks can skip and training cannot.
"""),

CO(r"""
import importlib, subprocess

deps = {
    "pysodmetrics":            "py_sod_metrics",
    "timm":                    "timm",
    "albumentations":          "albumentations",
    "opencv-python-headless":  "cv2",
}
for pip_name, mod_name in deps.items():
    try:
        importlib.import_module(mod_name)
        print(f"  ok       {pip_name}")
    except ImportError:
        print(f"  install  {pip_name} ...")
        subprocess.run(["pip", "install", "-q", pip_name], check=False)
        try:
            importlib.import_module(mod_name)
            print(f"  ok       {pip_name}")
        except ImportError:
            print(f"  FAILED   {pip_name} -- training will not start without it")
"""),

MD("## 2 · Fetch the source code and the dataset"),

CO(r"""
PROJECT_ROOT = Path("/kaggle/working/spatial_moe_sod")
REPO_ROOT = PROJECT_ROOT
if not (PROJECT_ROOT / "src" / "train_ddp.py").exists():
    print("fetching source from the Hub ...")
    z = hf_hub_download(REPO, "code/spatial_moe_sod_code.zip",
                        repo_type="dataset", token=HF_TOKEN)
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(REPO_ROOT)
    if not (REPO_ROOT / "src").exists():
        hits = [p for p in REPO_ROOT.rglob("src") if (p / "train_ddp.py").exists()]
        if hits:
            tmp = WORK / "repo_tmp"
            shutil.copytree(hits[0].parent, tmp)
            shutil.rmtree(REPO_ROOT)
            shutil.move(str(tmp), str(REPO_ROOT))

print("source root:", REPO_ROOT)
print("  src/train_ddp.py:", (REPO_ROOT / "src" / "train_ddp.py").exists())
print("  src/config.py   :", (REPO_ROOT / "src" / "config.py").exists())
print("  train.py        :", (REPO_ROOT / "train.py").exists())
print("  experiments/    :", (REPO_ROOT / "experiments").is_dir())
"""),

CO(r"""
DATA_FILE_ID = "1SUtAaxOy3c8Vzb5Ojp4NzxRB1vmsTjHL"   # project archive, Google Drive
EXTRACT_PATH = WORK / "WXSDO_data"
import glob as _glob


def is_dataset_valid(path):
    '''Return a WXSOD root under *path*, or None.

    A valid root has train_sys, test_sys and test_real, each with input/ and gt/.
    '''
    def check(r):
        reqs = [f"{sp}/{part}" for sp in ("train_sys", "test_sys", "test_real")
                for part in ("input", "gt")]
        return all(os.path.isdir(os.path.join(r, q)) for q in reqs)
    if not os.path.isdir(str(path)):
        return None
    if check(str(path)):
        return Path(path)
    for root, _dirs, _files in os.walk(str(path)):
        if check(root):
            return Path(root)
    return None


DATA_ROOT = None
for cand in _glob.glob("/kaggle/input/*") + _glob.glob("/kaggle/input/*/*"):
    found = is_dataset_valid(cand)
    if found:
        DATA_ROOT = found
        print("using attached dataset:", DATA_ROOT)
        break

if DATA_ROOT is None:
    DATA_ROOT = is_dataset_valid(EXTRACT_PATH)
    if DATA_ROOT:
        print("using existing download:", DATA_ROOT)

if DATA_ROOT is None:
    print("dataset not found; fetching from Google Drive ...")
    ZIP_PATH = WORK / "dataset.zip"
    if not ZIP_PATH.exists():
        os.system("pip install -q gdown 2>/dev/null")
        os.system(f"gdown {DATA_FILE_ID} -O {ZIP_PATH}")
    EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
    os.system(f"unzip -q -o {ZIP_PATH} -d {EXTRACT_PATH}")
    DATA_ROOT = is_dataset_valid(EXTRACT_PATH)

if DATA_ROOT is None:
    raise RuntimeError("no WXSOD ground truth found - check the Drive id, or attach "
                       "the dataset to the notebook as a Kaggle input")

print("DATA_ROOT =", DATA_ROOT)
for _sp in ("train_sys", "test_sys", "test_real"):
    _i = len([f for f in os.listdir(DATA_ROOT / _sp / "input") if not f.startswith(".")])
    _g = len([f for f in os.listdir(DATA_ROOT / _sp / "gt") if not f.startswith(".")])
    print(f"  {_sp}: {_i} inputs, {_g} gt")
"""),

MD(r"""
## 2.1 · Deploy gates (ported from `generate_notebook_accountA.py`)

The working notebooks check the checkout is complete and that every training module
**imports** before they spend GPU time. The import gate is the one that matters: it
catches a missing third-party package (`py_sod_metrics`, `albumentations`, ...) in about
two seconds with a readable traceback, instead of ten seconds into a 2-GPU torchrun
where torchrun's elastic summary buries the real cause.
"""),

CO(r"""
# ---- gate 1: is the checkout complete?
REQUIRED_FILES = [
    "src/train_ddp.py", "src/model.py", "src/loss.py", "src/dataset.py",
    "src/evaluate.py", "src/metrics.py", "src/config.py",
    "src/training/cli.py", "src/training/loop.py",
]
_missing = [f for f in REQUIRED_FILES if not (PROJECT_ROOT / f).exists()]
print("files:", "ALL PRESENT" if not _missing else f"MISSING {_missing}")
assert not _missing, f"incomplete checkout under {PROJECT_ROOT}: {_missing}"

# ---- gate 2: static import check (accountA section 08)
os.chdir(str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for _m in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
    del sys.modules[_m]

import importlib
_failed = []
for _mod in ("src.config", "src.model", "src.loss", "src.dataset",
             "src.metrics", "src.train_ddp"):
    try:
        importlib.import_module(_mod)
        print(f"  ok    import {_mod}")
    except Exception as _e:
        _failed.append((_mod, f"{type(_e).__name__}: {_e}"))
        print(f"  FAIL  import {_mod}  ->  {type(_e).__name__}: {_e}")

if _failed:
    print("\nFix the failing import(s) above before running anything else. "
          "A ModuleNotFoundError here means the matching pip package is missing "
          "(e.g. py_sod_metrics <- pysodmetrics).")
    raise ImportError(f"{len(_failed)} module(s) failed to import: "
                      f"{[m for m, _ in _failed]}")
print("\nstatic import check passed - training can start")
"""),

MD("## 3 · Write the two arm configs"),

CO(r"""
# Base: experiments/v_4expert_entropyconf.json, i.e. the ablation-family recipe
# (E=4, k=2, 8 epochs) with the confidence term enabled.  We change only the
# dataset root, the experiment id, and -- per arm -- the confidence weight.
import copy

RUN_CONTROL = True          # set False to skip the matched control

BASE = {
    "data": {"dataset_root": str(DATA_ROOT), "split_manifest_path": "",
             "split_manifest_hash": "", "max_samples": None},
    "model": {"backbone": "pvt_v2_b4", "working_dim": 256, "decoder_dim": 256,
              "num_experts": 4, "top_k": 2, "router_variant": "token_only",
              "expert_variant": "standard", "attention_config": "default",
              "window_size": 7, "deep_supervision": False, "moe_type": "sparse",
              "moe_16_mode": "sparse", "router_noise_enabled": False,
              "router_noise_scale": 1.0, "router_noise_min_std": 0.1,
              "gate_mode": "dense"},
    "loss": {"bce_weight": 1.0, "iou_weight": 1.0, "ssim_weight": 1.0,
             "boundary_weight": 1.5, "load_balance_weight": 0.01,
             "load_balance_weights": [0.016, 0.03, 0.02], "importance_weight": 0.0,
             "z_loss_weight": 0.0, "aux_boundary_weight": 0.5,
             "deep_supervision_weight": 0.0, "entropy_confidence_weight": 0.2},
    "opt": {"optimizer": "AdamW", "backbone_lr": 0.0001, "new_module_lr": 0.0001,
            "weight_decay": 0.0001, "warmup_ratio": 0.04, "scheduler": "WarmupCosine",
            "gradient_clipping": 1.0, "amp": True, "batch_per_gpu": 4,
            "grad_accum_steps": 5, "effective_global_batch": 40},
    "train": {"epochs": 8, "freeze_backbone_epochs": 1, "seed": 42, "num_workers": 2,
              "checkpoint_every_n_steps": 200},
    "eval": {"validation_metric_selection": "MAE", "tta_mode": "none",
             "metric_config": "default"},
    "diag": {"routing_diagnostic_epochs": 1, "expert_ablation_enabled": False},
    "experiment_id": "", "variant": "",
}

def arm_config(conf_weight: float, variant: str) -> dict:
    c = copy.deepcopy(BASE)
    c["loss"]["entropy_confidence_weight"] = conf_weight
    c["variant"] = variant
    c["experiment_id"] = f"EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_{variant}"
    return c

ARMS = {"CONF02": arm_config(0.2, "ENTROPYCONF")}
if RUN_CONTROL:
    ARMS["CONF00"] = arm_config(0.0, "ENTROPYCONFCTRL")

CFG_DIR = WORK / "arm_configs"
CFG_DIR.mkdir(exist_ok=True)
for name, cfg in ARMS.items():
    (CFG_DIR / f"{name}.json").write_text(json.dumps(cfg, indent=2))
    print(f"  {name}: entropy_confidence_weight={cfg['loss']['entropy_confidence_weight']}"
          f"  variant={cfg['variant']}  -> {CFG_DIR/(name+'.json')}")
"""),

MD("## 4 · Validate the configs before spending GPU time"),

CO(r"""
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

for name in ARMS:
    path = CFG_DIR / f"{name}.json"
    print(f"=== {name} ===")
    r = subprocess.run([sys.executable, "-c",
        f"from src.config import ExperimentConfig; "
        f"c = ExperimentConfig.load({str(path)!r}); "
        f"print('ok  epochs=', c.train.epochs, ' E=', c.model.num_experts,"
        f" ' k=', c.model.top_k, ' conf=', c.loss.entropy_confidence_weight,"
        f" ' hash=', c.get_canonical_hash()[:12])"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    print("  " + (r.stdout.strip() or r.stderr.strip()[-600:]))
"""),

MD(r"""
## 4.1 · Experiment IDs

The trainer does **not** read `experiment_id` from the config — it derives the name from
the config itself via `generate_experiment_id()`. Everything downstream (checkpoint
paths, Hub artifact names) keys off that derived ID, so we recompute it here rather
than guessing a path. This mirrors the account-A/B notebooks, which ran successfully.
"""),

CO(r"""
os.chdir(str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for _m in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
    del sys.modules[_m]

from src.config import ExperimentConfig
from src.experiment import generate_experiment_id

EXP_IDS = {}
for _name in ARMS:
    _cfg = ExperimentConfig.load(str(CFG_DIR / f"{_name}.json"))
    EXP_IDS[_name] = generate_experiment_id(_cfg)
    print(f"  {_name}: {EXP_IDS[_name]}")

CHECKPOINT_ROOT = Path("/kaggle/working/WXSOD_Checkpoints")
CHECKPOINT_ROOT.mkdir(exist_ok=True)
print("\ncheckpoints will appear under:", CHECKPOINT_ROOT)
"""),

MD("## 4.2 · Preflight (config, data, one backward pass)"),

CO(r"""
# The CLI exposes --preflight --max_optimizer_steps, which runs the real startup path
# and a few optimizer steps, then exits.  If a config field is wrong this fails here
# in seconds with a readable message, instead of 10 s into a 2-GPU torchrun where
# torchrun's own summary buries the real traceback.
for name in ARMS:
    cfg_path = CFG_DIR / f"{name}.json"
    cmd = ["torchrun", "--nproc_per_node=1", "-m", "src.train_ddp",
           "--config", str(cfg_path), "--preflight", "--max_optimizer_steps", "5"]
    print(f"{chr(61)*78}\n{name} preflight\n{chr(61)*78}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    out = (r.stdout + "\n" + r.stderr)
    lines = out.splitlines()
    print(f"  exit={r.returncode}  ({len(lines)} lines)")
    if r.returncode != 0:
        keys = ("Traceback (most recent call last)", "Error", "error:", "assert",
                "Exception", "raise ", "ValidationError", "ValueError")
        hits = [i for i, l in enumerate(lines) if any(k in l for k in keys)]
        if hits:
            i0 = max(0, hits[0] - 3)
            print("  " + "\n  ".join(lines[i0:hits[0] + 25]))
        else:
            print("  " + "\n  ".join(lines[-40:]))
    else:
        print("  " + "\n  ".join(lines[-12:]))
"""),

MD("## 5 · Train"),

CO(r"""
# 8 epochs, 2 GPUs, ~40 min-1 h per arm depending on the card.
LOGS = WORK / "train_logs"
LOGS.mkdir(exist_ok=True)
NPROC = max(1, NGPU)

for name in ARMS:
    cfg_path = CFG_DIR / f"{name}.json"
    log_path = LOGS / f"{name}.log"
    # --overwrite matches the account-A/B notebooks: without it the trainer can
    # refuse when a run directory from a previous attempt is still present.
    cmd = ["torchrun", f"--nproc_per_node={NPROC}", "-m", "src.train_ddp",
           "--config", str(cfg_path), "--overwrite"]
    print(f"\n{chr(61)*78}\n{name}: {' '.join(cmd)}\n  log -> {log_path}\n{chr(61)*78}")
    with open(log_path, "w") as lf:
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT), stdout=lf,
                             stderr=subprocess.STDOUT)
    print(f"  exit={rc}")
    text = Path(log_path).read_text(errors="replace")
    lines = text.splitlines()
    if rc != 0:
        # torchrun prints its own elastic summary; the real cause is earlier in the
        # log, so pull out every traceback / error line and show its neighbourhood.
        keys = ("Traceback (most recent call last)", "Error", "error:", "assert",
                "Exception", "raise ")
        hits = [i for i, l in enumerate(lines) if any(k in l for k in keys)]
        if hits:
            i0 = max(0, hits[0] - 3)
            i1 = min(len(lines), hits[0] + 25)
            print(f"  --- first error, log lines {i0+1}-{i1} of {len(lines)} ---")
            print("  " + "\n  ".join(lines[i0:i1]))
        else:
            print("  --- last 40 log lines ---")
            print("  " + "\n  ".join(lines[-40:]))
    else:
        print("  " + "\n  ".join(lines[-15:]))
"""),

CO(r"""
# surface the key diagnostics each run wrote
for name in ARMS:
    ck_dir = CHECKPOINT_ROOT / EXP_IDS[name]
    print(f"\n=== {name}: {EXP_IDS[name]} ===")
    print("   checkpoint dir:", ck_dir, "exists:", ck_dir.exists())
    if ck_dir.exists():
        for f in sorted(ck_dir.rglob("*")):
            if f.is_file():
                print(f"      {f.name}  {f.stat().st_size/1e6:.1f} MB")
"""),

MD(r"""
## 6 · Evaluate and produce per-image predictions

The confidence arm is only interesting if it changes *routing*. So alongside the
segmentation metrics we pull the routing diagnostics: mean normalised entropy,
hard expert-usage histogram, and the load-balance terms. If
`entropy_confidence_weight = 0.2` does its job, entropy should move **away** from
the ceiling and the usage histogram should flatten or sharpen measurably.
"""),

CO(r"""
# Evaluate with the flags the account-A/B notebooks actually use:
#   --checkpoint / --data_dir / --out_dir / --tta / --dataset <single split>
# src.evaluate is single-process and each split writes its own metrics, so we run
# test_real on one GPU and leave the other free.
EVAL_OUT = WORK / "WXSOD_EvalResults"
EVAL_OUT.mkdir(exist_ok=True)

for name in ARMS:
    # ONE DIRECTORY PER ARM.  A shared output directory was a real bug here: the
    # second arm's evaluation wrote over the first and both uploads pushed the same
    # images, so the two arms looked byte-identical on the Hub.  accountA raises on a
    # reused path for exactly this reason; we do the same, since the alternative is a
    # silently meaningless comparison.
    arm_out = EVAL_OUT / name
    arm_out.mkdir(parents=True, exist_ok=True)
    if any(arm_out.iterdir()):
        raise RuntimeError(
            f"{arm_out} already holds results -- one directory per arm is what makes "
            f"the comparison meaningful.")
    ck = CHECKPOINT_ROOT / EXP_IDS[name] / "best.pth"
    if not ck.exists():
        cands = sorted((CHECKPOINT_ROOT / EXP_IDS[name]).glob("*.pth")) \
                if (CHECKPOINT_ROOT / EXP_IDS[name]).exists() else []
        ck = cands[-1] if cands else None
    if ck is None or not ck.exists():
        print(f"{name}: no checkpoint, skipping evaluation")
        continue

    cmd = [sys.executable, "-u", "-m", "src.evaluate",
           "--checkpoint", str(ck),
           "--data_dir", str(DATA_ROOT),
           "--out_dir", str(arm_out),
           "--tta", "none",
           "--dataset", "test_real"]
    print(f"\n{name}: {' '.join(cmd)}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env,
                       capture_output=True, text=True)
    print(f"  exit={r.returncode}")
    out = (r.stdout + "\n" + r.stderr).splitlines()
    if r.returncode != 0:
        hits = [k for k, l in enumerate(out)
                if any(t in l for t in ("Traceback", "Error", "error:", "raise "))]
        k0 = max(0, (hits[0] - 3) if hits else len(out) - 40)
        print("  " + "\n  ".join(out[k0:k0 + 28]))
    else:
        print("  " + "\n  ".join(out[-12:]))

print("\neval outputs:", sorted(q.name for q in EVAL_OUT.glob("*"))[:20])
"""),

MD("## 7 · Upload to the Hub"),

CO(r"""
import datetime

uploaded = []
skipped = []
for name, cfg in ARMS.items():
    exp_id = EXP_IDS[name]
    dst    = f"analysis_results/{exp_id}"

    # 1. the checkpoint.  The trainer pushes <id>_latest.pth / _best.pth itself via
    #    src.hf_sync; this covers the local copy as well so nothing depends on that.
    ck_dir = CHECKPOINT_ROOT / exp_id
    if not ck_dir.exists():
        skipped.append(f"{name}: no checkpoint directory (training did not finish)")
    if ck_dir.exists():
        for c in sorted(ck_dir.glob("*.pth")):
            api.upload_file(repo_id=REPO, path_or_fileobj=str(c),
                            path_in_repo=f"checkpoints/{exp_id}/{c.name}",
                            repo_type="dataset", token=HF_TOKEN)
            uploaded.append(f"checkpoints/{exp_id}/{c.name}")

    # 2. evaluation outputs, including per-image predictions for the bootstrap
    ev = EVAL_OUT / name
    if ev.exists() and any(ev.iterdir()):
        api.upload_folder(repo_id=REPO, folder_path=str(ev),
                          path_in_repo=f"analysis_results/{exp_id}/eval/test_real",
                          repo_type="dataset", token=HF_TOKEN,
                          ignore_patterns=["*.pth", "__pycache__"])
        uploaded.append(f"{dst}/eval/test_real/ (eval)")

    # 3. routing diagnostics, wherever the trainer left them
    for pat in ("latest_diagnostics.json", "diagnostics.json"):
        for f in PROJECT_ROOT.rglob(pat):
            api.upload_file(repo_id=REPO, path_or_fileobj=str(f),
                            path_in_repo=f"{dst}/diagnostics/{f.name}",
                            repo_type="dataset", token=HF_TOKEN)
            uploaded.append(f"{dst}/diagnostics/{f.name}")
            break

    # 4. the exact config that produced it
    api.upload_file(repo_id=REPO, path_or_fileobj=str(CFG_DIR / f"{name}.json"),
                    path_in_repo=f"{dst}/arm_config.json",
                    repo_type="dataset", token=HF_TOKEN)
    uploaded.append(f"{dst}/arm_config.json")

for u in uploaded:
    print("uploaded", u)
if skipped:
    print("\nskipped (nothing to upload):")
    for k in skipped:
        print("  ", k)
print(f"\n{len(uploaded)} artefacts pushed to {REPO}")
if not uploaded:
    print("Nothing was uploaded.  Check the training log above for the real error; "
          "the first traceback is printed by the cell that ran the command.")
print("done.")
"""),

MD(r"""
## 8 · What to look for in the output

Before the paper can claim fairness, the confidence arm has to be shown to have
changed *something*. The numbers to compare against the `CONF00` control:

* **normalised routing entropy** — did it leave the ceiling?
* **hard expert-usage histogram** — flatter, or more concentrated?
* **MAE / maxF on `test_real`** — with the paired bootstrap from the companion
  notebook, not by eye.
* **router gradient norm share** — did the router start receiving task gradient?

If entropy is unmoved, the honest sentence for the paper is: *"a routing objective
that directly rewards confident routing for eight epochs under this recipe did not
move the routing distribution."* That is a real finding, and it closes the
fair-chance objection that currently sits in the Limitations section.
"""),
]


def main():
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code",
                "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
