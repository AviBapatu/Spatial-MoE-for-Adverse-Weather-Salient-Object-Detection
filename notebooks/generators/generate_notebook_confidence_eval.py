"""Generate the confidence-arm evaluation notebook.

Re-evaluates the two per-arm checkpoints already on the Hub, each into its own
output directory, and measures routing entropy on both.  No training.

Run:  python3 generate_notebook_confidence_eval.py
Writes: notebooks/training/moe-of-sod__confidence_arm_evaluation.ipynb
"""
import json, pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "training" / \
      "moe-of-sod__confidence_arm_evaluation.ipynb"

MD = lambda s: ("md", s.strip("\n"))
CO = lambda s: ("code", s.strip("\n"))

CELLS = [
MD(r"""
# Confidence arm — evaluation only

**No training.** The two arms have already trained; their checkpoints are on the Hub.
Only their *evaluation* is missing, for two reasons:

1. The first evaluation pass wrote both arms into **one shared output directory**, so
   the second upload pushed byte-identical images for both — the predictions are the
   same file on the Hub twice, and no comparison can be made from them.
2. The measurement the arm actually exists to make — **did the confidence loss move
   routing entropy off the ceiling?** — was never computed.

This notebook fixes both. Each arm gets its own output directory, and routing entropy
is read from `MoEOutput.entropy` (the accessor the account-A notebook uses).

| arm | `entropy_confidence_weight` |
|---|---|
| `CONF02` | 0.2 |
| `CONF00` | 0.0 (matched control) |

Everything is uploaded under each arm's own `analysis_results/<experiment_id>/` path.
"""),

MD("## 1 · Setup"),

CO(r"""
!pip install -q -U "huggingface_hub>=0.23" gdown pysodmetrics 2>/dev/null | tail -1
"""),

CO(r"""
import glob as _glob
import importlib, json, math, os, shutil, subprocess, sys, zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
REPO = "Avi2006/spatial-moe-results"
api = HfApi(token=HF_TOKEN)
api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=True)

WORK = Path("/kaggle/working")
PROJECT_ROOT = WORK / "spatial_moe_sod"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("hub:", REPO, "| device:", DEVICE)
"""),

CO(r"""
for _fordep, _mod in {"pysodmetrics": "py_sod_metrics", "timm": "timm",
                      "albumentations": "albumentations"}.items():
    try:
        importlib.import_module(_mod)
        print(f"  ok       {_fordep}")
    except ImportError:
        print(f"  install  {_fordep}")
        subprocess.run(["pip", "install", "-q", _fordep], check=False)
"""),

MD("## 2 · Source and data"),

CO(r"""
if not (PROJECT_ROOT / "src" / "evaluate.py").exists():
    print("fetching source ...")
    _z = hf_hub_download(REPO, "code/spatial_moe_sod_code.zip",
                         repo_type="dataset", token=HF_TOKEN)
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_z) as _zf:
        _zf.extractall(PROJECT_ROOT)
    if not (PROJECT_ROOT / "src").exists():
        _h = [_q for _q in PROJECT_ROOT.rglob("src") if (_q / "evaluate.py").exists()]
        if _h:
            shutil.copytree(_h[0].parent, WORK / "_src_tmp")
            shutil.rmtree(PROJECT_ROOT)
            shutil.move(str(WORK / "_src_tmp"), str(PROJECT_ROOT))

os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
print("source:", PROJECT_ROOT, "| src/evaluate.py:", (PROJECT_ROOT / "src" / "evaluate.py").exists())

# static import gate, before any GPU time
_failed = []
for _m in ("src.config", "src.model", "src.loss", "src.dataset", "src.metrics",
           "src.evaluate"):
    try:
        importlib.import_module(_m)
        print(f"  ok    import {_m}")
    except Exception as _e:
        _failed.append(_m)
        print(f"  FAIL  import {_m}: {type(_e).__name__}: {_e}")
assert not _failed, f"fix these imports first: {_failed}"
"""),

CO(r"""
DATA_FILE_ID = "1SUtAaxOy3c8Vzb5Ojp4NzxRB1vmsTjHL"
EXTRACT_PATH = WORK / "WXSDO_data"


def is_dataset_valid(path):
    '''Return a WXSOD root under *path*, or None.'''
    def check(r):
        reqs = [f"{s}/{part}" for s in ("train_sys", "test_sys", "test_real")
                for part in ("input", "gt")]
        return all(os.path.isdir(os.path.join(r, q)) for q in reqs)
    if not os.path.isdir(str(path)):
        return None
    if check(str(path)):
        return Path(path)
    for root, _d, _f in os.walk(str(path)):
        if check(root):
            return Path(root)
    return None


DATA_ROOT = None
for _c in _glob.glob("/kaggle/input/*") + _glob.glob("/kaggle/input/*/*"):
    _f = is_dataset_valid(_c)
    if _f:
        DATA_ROOT = _f
        break
if DATA_ROOT is None:
    DATA_ROOT = is_dataset_valid(EXTRACT_PATH)
if DATA_ROOT is None:
    print("fetching dataset ...")
    _zip = WORK / "dataset.zip"
    if not _zip.exists():
        os.system("pip install -q gdown 2>/dev/null")
        os.system(f"gdown {DATA_FILE_ID} -O {_zip}")
    EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
    os.system(f"unzip -q -o {_zip} -d {EXTRACT_PATH}")
    DATA_ROOT = is_dataset_valid(EXTRACT_PATH)
assert DATA_ROOT is not None, "no WXSOD ground truth found"
print("DATA_ROOT =", DATA_ROOT)
for _sp in ("test_sys", "test_real"):
    print(f"  {_sp}: {len(os.listdir(DATA_ROOT / _sp / 'input'))} inputs")
"""),

MD("## 3 · The two arms"),

CO(r"""
ARMS = {
    "CONF02": "EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_ENTROPYCONF",
    "CONF00": "EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_ENTROPYCONFCTRL",
}
SPLIT = "test_real"

# One output directory PER ARM.  Sharing one was the bug in the training notebook:
# the second arm's evaluation wrote over the first, and both uploads pushed the same
# images.  accountA raises on a shared path for exactly this reason.
EVAL_ROOT = WORK / "confidence_eval"
for _name in ARMS:
    (EVAL_ROOT / _name).mkdir(parents=True, exist_ok=True)
print("per-arm output roots:")
for _name in ARMS:
    print("   ", EVAL_ROOT / _name)
"""),

MD("## 4 · Load the checkpoints (reference loader)"),

CO(r"""
from src.model import SpatialMoESODNet


def load_model_from_checkpoint(ckpt_path, device):
    '''Copied from accountA section 15: the checkpoint carries its own config, so
    every architecture-defining field comes from checkpoint["config"]["model"].'''
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("config", {}).get("model", {})
    assert "num_experts" in model_cfg, "checkpoint config missing num_experts"
    model = SpatialMoESODNet(
        dim=model_cfg.get("working_dim", 256),
        num_experts=model_cfg.get("num_experts", 8),
        k=model_cfg.get("top_k", 2),
        gate_mode=model_cfg.get("gate_mode", "renormalized"),
        window_size=model_cfg.get("window_size", 7),
        moe_type=model_cfg.get("moe_type", "sparse"),
        moe_16_mode=model_cfg.get("moe_16_mode", "sparse"),
        use_deep_supervision=model_cfg.get("deep_supervision", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_cfg, checkpoint


CKPTS, MODELS, MODEL_CFGS = {}, {}, {}
for _name, _exp in ARMS.items():
    _p = hf_hub_download(REPO, f"checkpoints/{_exp}/best.pth",
                         repo_type="dataset", token=HF_TOKEN)
    _m, _cfg, _ck = load_model_from_checkpoint(_p, DEVICE)
    CKPTS[_name], MODELS[_name], MODEL_CFGS[_name] = _p, _m, _cfg
    _sd = _ck["model_state_dict"]
    print(f"{_name}: epoch {_ck.get('epoch','?')}  best_metric "
          f"{_ck.get('best_metric','?')}  conf_weight "
          f"{_ck.get('config',{}).get('loss',{}).get('entropy_confidence_weight','?')}"
          f"  aux_tensors {len([k for k in _sd if 'aux_head' in k])}"
          f"  params {sum(q.numel() for q in _m.parameters()):,}")
"""),

MD("## 5 · Evaluate each arm into its own directory"),

CO(r"""
# One subprocess per ARM, one GPU per arm -- the launch pattern accountA section 14
# uses (it gives each work item its own CUDA_VISIBLE_DEVICES).  Doing it this way also
# means only one 756 MB checkpoint is resident per process.
NGPU = max(1, torch.cuda.device_count())
print(f"{NGPU} GPU(s) available -- assigning one arm per GPU")

EVAL_JOBS = []
for _i, (_name, _exp) in enumerate(ARMS.items()):
    _out = EVAL_ROOT / _name
    if _out.exists() and any(_out.iterdir()):
        raise RuntimeError(
            f"{_out} already holds results -- one directory per arm is what makes the "
            "comparison meaningful.")
    _ck = hf_hub_download(REPO, f"checkpoints/{_exp}/best.pth",
                          repo_type="dataset", token=HF_TOKEN)
    _gpu = _i % NGPU
    _cmd = [sys.executable, "-u", "-m", "src.evaluate",
            "--checkpoint", _ck,
            "--data_dir", str(DATA_ROOT),
            "--out_dir", str(_out),
            "--tta", "none",
            "--dataset", SPLIT]
    _env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(_gpu))
    print(f"  [gpu{_gpu}] {_name}  ({_out})")
    EVAL_JOBS.append((_name, _cmd, subprocess.Popen(
        _cmd, cwd=str(PROJECT_ROOT), env=_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))

RESULTS = {}
for _name, _cmd, _proc in EVAL_JOBS:
    _txt, _ = _proc.communicate()
    print(f"  {_name}: exit={_proc.returncode}")
    if _proc.returncode != 0:
        print("\n".join(_txt.splitlines()[-30:]))
    RESULTS[_name] = {"returncode": _proc.returncode}
    _n = len(list((EVAL_ROOT / _name).glob("*.png")))
    print(f"    wrote {_n} prediction maps")
"""),

MD(r"""
## 6 · Did the confidence loss move the routing?

This is the measurement the arm exists to make. `MoEOutput.entropy` is the normalised
routing entropy per scale (`H / log E`), so a value at **1.0** means the router is at
its ceiling — maximally undecided — and a value below that means the confidence
objective actually pushed the routing distribution.
"""),

CO(r"""
def collect_entropy(model, dataloader, device, max_batches=None):
    '''Routing entropy per scale, read off MoEOutput -- accountA section 17.'''
    model.eval()
    per_scale = defaultdict(list)
    with torch.no_grad():
        for _i, _batch in enumerate(dataloader):
            if max_batches and _i >= max_batches:
                break
            _images = _batch["image"].to(device)
            _, _moe = model(_images)
            for _sname, _out in zip(("scale_4", "scale_8", "scale_16"), _moe):
                _e = getattr(_out, "entropy", None)
                if _e is not None:
                    per_scale[_sname].append(
                        float(_e.mean()) if torch.is_tensor(_e) else float(_e))
    return {k: sum(v) / len(v) for k, v in per_scale.items()}


ENTROPY = {}
for _name in ARMS:
    try:
        ENTROPY[_name] = collect_entropy(MODELS[_name], test_real_loader, DEVICE)
        print(f"  {_name}: " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(ENTROPY[_name].items())))
    except Exception as _e:
        ENTROPY[_name] = None
        print(f"  {_name}: entropy collection failed: {type(_e).__name__}: {_e}")
"""),

CO(r"""
# The comparison.  A confidence objective that works should pull entropy DOWN
# relative to the control; if the two columns match, it did nothing.
print("=" * 74)
print(f"{'scale':<12s} {'CONF02 (w=0.2)':>18s} {'CONF00 (w=0.0)':>18s} {'delta':>12s}")
print("-" * 74)
_scales = sorted(set(list((ENTROPY.get("CONF02") or {}).keys()) +
                     list((ENTROPY.get("CONF00") or {}).keys())))
for _s in _scales:
    _a = (ENTROPY.get("CONF02") or {}).get(_s)
    _b = (ENTROPY.get("CONF00") or {}).get(_s)
    if _a is None or _b is None:
        print(f"{_s:<12s} {'-':>18s} {'-':>18s} {'-':>12s}")
    else:
        print(f"{_s:<12s} {_a:18.4f} {_b:18.4f} {_a - _b:+12.4f}")
print("=" * 74)
print("1.0000 = the router is at its entropy ceiling (maximally undecided).")
"""),

MD("## 7 · Upload, per arm"),

CO(r"""
for _name, _exp in ARMS.items():
    _dst = f"analysis_results/{_exp}"
    _out = EVAL_ROOT / _name

    if _out.exists() and any(_out.iterdir()):
        api.upload_folder(repo_id=REPO, folder_path=str(_out),
                          path_in_repo=f"{_dst}/eval/{SPLIT}",
                          repo_type="dataset", token=HF_TOKEN)
        print(f"uploaded {_dst}/eval/{SPLIT}/")

    _payload = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        arm=_name, experiment_id=_exp, split=SPLIT,
        entropy_confidence_weight=MODEL_CFGS[_name].get("entropy_confidence_weight"),
        routing_entropy=ENTROPY.get(_name),
        metrics=(RESULTS.get(_name) or {}).get("global"),
    )
    _f = WORK / f"confidence_eval_{_name}.json"
    _f.write_text(json.dumps(_payload, indent=2, default=float))
    api.upload_file(repo_id=REPO, path_or_fileobj=str(_f),
                    path_in_repo=f"{_dst}/confidence_eval.json",
                    repo_type="dataset", token=HF_TOKEN)
    print(f"uploaded {_dst}/confidence_eval.json")

print("\ndone.")
"""),

MD(r"""
## 8 · How to read the result

**If the two entropy columns differ**, the confidence objective changed the routing
distribution under training — the MoE was given a fair chance, and the paper's
"collapse" framing is about a configurable property, not an invariant.

**If the two columns are identical**, the honest sentence is: *"a routing objective
that directly rewards confident routing, for eight epochs on this recipe, did not move
the routing distribution."* That is a real finding and it closes the reviewer's
fair-chance objection.

Either way this run supersedes the byte-identical predictions on the Hub — replace
those, because as they stand they show the two arms producing the same output, which
is an artefact of the shared output directory and not a property of the arms.
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
