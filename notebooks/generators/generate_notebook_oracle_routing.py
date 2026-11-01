"""Generate the oracle-routing positive-control notebook (review item 3).

Trains one arm with weather-label hard routing, then evaluates it with and without the
oracle.  No src/ edit and no code-zip repackaging: the oracle is installed at runtime by
wrapping each MoE layer's router MLP, so the training loop in this notebook is
self-contained and auditable.

Run:  python3 generate_notebook_oracle_routing.py
Writes: notebooks/training/moe-of-sod__oracle_routing.ipynb
"""
import json, pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "training" / \
      "moe-of-sod__oracle_routing.ipynb"

MD = lambda s: ("md", s.strip("\n"))
CO = lambda s: ("code", s.strip("\n"))

RUNNER = r'''
import argparse, json, math, os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.dataset import get_dataloaders, get_weather_type
from src.loss import CombinedLoss
from src.metrics import SODMetrics
from src.model import SpatialMoESODNet

# One expert per weather suffix.  Fixed table, alphabetic by weather name, so the
# oracle signal is deterministic and independent of anything the model learns.
WEATHERS = ["clean", "dark", "fog", "light", "rain", "rainafog", "rainasnow",
            "snow", "snowafog"]
_ORACLE = {"ids": None}


class OracleRouter(nn.Module):
    # Wraps a router MLP and adds a large bias to the label-chosen expert, so the
    # standard SpatialMoELayer.forward picks it first.  No source edit needed.
    def __init__(self, orig, num_experts):
        super().__init__()
        self.orig = orig
        self.E = num_experts

    def forward(self, x):
        logits = self.orig(x)
        ids = _ORACLE["ids"]
        if ids is not None:
            ids = ids.to(logits.device)
            B = logits.shape[0]
            bias = torch.zeros(B, 1, self.E, device=logits.device, dtype=logits.dtype)
            bias.scatter_(2, ids.view(B, 1, 1), 1e4)
            logits = logits + bias
        return logits


def install_oracle(model, num_experts):
    # Idempotent: skip a layer already wrapped.
    for layer in (model.moe_4, model.moe_8, model.moe_16):
        if hasattr(layer, "router_mlp") and not isinstance(layer.router_mlp, OracleRouter):
            layer.router_mlp = OracleRouter(layer.router_mlp, num_experts)


def ids_from_names(names, num_experts):
    table = {w: i % num_experts for i, w in enumerate(sorted(WEATHERS))}
    return torch.tensor([table.get(get_weather_type(n), 0) for n in names], dtype=torch.long)


def build_model(cfg):
    m = cfg.model
    return SpatialMoESODNet(
        dim=m.working_dim, num_experts=m.num_experts, k=m.top_k,
        gate_mode=m.gate_mode, window_size=m.window_size, moe_type=m.moe_type,
        moe_16_mode=getattr(m, "moe_16_mode", "sparse"),
        use_deep_supervision=m.deep_supervision,
        router_noise_enabled=m.router_noise_enabled,
        router_noise_scale=m.router_noise_scale,
        router_noise_min_std=m.router_noise_min_std,
    )


def forward_step(model, batch, cfg, oracle, device):
    imgs = batch["image"].to(device, non_blocking=True)
    _ORACLE["ids"] = ids_from_names(batch["name"], cfg.model.num_experts) if oracle else None
    try:
        dec, moe = model(imgs)
    finally:
        _ORACLE["ids"] = None
    return dec, moe


def evaluate(model, loader, cfg, oracle, device):
    model.eval()
    metric = SODMetrics()
    with torch.no_grad():
        for batch in loader:
            dec, _ = forward_step(model, batch, cfg, oracle, device)
            pred = torch.sigmoid(dec.saliency_logits).float().cpu().numpy()[:, 0]
            gt = batch["mask"].float().cpu().numpy()[:, 0]
            for i in range(pred.shape[0]):
                metric.step(pred[i], gt[i])
    return metric.get_results()


def build_optimizer(model, cfg):
    backbone, rest = [], []
    for n, p in model.named_parameters():
        (backbone if n.startswith("backbone") else rest).append(p)
    return torch.optim.AdamW([
        {"params": backbone, "lr": cfg.opt.backbone_lr},
        {"params": rest, "lr": cfg.opt.new_module_lr},
    ], weight_decay=cfg.opt.weight_decay)


def lr_lambda_factory(total_steps, warmup_ratio):
    warmup = max(1, int(warmup_ratio * total_steps))
    def fn(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--no_push", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"config loaded | device {device} | epochs {args.epochs}"
          f"{' | DRY RUN' if args.dry_run else ''}", flush=True)

    print("building model (first run downloads the pretrained backbone weights) ...",
          flush=True)
    model = build_model(cfg).to(device)
    print(f"model built: {sum(q.numel() for q in model.parameters()):,} params", flush=True)

    print("building dataloaders (scene-grouped split) ...", flush=True)
    train_loader, val_loader, _, _ = get_dataloaders(
        root_dir=cfg.data.dataset_root, image_size=384,
        batch_size=cfg.opt.batch_per_gpu, num_workers=cfg.train.num_workers,
        distributed=False)
    print(f"loaders ready: {len(train_loader)} train batches, "
          f"{len(val_loader)} val batches", flush=True)

    loss_fn = CombinedLoss(cfg.loss)
    opt = build_optimizer(model, cfg)
    # In a dry run, step every batch so 5 batches actually exercise the optimizer.
    accum = 1 if args.dry_run else max(1, cfg.opt.grad_accum_steps)
    steps_per_epoch = max(1, len(train_loader) // accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(steps_per_epoch * args.epochs, cfg.opt.warmup_ratio))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.opt.amp and device.type == "cuda")

    start_epoch, best = 0, float("inf")
    latest = out / "latest.pth"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" not in ck:
            print("found a checkpoint without optimizer state (a dry-run or older "
                  "artefact) -- ignoring it and starting fresh", flush=True)
        else:
            model.load_state_dict(ck["model_state_dict"])       # plain keys (pre-oracle)
            opt.load_state_dict(ck["optimizer_state_dict"])
            sched.load_state_dict(ck["scheduler_state_dict"])
            start_epoch, best = ck["epoch"] + 1, ck.get("best_metric", float("inf"))
            print(f"resumed from epoch {start_epoch}", flush=True)

    # Install the oracle AFTER any resume load.  The wrapper renames the router keys
    # (router_mlp -> router_mlp.orig), so a wrapped model cannot load a plain
    # checkpoint, and a checkpoint saved while wrapped would carry the wrapper in its
    # keys (which is what made the earlier save/load round-trip fail).
    install_oracle(model, cfg.model.num_experts)

    def save(epoch, metric, name):
        # Strip the oracle wrapper so the saved keys are the plain router_mlp.* names
        # every loader in this repo expects, and keep optimizer/scheduler state so a
        # restart resumes exactly.
        sd = model.state_dict()
        sd = {k.replace("router_mlp.orig.", "router_mlp."): v for k, v in sd.items()}
        payload = {"model_state_dict": sd, "optimizer_state_dict": opt.state_dict(),
                   "scheduler_state_dict": sched.state_dict(),
                   "epoch": epoch, "best_metric": metric, "config": asdict(cfg)}
        torch.save(payload, out / name)
        print(f"  wrote {out / name}", flush=True)
        if not args.no_push and os.environ.get("HF_TOKEN"):
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                repo_id=os.environ.get("HF_REPO", "Avi2006/spatial-moe-results"),
                path_or_fileobj=str(out / name),
                path_in_repo=f"checkpoints/{cfg.experiment_id or out.name}/{name}",
                repo_type="dataset", token=os.environ["HF_TOKEN"])
            print(f"  pushed checkpoints/{cfg.experiment_id or out.name}/{name}", flush=True)

    max_steps = args.dry_run if args.dry_run else 10 ** 9
    for epoch in range(start_epoch, args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            if step >= max_steps:
                break
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                dec, moe = forward_step(model, batch, cfg, True, device)
                total, comps = loss_fn(
                    dec.saliency_logits, dec.boundary_logits, moe,
                    batch["mask"].to(device, non_blocking=True),
                    gt_boundary=batch["edge_map"].to(device, non_blocking=True),
                    aux_logits_16=dec.aux_logits_16, aux_logits_8=dec.aux_logits_8,
                    aux_logits_4=dec.aux_logits_4,
                    pad_mask=batch["pad_mask"].to(device, non_blocking=True))
                loss = total / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.opt.gradient_clipping)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if args.dry_run:
                print(f"  step {step}: loss {float(total):.4f}", flush=True)
        if args.dry_run:
            print("dry run ok: forward, loss, backward and an optimizer step all ran; "
                  "no validation, no checkpoint written", flush=True)
            return
        print("validating ...", flush=True)
        metrics = evaluate(model, val_loader, cfg, True, device)
        mae = metrics["MAE"]
        improved = mae < best
        best = min(best, mae)
        save(epoch, best, "latest.pth")
        if improved:
            save(epoch, best, "best.pth")
        print(f"epoch {epoch}: val MAE {mae:.5f}  best {best:.5f}  "
              f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
'''

RUNNER_CELL = (
    'RUNNER = r"""\n' + RUNNER.strip("\n") + '\n"""\n'
    '(PROJECT_ROOT / "oracle_train.py").write_text(RUNNER)\n'
    'print("wrote", PROJECT_ROOT / "oracle_train.py", f"({len(RUNNER.splitlines())} lines)")\n'
)

CELLS = [
MD(r"""
# Oracle routing — the positive control

**Review item 3.** *"Does the architecture help at all if it is handed the answer?"*

Every other result in the paper is a null. This is the one run that can change what the null
*means*: if hard-routing each image to a weather-specific expert makes the saliency markedly
better, then the experts are useful and only the *selection* failed; if it does not, the
null is about the architecture, not the router.

**How the oracle works.** `SpatialMoELayer` has no per-image routing hook, but it routes
through a `router_mlp` whose logits are then top-k'd. This notebook installs the oracle at
runtime by wrapping that MLP (`OracleRouter`) and adding a `+1e4` bias to the expert chosen by
the image's weather label — applied to **all three scales**. That keeps every hard part of
training in the tested code paths (`src.model`, `src.dataset`, `src.loss`, `src.metrics`) and
needs no `src/` edit and no code-zip repackaging.

| | |
|---|---|
| routing | image's weather label → fixed expert at each scale (`+1e4` logit bias) |
| recipe | E8 k2, 8 epochs, AdamW, warmup-cosine, AMP, seed 42 (the reference family) |
| router aux losses | off — with a near one-hot gate their gradients are degenerate |
| compare | test-real MAE with the oracle on vs the same checkpoint with the oracle off |

**Nothing is lost.** The runner pushes `latest.pth`/`best.pth` to the Hub **after every epoch**
and resumes from `latest.pth` if it is already there, so a restart continues.

**Read the comparison, not the absolute MAE.** Both columns come from the same checkpoint and
the same model-frame scoring; the difference is the only quantity that answers the question.
"""),

MD("## 1 · Setup"),

CO(r"""
!pip install -q -U "huggingface_hub>=0.23" gdown pysodmetrics timm albumentations 2>/dev/null | tail -1
"""),

CO(r"""
import glob as _glob
import importlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

import torch

from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
REPO = "Avi2006/spatial-moe-results"
os.environ["HF_REPO"] = REPO
api = HfApi(token=HF_TOKEN)
api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=True)

WORK = Path("/kaggle/working")
PROJECT_ROOT = WORK / "spatial_moe_sod"
CKPT_ROOT = WORK / "WXSOD_Checkpoints"
CKPT_ROOT.mkdir(exist_ok=True)
print("hub:", REPO, "| gpus:", torch.cuda.device_count())


def run_streaming(cmd, log_path):
    '''Run cmd with LIVE output: streams to the notebook and tees to log_path.'''
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        proc.wait()
    return proc.returncode
"""),

MD("## 2 · Source and data"),

CO(r"""
if not (PROJECT_ROOT / "src" / "model.py").exists():
    print("fetching source ...")
    _z = hf_hub_download(REPO, "code/spatial_moe_sod_code.zip",
                         repo_type="dataset", token=HF_TOKEN)
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_z) as _zf:
        _zf.extractall(PROJECT_ROOT)
    if not (PROJECT_ROOT / "src").exists():
        _h = [_q for _q in PROJECT_ROOT.rglob("src") if (_q / "model.py").exists()]
        if _h:
            shutil.copytree(_h[0].parent, WORK / "_src_tmp")
            shutil.rmtree(PROJECT_ROOT)
            shutil.move(str(WORK / "_src_tmp"), str(PROJECT_ROOT))

os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

_failed = []
for _m in ("src.config", "src.model", "src.dataset", "src.loss", "src.metrics"):
    try:
        importlib.import_module(_m)
        print(f"  ok    import {_m}")
    except Exception as _e:
        _failed.append(_m)
        print(f"  FAIL  import {_m}: {type(_e).__name__}: {_e}")
assert not _failed, f"fix these imports first (pip: pysodmetrics, timm, albumentations)"
print("source root:", PROJECT_ROOT)
"""),

CO(r"""
DATA_FILE_ID = "1SUtAaxOy3c8Vzb5Ojp4NzxRB1vmsTjHL"
EXTRACT_PATH = WORK / "WXSDO_data"


def is_dataset_valid(path):
    def check(r):
        return all(os.path.isdir(os.path.join(r, f"{s}/{p}"))
                   for s in ("train_sys", "test_sys", "test_real")
                   for p in ("input", "gt"))
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
    DATA_ROOT = is_dataset_valid(_c)
    if DATA_ROOT:
        break
if DATA_ROOT is None:
    DATA_ROOT = is_dataset_valid(EXTRACT_PATH)
if DATA_ROOT is None:
    _zip = WORK / "dataset.zip"
    if not _zip.exists():
        os.system("pip install -q gdown 2>/dev/null")
        os.system(f"gdown {DATA_FILE_ID} -O {_zip}")
    EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
    os.system(f"unzip -q -o {_zip} -d {EXTRACT_PATH}")
    DATA_ROOT = is_dataset_valid(EXTRACT_PATH)
assert DATA_ROOT is not None, "no WXSOD ground truth found"
print("DATA_ROOT =", DATA_ROOT)
"""),

MD(r"""
## 3 · The oracle runner

Written to `oracle_train.py` inside the checkout. It reuses `src.model`, `src.dataset`,
`src.loss` and `src.metrics`; the only new logic is the `OracleRouter` wrapper and a plain
training loop. The loop pushes a checkpoint to the Hub after every epoch and resumes from
`latest.pth` when one is present.
"""),

CO(RUNNER_CELL),

CO(r"""
# compile check + the exact weather->expert table this run will use
import py_compile
py_compile.compile(str(PROJECT_ROOT / "oracle_train.py"), doraise=True)
sys.path.insert(0, str(PROJECT_ROOT))
import oracle_train as OT
print("oracle router imports ok")
print("weather -> expert:", {w: i % 8 for i, w in enumerate(sorted(OT.WEATHERS))})
"""),

MD(r"""
## 4 · The config

Same family as the reference arm (E8 k2, 8 epochs, seed 42). The routing auxiliary losses
(`load_balance_weight`, `importance_weight`) are set to `0.0`: with a near one-hot oracle
gate their gradients are degenerate and they would only add noise to a control run.
"""),

CO(r"""
import copy

BASE = {
    "data": {"dataset_root": str(DATA_ROOT), "split_manifest_path": "",
             "split_manifest_hash": "", "max_samples": None},
    "model": {"backbone": "pvt_v2_b4", "working_dim": 256, "decoder_dim": 256,
              "num_experts": 8, "top_k": 2, "router_variant": "token_only",
              "expert_variant": "standard", "attention_config": "default",
              "window_size": 7, "deep_supervision": False, "moe_type": "sparse",
              "moe_16_mode": "sparse", "router_noise_enabled": True,
              "router_noise_scale": 1.0, "router_noise_min_std": 0.05,
              "gate_mode": "renormalized"},
    "loss": {"bce_weight": 1.0, "iou_weight": 1.0, "ssim_weight": 1.0,
             "boundary_weight": 1.5, "load_balance_weight": 0.0,
             "load_balance_weights": [0.016, 0.03, 0.02], "importance_weight": 0.0,
             "z_loss_weight": 0.0, "aux_boundary_weight": 0.5,
             "deep_supervision_weight": 0.0, "entropy_confidence_weight": 0.0},
    "opt": {"optimizer": "AdamW", "backbone_lr": 0.0001, "new_module_lr": 0.0001,
            "weight_decay": 0.0001, "warmup_ratio": 0.04, "scheduler": "WarmupCosine",
            "gradient_clipping": 1.0, "amp": True, "batch_per_gpu": 4,
            "grad_accum_steps": 5, "effective_global_batch": 40},
    "train": {"epochs": 8, "freeze_backbone_epochs": 1, "seed": 42, "num_workers": 2,
              "checkpoint_every_n_steps": 200},
    "eval": {"validation_metric_selection": "MAE", "tta_mode": "none",
             "metric_config": "default"},
    "diag": {"routing_diagnostic_epochs": 1, "expert_ablation_enabled": False},
    "experiment_id": "", "variant": "ORACLE",
}

CFG_DIR = WORK / "oracle_configs"
CFG_DIR.mkdir(exist_ok=True)
CFG_PATH = CFG_DIR / "oracle.json"
CFG_PATH.write_text(json.dumps(BASE, indent=2))

from src.config import ExperimentConfig
cfg = ExperimentConfig.load(str(CFG_PATH))
print("epochs:", cfg.train.epochs, " E:", cfg.model.num_experts, " k:", cfg.model.top_k,
      " lb:", cfg.loss.load_balance_weight, " imp:", cfg.loss.importance_weight)
print("cfg ok")
"""),

CO(r"""
import re, subprocess

# derive the experiment id the same way the trainer does, so Hub paths match the family
from src.experiment import generate_experiment_id
EXP_ID = generate_experiment_id(cfg)
print("EXP_ID:", EXP_ID)

CFG_DIR_STAMP = CFG_DIR / "oracle.json"
OUT_DIR = CKPT_ROOT / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("checkpoints ->", OUT_DIR)
"""),

MD("## 5 · Dry run (forward, loss, backward, one optimizer step — 5 batches)"),

CO(r"""
cmd = [sys.executable, "-u", "oracle_train.py",
       "--config", str(CFG_PATH), "--out_dir", str(OUT_DIR), "--dry_run", "5",
       "--no_push"]
print(" ".join(cmd), "\n")
# Output streams live. The runner prints its phase banners (model build, dataloader,
# step, validation) as it goes, so silence here means it is loading, not hung.
rc = run_streaming(cmd, WORK / "oracle_dryrun.log")
print(f"\nexit={rc}")
if rc != 0:
    lines = (WORK / "oracle_dryrun.log").read_text(errors="replace").splitlines()
    keys = ("Traceback", "Error", "error:", "raise ", "assert")
    hits = [i for i, l in enumerate(lines) if any(k in l for k in keys)]
    k0 = max(0, (hits[0] - 3) if hits else len(lines) - 45)
    print("\n".join(lines[k0:k0 + 40]))
"""),

MD("## 6 · Train (resumes automatically)"),

CO(r"""
# The runner pushes latest.pth / best.pth to the Hub every epoch and resumes from
# latest.pth, so this cell is safe to re-run after a restart. Output streams live.
cmd = [sys.executable, "-u", "oracle_train.py",
       "--config", str(CFG_PATH), "--out_dir", str(OUT_DIR), "--epochs", "8"]
print(" ".join(cmd), "\n")
rc = run_streaming(cmd, WORK / "oracle_train.log")
print(f"\nexit={rc}")
"""),

MD(r"""
## 7 · The comparison — oracle on vs off

Same checkpoint, same images. The oracle column forces the weather expert; the learned column
is the model's own router. **The difference is the positive control.**
"""),

CO(r"""
sys.path.insert(0, str(PROJECT_ROOT))
import oracle_train as OT
from src.config import ExperimentConfig
from src.dataset import get_dataloaders

cfg = ExperimentConfig.load(str(CFG_PATH))
ckpt = OUT_DIR / "best.pth"
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
model = OT.build_model(cfg)
# Plain router_mlp.* keys (the runner strips the oracle wrapper on save); the
# .replace is a no-op on clean checkpoints and rescues any older wrapped one.
_sd = {k.replace("router_mlp.orig.", "router_mlp."): v
       for k, v in ck["model_state_dict"].items()}
model.load_state_dict(_sd)
model.eval()
OT.install_oracle(model, cfg.model.num_experts)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
_, _, _, real_loader = get_dataloaders(
    root_dir=str(DATA_ROOT), batch_size=8, num_workers=2, distributed=False)

import json
res = {
    "experiment_id": EXP_ID, "checkpoint": f"checkpoints/{EXP_ID}/best.pth",
    "epoch": ck.get("epoch"), "test_real": {
        "oracle_routing_mae": OT.evaluate(model, real_loader, cfg, True, DEVICE)["MAE"],
        "learned_routing_mae": OT.evaluate(model, real_loader, cfg, False, DEVICE)["MAE"],
    },
}
res["test_real"]["delta"] = res["test_real"]["oracle_routing_mae"] - \
                            res["test_real"]["learned_routing_mae"]
print(json.dumps(res, indent=2, default=float))
"""),

MD("## 8 · Upload"),

CO(r"""
from datetime import datetime, timezone

DST = f"analysis_results/{EXP_ID}"
payload = dict(res, generated_utc=datetime.now(timezone.utc).isoformat(),
               note="model-frame MAE; oracle and learned scored identically. "
                    "delta < 0 means the oracle helped.")
f = WORK / "oracle_routing_summary.json"
f.write_text(json.dumps(payload, indent=2, default=float))
api.upload_file(repo_id=REPO, path_or_fileobj=str(f),
                path_in_repo=f"{DST}/oracle_routing_summary.json",
                repo_type="dataset", token=HF_TOKEN)
for c in sorted(OUT_DIR.glob("*.pth")):
    api.upload_file(repo_id=REPO, path_or_fileobj=str(c),
                    path_in_repo=f"checkpoints/{EXP_ID}/{c.name}",
                    repo_type="dataset", token=HF_TOKEN)
    print("uploaded", f"checkpoints/{EXP_ID}/{c.name}")
api.upload_file(repo_id=REPO, path_or_fileobj=str(CFG_PATH),
                path_in_repo=f"{DST}/oracle_config.json",
                repo_type="dataset", token=HF_TOKEN)
print("uploaded", f"{DST}/oracle_routing_summary.json and oracle_config.json")
print("done.")
"""),

MD(r"""
## 9 · How to read the result

* **oracle MAE well below learned MAE** — the experts *can* help when selection is correct;
  the paper's null becomes a statement about *routing*, not about the architecture.
* **oracle MAE ≈ learned MAE** — even being handed the weather per image does not help with
  this expert pool and recipe; that is a stronger statement than "the router was inert", and
  Section 5 can say so with evidence.

Either way the checkpoint and both numbers are on the Hub, keyed by `EXP_ID`, so the paper
table can cite `analysis_results/<EXP_ID>/oracle_routing_summary.json`.
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
