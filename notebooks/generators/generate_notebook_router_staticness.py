"""Generate the router-staticness forensics notebook (review items 1 & 2).

No training: reads one checkpoint already on the Hub and rebuilds the model at the
same init seed.

Measures three things:
  A. router parameter displacement  init -> trained   (is the gate a different gate?)
  B. top-2 assignment agreement      init vs trained   (did the ranking move?)
  C. mean-logit staticness                             (does routing change the output?)

Each phase writes its own JSON to the Hub and is skipped if that JSON is already there,
so a re-run resumes.

Run:  python3 generate_notebook_router_staticness.py
Writes: notebooks/statistics/moe-of-sod__router_staticness.ipynb
"""
import json, pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "statistics" / \
      "moe-of-sod__router_staticness.ipynb"

MD = lambda s: ("md", s.strip("\n"))
CO = lambda s: ("code", s.strip("\n"))

CELLS = [
MD(r"""
# Router staticness — did the gate ever move?

**No training.** Review items 1 & 2. Runs in a few minutes, CPU is fine.

The paper says the router is inert. The strongest form of that claim is *"the gate never
trained"*, and the current draft argues it from a near-flat entropy — which, as the paper
itself concedes, is not the right quantity (top-2 selection depends on the *ranking*, not
the magnitude). This notebook measures the ranking directly, in three ways:

| | measurement | reads |
|---|---|---|
| **A** | **Router displacement** — `‖θ_trained − θ_init‖ / ‖θ_init‖` and cosine, per scale | checkpoint vs a fresh model at the checkpoint's own seed |
| **B** | **Top-2 agreement** — exact-set match and mean overlap of the top-2 experts on identical real-test tokens | init model vs trained model, same images |
| **C** | **Mean-logit staticness** — replace every token's router logits with the dataset-mean logits and measure MAE | trained model, patched vs unpatched |

**Chance baselines for B.** With `E=8`, `K=2` there are `C(8,2)=28` pairs, so a random exact
top-2 match happens `1/28 ≈ 3.6%` of the time, and random overlap averages `K²/E = 0.5`
experts. Agreement far above that means the ranking barely moved.

**Reading C.** If routing is a static selector, the *choice* of expert does not depend on the
token; replacing it with a constant should leave the MAE unchanged.

Everything is uploaded to `analysis_results/router_staticness/` on the Hub, one JSON per
phase, and each phase is skipped if its JSON is already there (so a re-run resumes).
"""),

MD("## 1 · Setup"),

CO(r"""
!pip install -q -U "huggingface_hub>=0.23" gdown pysodmetrics 2>/dev/null | tail -1
"""),

CO(r"""
import glob as _glob
import importlib, json, math, os, random, shutil, sys, zipfile
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
DST = "analysis_results/router_staticness"
print("hub:", REPO, "| device:", DEVICE)
"""),

CO(r"""
def load_done(name):
    '''Return this phase's previous JSON from the Hub, or None (resume guard).'''
    try:
        p = hf_hub_download(REPO, f"{DST}/{name}.json", repo_type="dataset", token=HF_TOKEN)
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def save_upload(name, payload):
    '''Write the phase JSON locally and push it; called at the end of each phase.'''
    payload = dict(payload, generated_utc=datetime.now(timezone.utc).isoformat())
    f = WORK / f"router_staticness_{name}.json"
    f.write_text(json.dumps(payload, indent=2, default=float))
    api.upload_file(repo_id=REPO, path_or_fileobj=str(f),
                    path_in_repo=f"{DST}/{name}.json",
                    repo_type="dataset", token=HF_TOKEN)
    print("uploaded", f"{DST}/{name}.json")
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
for _m in ("src.config", "src.model", "src.dataset", "src.metrics"):
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
"""),

MD(r"""
## 3 · The checkpoint (resolved from the Hub)

The path is **not hard-coded**. The cell lists `checkpoints/` on the Hub, prefers the reference
arm (`EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO`), and falls back to `checkpoints/best.pth` — the
reported model, and the same file the companion forensics notebook loads. It prints every
`.pth` it found, so a missing arm is visible instead of a 404.

The "init" model is a fresh build at the checkpoint's *own* seed, so it is the state the router
started from.
"""),

CO(r"""
from src.model import SpatialMoESODNet

# THE CHECKPOINT IS RESOLVED, NOT HARD-CODED.  Verified against the live Hub layout:
# the trainer stores each run FLAT as checkpoints/<EXP_ID>_best.pth (accountA/B pull
# f"{experiment_id}_best.pth" the same way); only the two confidence arms use a
# checkpoints/<EXP_ID>/best.pth subfolder.  checkpoints/best.pth is the reported model.
# We list the Hub first, prefer the reference arm, and print everything found.
EXP_ID = "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO"
_f = api.list_repo_files(REPO, repo_type="dataset")
_found = sorted(f for f in _f if f.startswith("checkpoints/") and f.endswith(".pth"))
print("checkpoints on the Hub:")
for _p in _found:
    print("   ", _p)

CANDIDATES = [f"checkpoints/{EXP_ID}_best.pth",     # reference arm (verified present)
              f"checkpoints/{EXP_ID}/best.pth",     # subfolder form (confidence arms)
              "checkpoints/best.pth"]               # the reported model
CKPT_REL = next((c for c in CANDIDATES if c in _f), None)
assert CKPT_REL, (f"none of {CANDIDATES} exist on the Hub. Available .pth files: {_found}")
MODEL_LABEL = CKPT_REL[len("checkpoints/"):].replace("/best.pth", "").replace("_best.pth", "")
print(f"\nusing {CKPT_REL}  ->  label {MODEL_LABEL}")

CKPT = hf_hub_download(REPO, CKPT_REL, repo_type="dataset", token=HF_TOKEN)
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
CFG = _ck.get("config", {})
MCFG = CFG.get("model", {})
SEED = int(CFG.get("train", {}).get("seed", 42))
assert "num_experts" in MCFG, "checkpoint config missing num_experts"
E = int(MCFG.get("num_experts", 8)); K = int(MCFG.get("top_k", 2))
print(f"{MODEL_LABEL}: epoch {_ck.get('epoch','?')}  seed {SEED}  E={E} k={K} "
      f"gate={MCFG.get('gate_mode')}")


def build_model(mcfg, seed=None):
    '''Architecture-defining fields all come from the checkpoint config.'''
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return SpatialMoESODNet(
        dim=mcfg.get("working_dim", 256),
        num_experts=mcfg.get("num_experts", 8),
        k=mcfg.get("top_k", 2),
        gate_mode=mcfg.get("gate_mode", "renormalized"),
        window_size=mcfg.get("window_size", 7),
        moe_type=mcfg.get("moe_type", "sparse"),
        moe_16_mode=mcfg.get("moe_16_mode", "sparse"),
        use_deep_supervision=mcfg.get("deep_supervision", False),
        router_noise_enabled=mcfg.get("router_noise_enabled", True),
        router_noise_scale=mcfg.get("router_noise_scale", 1.0),
        router_noise_min_std=mcfg.get("router_noise_min_std", 0.05),
    )


TRAINED = build_model(MCFG)
TRAINED.load_state_dict(_ck["model_state_dict"])
TRAINED.eval().to(DEVICE)
INIT = build_model(MCFG, seed=SEED).eval().to(DEVICE)
for _m in (TRAINED, INIT):
    for _p in _m.parameters():
        _p.requires_grad_(False)
print("models ready: trained + fresh-at-seed")

from src.dataset import get_dataloaders

_, _, _, test_real_loader = get_dataloaders(
    root_dir=str(DATA_ROOT), batch_size=8, num_workers=2, distributed=False)
MAX_BATCHES = 20          # 20 x 8 = 160 real-test images; raise for tighter error bars

SCALES = ("moe_16", "moe_8", "moe_4")
LAYERS = (lambda m: m.moe_16, lambda m: m.moe_8, lambda m: m.moe_4)
"""),

MD("## 4 · Phase A — did the router parameters move?"),

CO(r"""
disp = None
_done = load_done("A_router_displacement")
if _done:
    disp = _done["per_scale"]
    print("phase A already on the Hub — loaded, skipping compute")
else:
    def router_groups(model):
        out = defaultdict(list)
        for n, p in model.named_parameters():
            if any(t in n for t in ("router_mlp", "router_dwconv", "router_noise")):
                out[n.split(".")[0]].append(p.detach().float().cpu().reshape(-1))
        return {k: torch.cat(v) for k, v in out.items()}

    G0, G1 = router_groups(INIT), router_groups(TRAINED)
    disp = {}
    for scale in sorted(G0):
        f0, f1 = G0[scale], G1[scale]
        d = (f1 - f0).norm().item()
        disp[scale] = {
            "n_params": int(f0.numel()),
            "relative_displacement": d / (f0.norm().item() + 1e-12),
            "cosine": torch.nn.functional.cosine_similarity(f0, f1, dim=0).item(),
        }
    save_upload("A_router_displacement", {
        "checkpoint": CKPT_REL, "model_label": MODEL_LABEL, "seed": SEED, "per_scale": disp,
        "note": "relative_displacement = ||theta_trained - theta_init|| / ||theta_init||; "
                "0 = the router did not move."})

for _s in sorted(disp):
    print(f"  {_s}: rel.displacement={disp[_s]['relative_displacement']:.4f}  "
          f"cosine={disp[_s]['cosine']:.4f}  ({disp[_s]['n_params']:,} params)")
print("\nreference: an untouched router has relative_displacement 0.0000, cosine 1.0000")
"""),

MD("## 5 · Phase B — top-2 agreement on identical tokens"),

CO(r"""
agree = None
_done = load_done("B_top2_agreement")
if _done:
    agree = _done["per_scale"]
    print("phase B already on the Hub — loaded, skipping compute")
else:
    def topk_sets(model, loader, max_batches):
        acc = defaultdict(list)
        with torch.no_grad():
            for b, batch in enumerate(loader):
                if b >= max_batches:
                    break
                _dec, moe = model(batch["image"].to(DEVICE))
                for sname, mo in zip(SCALES, moe):
                    if mo.has_router:
                        acc[sname].append(mo.topk_indices.detach().cpu())
        return {k: torch.cat(v).reshape(-1, K) for k, v in acc.items()}

    S_INIT = topk_sets(INIT, test_real_loader, MAX_BATCHES)
    S_TRAIN = topk_sets(TRAINED, test_real_loader, MAX_BATCHES)

    agree = {}
    for sname in sorted(S_INIT):
        a, b = S_INIT[sname], S_TRAIN[sname]                       # [M, K]
        # top-k ids are distinct, so the sorted-match count IS the set overlap —
        # no .tolist()/set, which is what made the first version raise.
        same = torch.sort(a, -1).values == torch.sort(b, -1).values
        exact = same.all(-1).float().mean().item()
        overlap = same.sum(-1).float().mean().item()
        agree[sname] = {"n_tokens": int(a.shape[0]),
                        "exact_topk_match": exact, "mean_overlap": overlap,
                        "chance_exact_match": 1.0 / math.comb(E, K),
                        "chance_overlap": K * K / E}
    save_upload("B_top2_agreement", {
        "checkpoint": CKPT_REL, "model_label": MODEL_LABEL, "n_batches": MAX_BATCHES,
        "per_scale": agree,
        "note": "exact_topk_match above chance_exact_match (3.6% at E=8,K=2) means the "
                "ranking barely moved."})

for _s in sorted(agree):
    print(f"  {_s}: exact top-2 match={agree[_s]['exact_topk_match']:.4f} "
          f"(chance {agree[_s]['chance_exact_match']:.4f})  "
          f"overlap={agree[_s]['mean_overlap']:.3f} "
          f"(chance {agree[_s]['chance_overlap']:.3f})  "
          f"[{agree[_s]['n_tokens']:,} tokens]")
"""),

MD(r"""
## 6 · Phase C — is the router a static selector?

If the router's *choice* does not depend on the token, replacing every token's logits with the
dataset-mean logits should not change the prediction. We swap each MoE layer's `router_mlp` for a
module that returns the mean logits regardless of input, and compare MAE.

MAE is scored in the model's 384×384 frame (prediction vs the mask the loader returns); it is a
**relative** comparison between the patched and unpatched model on the same images, not an
absolute number for the paper. Noise is off in both (`.eval()`).
"""),

CO(r"""
import torch.nn as nn

static = None
_done = load_done("C_meanlogit_staticness")
if _done:
    static = _done
    print("phase C already on the Hub — loaded, skipping compute")
else:
    def mean_logits_for(model, loader, max_batches):
        acc, n = {}, 0
        with torch.no_grad():
            for b, batch in enumerate(loader):
                if b >= max_batches:
                    break
                _dec, moe = model(batch["image"].to(DEVICE))
                for sname, mo in zip(SCALES, moe):
                    if mo.has_router:
                        m = mo.clean_logits.mean(dim=(0, 2, 3)).detach().cpu()  # [E]
                        acc[sname] = acc.get(sname, 0) + m
                n += 1
        return {k: v / max(n, 1) for k, v in acc.items()}

    class _ConstRouter(nn.Module):
        def __init__(self, vec):
            super().__init__()
            self.register_buffer("vec", vec)
        def forward(self, x):                       # [B, N, 2C] -> [B, N, E]
            return self.vec.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)

    def score_mae(model, loader, max_batches):
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for b, batch in enumerate(loader):
                if b >= max_batches:
                    break
                out, _ = model(batch["image"].to(DEVICE))
                p = torch.sigmoid(out.saliency_logits)
                g = batch["mask"].to(DEVICE)
                tot += (p - g).abs().sum().item(); cnt += g.numel()
        return tot / max(cnt, 1)

    means = mean_logits_for(TRAINED, test_real_loader, MAX_BATCHES)
    mae_full = score_mae(TRAINED, test_real_loader, MAX_BATCHES)

    saved = {}
    for sname, get in zip(SCALES, LAYERS):
        layer = get(TRAINED)
        if hasattr(layer, "router_mlp"):
            saved[sname] = layer.router_mlp
            layer.router_mlp = _ConstRouter(means[sname].to(DEVICE))
    mae_static = score_mae(TRAINED, test_real_loader, MAX_BATCHES)
    for sname, get in zip(SCALES, LAYERS):
        if sname in saved:
            get(TRAINED).router_mlp = saved[sname]

    static = {"mae_full": mae_full, "mae_mean_logit": mae_static,
              "delta": mae_static - mae_full}
    save_upload("C_meanlogit_staticness", dict(
        static, checkpoint=CKPT_REL, model_label=MODEL_LABEL, n_batches=MAX_BATCHES,
        per_scale_mean_logits=means,
        note="scored in the model 384 frame; relative comparison only."))

print(f"MAE full model        : {static['mae_full']:.5f}")
print(f"MAE mean-logit router : {static['mae_mean_logit']:.5f}")
print(f"delta                 : {static['delta']:+.5f}")
"""),

MD(r"""
## 7 · How to read the result

* **A** — a relative displacement near 0 and a cosine near 1 say the gate is a different gate in
  name only. This is the cleanest single number for *"the router never trained."*
* **B** — top-2 agreement far above the 3.6% chance level says the *ranking* did not move either,
  which is exactly what the entropy argument could not show.
* **C** — if the mean-logit MAE equals the full-model MAE, the expert *choice* carries no
  information: a static selector. If it drops, routing was doing something.

If all three point the same way, the paper's leading hypothesis stops being a hypothesis and
Section 5 can rest on measurements instead of ranked guesses.
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
