"""Generate the paired-bootstrap / TOST + router-forensics notebook (review items 1 and 2).



No training: everything here reads data already on the Hub plus one checkpoint.

Run:  python3 generate_notebook_bootstrap_and_router.py

Writes: notebooks/statistics/moe-of-sod__bootstrap_and_router_forensics.ipynb

"""

import json, pathlib



OUT = pathlib.Path(__file__).resolve().parents[1] / "statistics" / \
      "moe-of-sod__bootstrap_and_router_forensics.ipynb"



MD = lambda s: ("md", s.strip("\n"))

CO = lambda s: ("code", s.strip("\n"))



CELLS = [

MD(r"""

# Paired bootstrap / TOST + router-at-init forensics



**Review items 1 and 2. No training.** Runs in a few minutes on CPU or GPU.



This notebook answers two objections:



1. *"With a single seed and a stated ±0.0004 floor, 'no effect' is indistinguishable

   from 'effect under 0.0005'."* — We recompute **per-image** MAE from the saved

   prediction maps for every arm, then run a **paired bootstrap** and a **TOST**

   equivalence test against an explicit margin. This converts "no effect" from an

   assertion into a tested claim.



2. *"You call this collapse, but the router may never have left initialization."* —

   We measure routing entropy **at initialization** versus **after training**, the

   router's **gradient norm** relative to the experts and decoder, and the cosine

   similarity of the **branch difference** `E_a − E_b` (not the full outputs, which

   carry the shared residual `x` and therefore inflate the measured similarity).



Everything is uploaded to `figures/paper/` and `analysis_results/` on the Hub.

"""),



MD("## 1 · Setup"),



CO(r"""

# pysodmetrics is installed defensively: this notebook only needs src.model, but if

# any import path reaches src.metrics it would otherwise die on py_sod_metrics.

!pip install -q -U "huggingface_hub>=0.23" gdown pysodmetrics 2>/dev/null | tail -1

"""),



CO(r"""

import io, json, math, os, re, shutil, sys, warnings, zipfile

from pathlib import Path



import numpy as np

import torch

import torch.nn.functional as F



warnings.filterwarnings("ignore")

np.set_printoptions(precision=6, suppress=True)

torch.set_grad_enabled(False)



from kaggle_secrets import UserSecretsClient

from huggingface_hub import HfApi, hf_hub_download, snapshot_download



HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

REPO     = "Avi2006/spatial-moe-results"

api      = HfApi(token=HF_TOKEN)

print("hub ready:", REPO)

"""),



MD("## 2 · Configuration"),



CO(r"""

# ---------------------------------------------------------------- data

DATA_FILE_ID = "1SUtAaxOy3c8Vzb5Ojp4NzxRB1vmsTjHL"   # project archive, Google Drive

EXTRACT_PATH = Path("/kaggle/working/WXSDO_data")

SPLIT        = "test_real"              # the 554-image real-world split

GT_SIZE      = 256                      # GT and predictions are both read at this size



# ---------------------------------------------------------------- statistics

N_BOOT = 20000

MARGIN = 0.001          # equivalence margin, in MAE. Reviewer suggested 0.001.

RNG_SEED = 20261002



# ---------------------------------------------------------------- arms

PRED_SUBDIR = "proxy_ablations/full_model"      # 554 PNGs per arm, "{idx:04d}_{weather}.png"



REF = "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO"   # the headline model R



# every other arm we have per-image predictions for, with the paper's label

ARMS = {

    "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO":       "Ref (E8 k2)",

    "EXP_B4_E2_K2_S32_R1_L3_M_SPARSE":             "A2 (E2 k2)",

    "EXP_B4_E4_K1_S32_R1_L3_M_SPARSE":             "A3 (E4 k1)",

    "EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL":   "A5 (E4 k2, MoE off)",

    "EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL": "A6 (E4 k2, dense)",

    "EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB":        "A8 (E4 k2, no LB)",

    "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14":   "R@14ep",

    "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43":"Ref seed43",

    "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44":"Ref seed44",

    "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1":      "scale1",

    "EXP_B4_E2_K1_S40_R1_L3_M_SPARSE":             "E2 k1 S40",

    "EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE":   "E4 k2 S40 gd",

}

OUT_DIR = "analysis_results/bootstrap_tost"

print(f"{len(ARMS)} arms | margin {MARGIN} | {N_BOOT} bootstrap resamples")

"""),



MD("## 3 · Ground truth for `test_real`"),



CO(r"""

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



# 1. attached to the notebook as a Kaggle dataset?

for cand in _glob.glob("/kaggle/input/*") + _glob.glob("/kaggle/input/*/*"):

    found = is_dataset_valid(cand)

    if found:

        DATA_ROOT = found

        print("using attached dataset:", DATA_ROOT)

        break



# 2. already unpacked earlier in this session?

if DATA_ROOT is None:

    DATA_ROOT = is_dataset_valid(EXTRACT_PATH)

    if DATA_ROOT:

        print("using existing download:", DATA_ROOT)



# 3. fetch the project archive from Google Drive

if DATA_ROOT is None:

    print("dataset not found; fetching from Google Drive ...")

    ZIP_PATH = Path("/kaggle/working/dataset.zip")

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



CO(r"""

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")





def gt_path(stem, split=SPLIT):

    for ext in IMG_EXT:

        q = DATA_ROOT / split / "gt" / (stem + ext)

        if q.exists():

            return q

    return None





def input_path(stem, split=SPLIT):

    for ext in IMG_EXT:

        q = DATA_ROOT / split / "input" / (stem + ext)

        if q.exists():

            return q

    return None





# Files are named '{index:04d}_{weather}', e.g. '0126_fog'. The Hub's prediction maps

# use the same stem, so we match on the stem rather than relying on sorted index order.

stems = sorted(q.stem for q in (DATA_ROOT / SPLIT / "input").iterdir()

               if q.suffix.lower() in IMG_EXT)

paired = [(st, st.split("_", 1)[1] if "_" in st else "none", gt_path(st)) for st in stems]

paired = [(st, w, g) for st, w, g in paired if g is not None]



print(f"{SPLIT}: {len(stems)} inputs, {len(paired)} with ground truth")

print("weathers:", sorted({w for _, w, _ in paired}))

assert len(paired) >= 500, "expected ~554 test_real images"

"""),



CO(r"""

# Ground truth at 256x256, matching the evaluator's protocol.

import cv2



GT_SIZE = 256



def load_gt(path: Path) -> np.ndarray:

    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if g is None:

        raise IOError(path)

    g = cv2.resize(g, (GT_SIZE, GT_SIZE), interpolation=cv2.INTER_LINEAR)

    return (g.astype(np.float32) / 255.0).clip(0, 1)



gt = {st: load_gt(gp) for st, _w, gp in paired}



print("ground truth loaded:", len(gt), "masks")

print("positive-pixel fraction, mean:", np.mean([v.mean() for v in gt.values()]).round(4))

"""),



MD(r"""

## 4 · Per-image errors for every arm



The saved maps are 8-bit PNGs, so a recomputed MAE carries ~1/255 quantisation.

That noise is **common-mode across arms**, so paired *differences* are unaffected to

first order. We nevertheless check the recomputed aggregate against the published

JSON value below, so the size of the discrepancy is on the record.

"""),



CO(r"""

def load_pred(path: Path) -> np.ndarray:

    p = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if p is None:

        raise IOError(path)

    if p.shape != (GT_SIZE, GT_SIZE):

        p = cv2.resize(p, (GT_SIZE, GT_SIZE), interpolation=cv2.INTER_LINEAR)

    return (p.astype(np.float32) / 255.0).clip(0, 1)



# ---------------------------------------------------------------- resume hook

# The per-arm download + scoring loop below is the slow part of this notebook

# (~35 s x 12 arms, and the PNGs are re-fetched in a fresh Kaggle session).  Each run

# uploads its result to the Hub as soon as it exists, so a re-run can pick that up and

# skip straight to the statistics instead of starting over.

RESUME_FROM_HUB = True

NPZ_PATH = f"{OUT_DIR}/per_image_mae.npz"

PAIRED_STEMS = [st for st, _w, _gp in paired]



per_image = {}      # arm -> np.ndarray of per-image MAE, in `paired` order

missing   = {}



if RESUME_FROM_HUB:

    try:

        _npz = hf_hub_download(REPO, NPZ_PATH, repo_type="dataset", token=HF_TOKEN)

        _z = np.load(_npz)

        _saved = [str(q) for q in _z["__stems__"]] if "__stems__" in _z else None

        if _saved == PAIRED_STEMS:

            for _k in _z.files:

                if _k != "__stems__":

                    per_image[_k] = _z[_k]

            print(f"RESUMED: restored per-image errors for {len(per_image)} arms from "

                  f"{NPZ_PATH} (skipping the download loop)")

        else:

            print("cached per-image errors are for a different image set; recomputing")

    except Exception as _e:

        print(f"no usable cached copy ({type(_e).__name__}); computing from scratch")



for arm in ([] if per_image else ARMS):

    # download the whole prediction folder once, then index into it by stem

    cache = Path("/kaggle/working/preds") / arm

    if not cache.exists():

        try:

            snapshot_download(

                REPO, repo_type="dataset", token=HF_TOKEN,

                allow_patterns=[f"analysis_results/{arm}/{PRED_SUBDIR}/*.png"],

                local_dir="/kaggle/working/hubcache", local_dir_use_symlinks=False)

            src = Path("/kaggle/working/hubcache") / "analysis_results" / arm / PRED_SUBDIR

            cache.parent.mkdir(parents=True, exist_ok=True)

            shutil.copytree(src, cache)

        except Exception as e:

            missing[arm] = f"download failed: {type(e).__name__}: {e}"

            continue



    have = {q.stem: q for q in cache.glob("*.png")}

    errs, got = [], 0

    for stem, _w, _gp in paired:

        f = have.get(stem)

        if f is None:

            errs.append(np.nan)

        else:

            errs.append(float(np.abs(load_pred(f) - gt[stem]).mean()))

            got += 1

    per_image[arm] = np.array(errs, dtype=np.float64)

    print(f"  {ARMS[arm]:<26s} n={got:3d}/{len(paired)}  "

          f"MAE={np.nanmean(per_image[arm]):.6f}")



if missing:

    print("\nnot available:")

    for k, v in missing.items():

        print(f"  {k}: {v}")

print(f"\nloaded {len(per_image)} arms")



# ---------------------------------------------------------------- upload (part 0)

# The per-image errors are the expensive artefact and everything downstream needs

# them, so they go to the Hub before anything else can fail.  This is what makes a

# re-run cheap: the next run restores them and skips the loop above.

_save = dict(per_image)

_save["__stems__"] = np.array(PAIRED_STEMS)

np.savez_compressed("/kaggle/working/per_image_mae.npz", **_save)

api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=True)

api.upload_file(repo_id=REPO, path_or_fileobj="/kaggle/working/per_image_mae.npz",

                path_in_repo=NPZ_PATH, repo_type="dataset", token=HF_TOKEN)

print("uploaded", NPZ_PATH, "-> a re-run will skip the download loop")

"""),



MD("### 4.1 · Sanity check against the published JSON"),



CO(r"""

# If our per-image ordering disagreed with the evaluator's, the recomputed mean

# would be visibly wrong. Compare against metrics.json where we can fetch it.

from huggingface_hub import HfApi as _HfApi
_api = _HfApi(token=HF_TOKEN)


def find_metrics(arm):
    '''Locate this arm's published metrics file.

    The evaluator writes eval_results/best/<split>/<tta>/metrics_<split>_<stamp>.json.
    The timestamp makes the filename unpredictable, so list the directory and match
    rather than guessing a path -- guessing is why this table used to be all nan.
    '''
    for tta in ("hflip", "none"):
        try:
            tree = _api.list_repo_tree(
                REPO, path_in_repo=f"analysis_results/{arm}/eval_results/best/{SPLIT}/{tta}",
                repo_type="dataset")
            for e in tree:
                name = e.path.split("/")[-1]
                if name.startswith("metrics") and name.endswith(".json"):
                    return e.path, tta
        except Exception:
            continue
    return None, None


json_mae = {}
for arm in list(per_image):
    path, tta = find_metrics(arm)
    if path is None:
        json_mae[arm] = ("--", float("nan"))
        continue
    try:
        _p = hf_hub_download(REPO, path, repo_type="dataset", token=HF_TOKEN)
        _d = json.load(open(_p))
        _g = _d.get("global", _d)
        json_mae[arm] = (tta, float(_g.get("MAE", float("nan"))))
    except Exception as _e:
        json_mae[arm] = (f"err:{type(_e).__name__}", float("nan"))

print(f"{'arm':<28s} {'tta':>7s} {'recomputed':>12s} {'json':>12s} {'delta':>10s}")

for arm, arr in per_image.items():

    tta, j = json_mae.get(arm, ("--", float("nan")))

    r = float(np.nanmean(arr))

    d = (r - j) if not math.isnan(j) else float("nan")

    print(f"{ARMS[arm]:<28s} {tta:>7s} {r:12.6f} {j:12.6f} {d:+10.6f}")

print("\nIf the deltas are ~1e-5 the ordering agrees; if any row is wildly off, "

      "the alignment for that arm is wrong and it must be dropped from the test.")

"""),



MD(r"""

## 5 · Paired bootstrap and TOST



For each arm `a` we form the per-image difference `d_i = MAE_i(a) − MAE_i(Ref)` and ask

two questions:



* **Bootstrap CI** — resample image indices with replacement 20 000 times; if the 95%

  interval on the mean difference contains 0, the arms are not distinguishable at

  this sample size.

* **TOST** — test `H0: |μ_d| ≥ 0.001` against `H1: |μ_d| < 0.001`. Equivalence is

  declared when the TOST p-value is below 0.05. This is the test that licenses the

  word "no effect"; a non-significant difference alone does not.

"""),



CO(r"""

def paired_bootstrap(d: np.ndarray, n: int = N_BOOT, seed: int = RNG_SEED):

    '''Bootstrap the mean of a paired difference vector.'''

    d = d[np.isfinite(d)]

    if d.size < 30:

        return np.nan, np.nan, np.nan, d.size

    rng = np.random.default_rng(seed)

    draws = rng.integers(0, d.size, size=(n, d.size))

    boot = d[draws].mean(axis=1)

    lo, hi = np.percentile(boot, [2.5, 97.5])

    return float(d.mean()), float(lo), float(hi), int(d.size)





def tost(d: np.ndarray, margin: float = MARGIN, n: int = N_BOOT, seed: int = RNG_SEED):

    '''Two one-sided tests for equivalence within +/- margin.



    p = max(P(boot <= -margin), P(boot >= +margin)).  Reject H0 (declare

    equivalence) when p < 0.05.

    '''

    d = d[np.isfinite(d)]

    if d.size < 30:

        return np.nan

    rng = np.random.default_rng(seed + 1)

    draws = rng.integers(0, d.size, size=(n, d.size))

    boot = d[draws].mean(axis=1)

    p_lower = float(np.mean(boot <= -margin))

    p_upper = float(np.mean(boot >=  margin))

    return max(p_lower, p_upper), p_lower, p_upper

"""),



CO(r"""

assert REF in per_image, f"reference arm {REF} not available"

ref = per_image[REF]

rows = []



for arm, arr in per_image.items():

    if arm == REF:

        continue

    d = arr - ref

    mean_d, lo, hi, n = paired_bootstrap(d)

    p, pl, pu = tost(d)

    rows.append(dict(

        arm=arm, label=ARMS[arm], n=n,

        ref_mae=float(np.nanmean(ref)), arm_mae=float(np.nanmean(arr)),

        delta=mean_d, ci_lo=lo, ci_hi=hi,

        tost_p=p, equivalent=bool(p < 0.05) if np.isfinite(p) else False,

        ci_excludes_zero=bool(lo > 0 or hi < 0) if np.isfinite(lo) else False,

    ))



rows.sort(key=lambda r: abs(r["delta"]))

w = 30

print(f"{'arm':<{w}} {'Δ MAE':>10s} {'95% CI':>22s} {'TOST p':>9s}  verdict")

print("-" * (w + 46))

for r in rows:

    ci = f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]"

    v = ("EQUIVALENT to Ref" if r["equivalent"]

         else ("differs" if r["ci_excludes_zero"] else "inconclusive"))

    print(f"{r['label']:<{w}} {r['delta']:+10.6f} {ci:>22s} {r['tost_p']:9.4f}  {v}")

"""),



MD(r"""

### 5.1 · The seed-variance floor, measured the same way



`seed43` and `seed44` differ from `Ref` only in the training seed. Their per-image

differences give the honest noise scale that every other comparison should be read

against — computed on the *same* image set, with the *same* estimator, rather than

quoted as a single number.

"""),



CO(r"""

seed_arms = [a for a in ("EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43",

                         "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44")

             if a in per_image]



seed_stats = {}

for a in seed_arms:

    d = per_image[a] - ref

    m, lo, hi, n = paired_bootstrap(d)

    seed_stats[a] = dict(mean=m, lo=lo, hi=hi, n=n)

    print(f"{ARMS[a]:<14s} Δ={m:+.6f}  95% CI [{lo:+.6f}, {hi:+.6f}]  n={n}")



if seed_arms:

    floor = [abs(seed_stats[a]["mean"]) for a in seed_arms]

    lo_all = [seed_stats[a]["lo"] for a in seed_arms]

    hi_all = [seed_stats[a]["hi"] for a in seed_arms]

    print(f"\nseed floor (|mean| over {len(seed_arms)} arm(s)): "

          f"{np.mean(floor):.6f}   max {np.max(floor):.6f}")

    print(f"pooled seed CI width: "

          f"[{min(lo_all):+.6f}, {max(hi_all):+.6f}]")

    print("\nAny |delta| inside this band is seed noise, not an architecture effect.")

else:

    print("no seed arms downloaded")

"""),



CO(r"""

# ---------------------------------------------------------------- upload (part 1)

# Done HERE, before the forensics section, on purpose: the statistics above are the

# whole point of this notebook and they must not be lost if the checkpoint forensics

# below fails.  Re-uploading at the end is idempotent.

api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=True)



payload_stats = dict(

    generated_utc=__import__("datetime").datetime.utcnow().isoformat() + "Z",

    split=SPLIT, n_images=len(paired), margin=MARGIN, n_boot=N_BOOT,

    reference=REF, reference_mae=float(np.nanmean(per_image[REF])),

    arms=rows, seed_floor=seed_stats,

)

_f = Path("/kaggle/working/bootstrap_tost.json")

_f.write_text(json.dumps(payload_stats, indent=2, default=float))

api.upload_file(repo_id=REPO, path_or_fileobj=str(_f),

                path_in_repo=f"{OUT_DIR}/bootstrap_tost.json",

                repo_type="dataset", token=HF_TOKEN)

print("uploaded", f"{OUT_DIR}/bootstrap_tost.json")



np.savez_compressed("/kaggle/working/per_image_mae.npz",

                    __stems__=np.array(PAIRED_STEMS),

                    **{a: v for a, v in per_image.items()})

api.upload_file(repo_id=REPO, path_or_fileobj="/kaggle/working/per_image_mae.npz",

                path_in_repo=f"{OUT_DIR}/per_image_mae.npz",

                repo_type="dataset", token=HF_TOKEN)

print("uploaded", f"{OUT_DIR}/per_image_mae.npz")



# FIGDIR is created by the plotting section below this cell, so guard on it.

_figdir = Path("/kaggle/working/figs")

if _figdir.exists():

    for _fig in sorted(_figdir.glob("*")):

        api.upload_file(repo_id=REPO, path_or_fileobj=str(_fig),

                        path_in_repo=f"figures/paper/{_fig.name}",

                        repo_type="dataset", token=HF_TOKEN)

        print("uploaded", f"figures/paper/{_fig.name}")

else:

    print("no figures yet (they are drawn after the forensics section)")

print("statistics are safely on the Hub - the forensics section may fail from here.")

"""),



MD(r"""

## 6 · Router forensics: did it ever leave initialization?



The reviewer's hypothesis is that the router receives almost no task gradient —

it is reached only through `g_{i,k} · (E_a − E_b)` — so it stays at its

initialization, where the logits are small and routing entropy is already near

`log E`. If true, "collapse" is the wrong word and the paper's framing changes.



Three measurements:



1. **Entropy at initialization versus after training.** Build fresh models at several

   seeds and compare their routing entropy with the trained checkpoint's.

2. **Gradient norms.** On a few real batches, compare `‖∂L/∂θ_router‖` with the

   experts and the decoder.

3. **Branch-difference similarity.** The experts are residual, so the full outputs

   share `x` and their cosine is inflated. What matters for re-routing is

   `E_a − E_b`, which is the MLP branches alone.

"""),



MD("### 6.1 · Load the trained checkpoint"),



CO(r"""

# --- source first: the loader below imports src.model

ROOT_DIR = Path("/kaggle/working/spatial_moe_sod")

PROJECT_ROOT = ROOT_DIR

if not (PROJECT_ROOT / "src" / "model.py").exists():

    print("fetching source from the Hub ...")

    z = hf_hub_download(REPO, "code/spatial_moe_sod_code.zip",

                        repo_type="dataset", token=HF_TOKEN)

    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(z) as zf:

        zf.extractall(PROJECT_ROOT)

    if not (PROJECT_ROOT / "src").exists():

        hits = [q for q in PROJECT_ROOT.rglob("src") if (q / "model.py").exists()]

        if hits:

            shutil.copytree(hits[0].parent, ROOT_DIR / "_src_tmp")

            shutil.rmtree(PROJECT_ROOT)

            shutil.move(str(ROOT_DIR / "_src_tmp"), str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(str(PROJECT_ROOT))

print("source root:", PROJECT_ROOT, "| src/model.py:",

      (PROJECT_ROOT / "src" / "model.py").exists())

"""),



CO(r"""

# --- loader copied from accountA section 15.

# The checkpoint embeds its own config, so every architecture-defining field comes

# from `checkpoint["config"]["model"]`.  That matters most for use_deep_supervision:

# building with the wrong value silently drops (or invents) the aux heads, and the

# model is then not the one that was trained.

from src.model import SpatialMoESODNet





def load_model_from_checkpoint(ckpt_path, device):

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_cfg = checkpoint.get("config", {}).get("model", {})

    assert "num_experts" in model_cfg, "checkpoint config missing num_experts"

    model = SpatialMoESODNet(

        dim=model_cfg.get("working_dim", 256),

        num_experts=model_cfg.get("num_experts", 8),

        k=model_cfg.get("top_k", 2),

        gate_mode=model_cfg.get("gate_mode", "renormalized"),

        window_size=model_cfg.get("window_size", 7),

        # moe_type / moe_16_mode decide WHICH MODULES EXIST; omitting them is what

        # produced the Missing/Unexpected keys the reference notebook warns about.

        moe_type=model_cfg.get("moe_type", "sparse"),

        moe_16_mode=model_cfg.get("moe_16_mode", "sparse"),

        use_deep_supervision=model_cfg.get("deep_supervision", False),

    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print(f"Loaded {ckpt_path}")

    print(f"  epoch {checkpoint.get('epoch', '?')}  best_metric {checkpoint.get('best_metric', '?')}")

    return model, model_cfg, checkpoint





def build_fresh_model(seed=None):

    '''A randomly-initialised model with the checkpoint's architecture.



    Same constructor mapping as load_model_from_checkpoint, so the only difference

    from the trained model is the weights.  Used to read routing entropy at

    initialisation and compare it with the trained value.

    '''

    if seed is not None:

        torch.manual_seed(seed)

    return SpatialMoESODNet(

        dim=model_cfg.get("working_dim", 256),

        num_experts=model_cfg.get("num_experts", 8),

        k=model_cfg.get("top_k", 2),

        gate_mode=model_cfg.get("gate_mode", "renormalized"),

        window_size=model_cfg.get("window_size", 7),

        moe_type=model_cfg.get("moe_type", "sparse"),

        moe_16_mode=model_cfg.get("moe_16_mode", "sparse"),

        use_deep_supervision=model_cfg.get("deep_supervision", False),

    ).to(DEVICE).eval()





DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT = hf_hub_download(REPO, "checkpoints/best.pth", repo_type="dataset", token=HF_TOKEN)



model = model_cfg = checkpoint = None

try:

    model, model_cfg, checkpoint = load_model_from_checkpoint(CKPT, DEVICE)

    # The decisive deep-supervision evidence: what the checkpoint's own config says,

    # versus whether aux tensors are actually present in the state dict.

    sd_keys = checkpoint["model_state_dict"]

    n_aux = len([k for k in sd_keys if "aux_head" in k])

    print(f"\n  config.model.deep_supervision : {model_cfg.get('deep_supervision')}")

    print(f"  decoder.aux_head_* tensors    : {n_aux}")

    print(f"  parameters                    : {sum(q.numel() for q in model.parameters()):,}")

except Exception as exc:

    print(f"could not load the checkpoint: {type(exc).__name__}: {exc}")

    model = None

"""),



MD(r"""

### 6.1b · Forensics preconditions



Everything the forensics section needs is checked **here**, once, so a problem

surfaces immediately with a complete list instead of failing three cells later. This

section raises rather than continuing: the statistics are already on the Hub by this

point and the resume hook makes a re-run cheap, so a loud failure is strictly better

than a silent one.

"""),



CO(r"""

_problems = []



# 1. the model

if model is None:

    _problems.append("checkpoint did not load (see the message in 6.1)")



# 2. the architecture we assume for the reading

for _k, _default in (("num_experts", 8), ("deep_supervision", None)):

    if model is not None and _k not in model_cfg:

        _problems.append(f"checkpoint config is missing model.{_k}")



# 3. the routing head must be findable

# Discovered inline rather than via router_modules(): this gate runs BEFORE that
# helper is defined, and a gate that cannot run is worse than no gate.
_routers = ([(n, m) for n, m in model.named_modules()
             if n.endswith(".router_mlp") or n.endswith(".router_dwconv")]
            if model is not None else [])

if model is not None and not _routers:

    _problems.append(

        "no router modules matched -- expected names ending in .router_mlp / "

        ".router_dwconv; got: "

        + ", ".join(n for n, _ in list(model.named_modules())[:0]) or "none")



# 4. MoEOutput must actually expose the entropy accessor this section reads

if model is not None:

    try:

        with torch.no_grad():

            _x = torch.randn(1, 3, 384, 384).to(DEVICE)

            _, _moe = model(_x)

        _have = [hasattr(o, "entropy") for o in _moe]

        if not all(_have):

            _problems.append(f"MoEOutput.entropy missing at some scales: {_have}")

        else:

            print("  entropy accessor present at all "

                  f"{len(_moe)} scales")

    except Exception as _e:

        _problems.append(f"forward pass failed: {type(_e).__name__}: {_e}")



print(f"  model: {'loaded' if model is not None else 'MISSING'}")

print(f"  router modules found: {len(_routers)}")

for _n, _m in _routers[:6]:

    print(f"     {_n}  ({type(_m).__name__})")



if _problems:

    raise RuntimeError(

        "Forensics cannot run. The statistics are already uploaded, so nothing is "

        "lost -- fix these and re-run (the resume hook skips the download loop):\n  - "

        + "\n  - ".join(_problems))

print("  all forensics preconditions pass")

"""),



MD("### 6.2 · Is the router already at maximum entropy before training?"),



CO(r"""

def router_modules(m):

    '''The routing head is moe_{4,8,16}.router_mlp (a Sequential ending in a

    Linear that emits [B, H*W, E]).  There is no module literally named .router;

    router_dwconv feeds it and router_noise perturbs its logits.'''

    out = []

    for name, mod in m.named_modules():

        if name.endswith(".router_mlp") or name.endswith(".router_dwconv"):

            out.append((name, mod))

    return out





# The routing head emits [B, H*W, E] from moe_{4,8,16}.router_mlp; there is no module

# literally named ".router".  The entropy itself is read from MoEOutput.entropy.



if model is not None:

    print(f"{len(router_modules(model))} router modules; showing first few:")

    for n, mod in router_modules(model)[:8]:

        print(f"  {n:<44s} {type(mod).__name__}")

"""),



CO(r"""

def routing_entropy_of(m, n_batches=4, batch=4, seed=0, target_size=384):

    '''Mean routing entropy per scale, read off MoEOutput -- accountA section 17.



    MoEOutput.entropy is already the normalised routing entropy, so this needs no

    forward hooks and no logit reconstruction.  Random inputs are used so the reading

    is a property of the weights, which is what comparing init vs trained requires.

    '''

    m.eval()

    g = torch.Generator().manual_seed(seed)

    vals = {}

    with torch.no_grad():

        for b in range(n_batches):

            x = torch.randn(batch, 3, target_size, target_size, generator=g).to(DEVICE)

            try:

                _, moe_outputs = m(x)

            except Exception as exc:

                print("  forward failed:", type(exc).__name__, exc)

                return None

            for scale, out in zip(("scale_4", "scale_8", "scale_16"), moe_outputs):

                e = getattr(out, "entropy", None)

                if e is None:

                    continue

                vals.setdefault(scale, []).append(

                    float(e.mean()) if torch.is_tensor(e) else float(e))

    if not vals:

        return None

    E = int(model_cfg.get("num_experts", 8))

    ceiling = math.log(E)

    return [dict(module="moe_" + scale.split("_")[1], scale=scale, E=E,

                 entropy=sum(v) / len(v), ceiling=ceiling,

                 frac=(sum(v) / len(v)) / ceiling)

            for scale, v in sorted(vals.items())]





init_rows = []

# Fail-soft: the statistics are already on the Hub by this point, so a problem here

# must not abort the notebook.  Each cell in this section prints what went wrong and

# carries on.

for _s in (0, 1, 2, 3, 4):

    _fresh = build_fresh_model(seed=_s)

    _v = routing_entropy_of(_fresh, seed=_s)

    if not _v:

        del _fresh

        raise RuntimeError(

            f"routing_entropy_of returned nothing for an initialised model at seed "

            f"{_s}; the entropy accessor is not what this section assumes.")

    for _r in _v:

        init_rows.append(dict(seed=_s, **_r))

    del _fresh

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

print(f"collected {len(init_rows)} (seed, scale) entropy readings at init")

"""),



CO(r"""

trained_rows = (routing_entropy_of(model, seed=0) or []) if model is not None else []

if not init_rows and not trained_rows:

    print("No routing-entropy readings were collected (see the warnings above).")

    print("The forensics section cannot run; the uploaded statistics are unaffected.")

import collections



print("=== routing entropy: initialization vs trained ===\n")

print(f"{'scale':<26s} {'E':>3s} {'init mean':>11s} {'init frac':>10s} "

      f"{'trained':>10s} {'tr frac':>9s}")

print("-" * 76)



by_mod_init = collections.defaultdict(list)

for r in init_rows:

    by_mod_init[r["module"]].append(r["frac"])



tr = {r["module"]: r for r in trained_rows}

for name in sorted(by_mod_init, key=lambda s: -int(re.search(r"moe_(\d+)", s).group(1))

                   if re.search(r"moe_(\d+)", s) else 0):

    fr = by_mod_init[name]

    E = next(r["E"] for r in init_rows if r["module"] == name)

    t = tr.get(name)

    ts = f"{t['frac']:9.4f}" if t else "        -"

    tv = f"{t['entropy']:10.4f}" if t else "         -"

    print(f"{name:<26s} {E:3d} {np.mean(fr):11.4f} {np.mean(fr):10.4f}  {tv} {ts}")



print("\nReading: if init frac is already ~1.0 and trained frac is unchanged, the "

      "router did not move — 'collapse' is the wrong description and the paper's "

      "framing should change to 'the router never trained'.")

"""),



CO(r"""
# ---------------------------------------------------------------- upload (part 2)
# The init-vs-trained entropy table is the most important output of the forensics
# section, and it is produced BEFORE the gradient and branch measurements that
# follow.  Upload it here so a later failure cannot discard it.
if init_rows or trained_rows:
    _fore = dict(generated_utc=__import__("datetime").datetime.utcnow().isoformat() + "Z",
                 checkpoint="checkpoints/best.pth",
                 model_config=model_cfg,
                 routing_entropy_at_init=init_rows,
                 routing_entropy_trained=trained_rows)
    _ff = Path("/kaggle/working/router_entropy_init_vs_trained.json")
    _ff.write_text(json.dumps(_fore, indent=2, default=float))
    api.upload_file(repo_id=REPO, path_or_fileobj=str(_ff),
                    path_in_repo=f"{OUT_DIR}/router_entropy_init_vs_trained.json",
                    repo_type="dataset", token=HF_TOKEN)
    print("uploaded", f"{OUT_DIR}/router_entropy_init_vs_trained.json")
else:
    print("no entropy readings to upload")
"""),

MD("### 6.3 · Router gradient norm relative to experts and decoder"),



CO(r"""

if model is not None:

    model.train()          # routing noise is gated on self.training

    try:

        from src.loss import CombinedLoss

        from src.config import LossConfig

        crit = CombinedLoss(LossConfig(deep_supervision_weight=0.0,

                                       load_balance_weights=[0.08, 0.15, 0.1]))

    except Exception as e:

        print("could not build CombinedLoss:", type(e).__name__, e)

        crit = None



    if crit is not None:

      # cell 1 calls torch.set_grad_enabled(False) for the inference-only work

      # above; without this the backward pass has nothing to attach to.

      torch.set_grad_enabled(True)

      g = torch.Generator().manual_seed(0)

      x = torch.randn(4, 3, 384, 384, generator=g).to(DEVICE)

      y = (torch.rand(4, 1, 384, 384, generator=g) > 0.7).float().to(DEVICE)

      # model(x) -> (DecoderOutput, List[MoEOutput]); the criterion needs both.

      out, moe_outputs = model(x)

      loss, _components = crit(out.saliency_logits, out.boundary_logits,

                               moe_outputs, y)

      model.zero_grad(set_to_none=True)

      loss.backward()



      groups = collections.defaultdict(float)

      for n, p in model.named_parameters():

          if p.grad is None:

              continue

          gn = float(p.grad.detach().norm())

          if "router_mlp" in n or "router_dwconv" in n or "router_noise" in n:

              groups["router"] += gn ** 2

          elif "expert" in n:

              groups["experts"] += gn ** 2

          elif "decoder" in n:

              groups["decoder"] += gn ** 2

          elif "backbone" in n:

              groups["backbone"] += gn ** 2

          else:

              groups["other"] += gn ** 2



      print("total gradient norm by group (one synthetic batch, loss "

            f"{float(loss):.4f}):\n")

      tot = sum(groups.values()) ** 0.5 or 1.0

      for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):

          share = (v ** 0.5) / tot

          print(f"  {k:<10s} ‖g‖ = {v ** 0.5:12.6f}   {share * 100:6.2f}% of total")

      print("\nA router share near zero means the gate receives almost no task "

            "gradient — the reviewer's hypothesis.")

    model.eval()

"""),



MD(r"""

### 6.4 · Branch-difference similarity (the correct quantity for re-routing)



Experts are residual: `E_k(x) = x + B_k(x)`. The full outputs therefore share `x`,

and their cosine similarity is inflated by it. Re-routing swaps experts, so what

matters is `E_a − E_b = B_a(x) − B_b(x)` — the branches alone. We capture each expert

module's input and output with hooks and subtract.

"""),



CO(r"""

expert_mods = [(n, m) for n, m in model.named_modules()

               if n.count(".") == 1 and n.startswith("experts")] if model else []

if model is not None and not expert_mods:

    # experts may live under moe_*/experts.*

    expert_mods = [(n, m) for n, m in model.named_modules()

                   if re.search(r"experts?\.\d+$", n) or re.search(r"\.experts$", n)]

print(f"{len(expert_mods)} expert modules found")

for n, m in expert_mods[:12]:

    print("   ", n, type(m).__name__)

"""),



CO(r"""

def branch_cosines(m, n_batches=4, batch=4, seed=0):

    '''Cosine similarity of expert *branches* B_k = E_k(x) - x, per forward pass.'''

    mods = [(n, mm) for n, mm in m.named_modules()

            if re.search(r"(?:^|\.)experts?\.\d+$", n) or re.search(r"\.experts$", n)]

    if len(mods) < 2:

        return None, "fewer than two expert modules located"

    store = {}

    handles = []

    for name, mod in mods:

        def pre(_m, inp, name=name):

            store.setdefault(name, {})["in"] = inp[0].detach().float()

        def post(_m, inp, out, name=name):

            o = out[0] if isinstance(out, (tuple, list)) else out

            store.setdefault(name, {})["out"] = o.detach().float()

        handles.append(mod.register_forward_pre_hook(pre))

        handles.append(mod.register_forward_hook(post))



    g = torch.Generator().manual_seed(seed)

    x = torch.randn(batch, 3, 384, 384, generator=g).to(DEVICE)

    m.eval()

    with torch.no_grad():

        try:

            m(x)

        except Exception as e:

            for h in handles:

                h.remove()

            return None, f"forward failed: {e}"

    for h in handles:

        h.remove()



    full, br = {}, {}

    for name, d in store.items():

        if "in" not in d or "out" not in d:

            continue

        fi, fo = d["in"], d["out"]

        if fi.shape != fo.shape:

            continue

        # flatten spatial dims: [B,C,H,W] -> [B*H*W, C]

        full[name] = fo.reshape(-1, fo.shape[1])

        br[name] = (fo - fi).reshape(-1, fo.shape[1])



    def mean_pairwise_cos(d):

        ks = sorted(d)

        vals = []

        for i in range(len(ks)):

            for j in range(i + 1, len(ks)):

                a, b = d[ks[i]].flatten(), d[ks[j]].flatten()

                if a.numel() != b.numel():

                    continue

                na, nb = a.norm(), b.norm()

                if na > 0 and nb > 0:

                    vals.append(float((a @ b) / (na * nb)))

        return (float(np.mean(vals)) if vals else np.nan), len(ks), len(vals)



    # Experts at DIFFERENT scales have different token counts (9216 / 2304 / 576), so
    # a pairwise cosine between them is meaningless and previously raised.  Group by
    # scale, take the mean within each scale, then average those means.
    def _by_scale(d):
        g = {}
        for _k, _v in d.items():
            g.setdefault(_k.split(".")[0], {})[_k] = _v
        return g

    _fb, _bb = _by_scale(full), _by_scale(br)
    per_scale = {}
    for _s in sorted(_fb):
        _cf, _nk, _npair = mean_pairwise_cos(_fb[_s])
        _cb, _, _ = mean_pairwise_cos(_bb.get(_s, {}))
        per_scale[_s] = dict(cos_full=_cf, cos_branch=_cb,
                             n_experts=_nk, n_pairs=_npair)
    cf = float(np.mean([v["cos_full"] for v in per_scale.values()]))
    cb = float(np.mean([v["cos_branch"] for v in per_scale.values()]))
    nk = int(np.mean([v["n_experts"] for v in per_scale.values()]))
    npair = int(sum(v["n_pairs"] for v in per_scale.values()))
    return dict(cos_full=cf, cos_branch=cb, n_experts=nk, n_pairs=npair,
                per_scale=per_scale), None

"""),



CO(r"""

res, err = branch_cosines(model)

if err:

    raise RuntimeError(f"branch-cosine measurement failed: {err}")

rho_b = res["cos_branch"]

print(f"experts compared: {res['n_experts']}   pairs: {res['n_pairs']}")

print(f"  mean pairwise cosine, FULL outputs      : {res['cos_full']:+.4f}")

print(f"  mean pairwise cosine, BRANCH difference : {rho_b:+.4f}")

print()

print(f"  sqrt(1 - rho_branch) = {math.sqrt(max(0.0, 1 - rho_b)):.4f}")

print()

print("The paper's collinearity argument used the FULL-output cosine, which is")

print("inflated by the shared residual x. If the branch cosine is much lower,")

print("the collinearity explanation does not hold at this scale and the")

print("attenuation must come from the gate spread instead.")

"""),



MD("## 7 · Save figures and upload to the Hub"),



CO(r"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt



FIGDIR = Path("/kaggle/working/figs")

FIGDIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({

    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,

    "figure.dpi": 200, "savefig.bbox": "tight",

})



# ---- forest plot of paired mean differences

if rows:

    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.5))

    ys = np.arange(len(rows))[::-1]

    ax.axvline(0, color="#444", lw=0.8)

    ax.axvspan(-MARGIN, MARGIN, color="#F8DCC8", alpha=0.55, zorder=0,

               label=f"equivalence margin $\\pm${MARGIN}")

    for y, r in zip(ys, rows):

        col = "#6B8F4E" if r["equivalent"] else ("#B03A2E" if r["ci_excludes_zero"] else "#777")

        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=col, lw=1.4, solid_capstyle="round")

        ax.plot([r["delta"]], [y], "o", color=col, ms=4.5)

    ax.set_yticks(ys)

    ax.set_yticklabels([r["label"] for r in rows])

    ax.set_xlabel("paired mean difference in MAE vs Ref  (negative = arm better)")

    ax.legend(loc="lower right", frameon=False, fontsize=8)

    ax.set_title("Paired bootstrap, 95% CI on per-image MAE differences", fontsize=10)

    fig.savefig(FIGDIR / "fig_bootstrap_forest.pdf")

    fig.savefig(FIGDIR / "fig_bootstrap_forest.png")

    plt.show()



# ---- seed floor

if seed_arms:

    fig, ax = plt.subplots(figsize=(4.6, 2.2))

    ax.axhline(0, color="#999", lw=0.7)

    for i, a in enumerate(seed_arms):

        s = seed_stats[a]

        ax.bar(i, s["mean"], width=0.45,

               color="#DCE8F5", edgecolor="#4F76A3", linewidth=0.7,

               yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],

               error_kw=dict(ecolor="#333", lw=0.9, capsize=3))

    ax.axhspan(-MARGIN, MARGIN, color="#F8DCC8", alpha=0.5, zorder=0)

    ax.set_xticks(range(len(seed_arms)))

    ax.set_xticklabels([ARMS[a].replace("Ref ", "") for a in seed_arms])

    ax.set_ylabel("Δ MAE vs Ref")

    ax.set_title("Seed-only variation, same estimator", fontsize=10)

    fig.savefig(FIGDIR / "fig_seed_floor.pdf")

    fig.savefig(FIGDIR / "fig_seed_floor.png")

    plt.show()



# ---- entropy init vs trained

if init_rows:

    fig, ax = plt.subplots(figsize=(6.0, 2.6))

    mods_sorted = sorted(by_mod_init, key=lambda s: -int(re.search(r"moe_(\d+)", s).group(1))

                         if re.search(r"moe_(\d+)", s) else 0)

    xs = np.arange(len(mods_sorted))

    ax.bar(xs - 0.19, [np.mean(by_mod_init[m]) for m in mods_sorted], width=0.36,

           color="#FBEFC2", edgecolor="#B08D2B", linewidth=0.7, label="at initialization")

    ax.bar(xs + 0.19, [tr[m]["frac"] if m in tr else np.nan for m in mods_sorted], width=0.36,

           color="#DCEBD2", edgecolor="#6B8F4E", linewidth=0.7, label="after training")

    ax.axhline(1.0, color="#777", ls="--", lw=0.8, label="ceiling (log E)")

    ax.set_xticks(xs)

    ax.set_xticklabels([m.split(".")[-2] if "." in m else m for m in mods_sorted],

                       rotation=45, ha="right", fontsize=7)

    ax.set_ylim(0, 1.12)

    ax.set_ylabel("routing entropy / log E")

    ax.legend(frameon=False, fontsize=8, ncol=3)

    ax.set_title("Did the router leave initialization?", fontsize=10)

    fig.savefig(FIGDIR / "fig_router_init.pdf")

    fig.savefig(FIGDIR / "fig_router_init.png")

    plt.show()



# ---- branch vs full cosine

if model is not None and not err:

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    ax.bar([0, 1], [res["cos_full"], res["cos_branch"]], width=0.5,

           color=["#DCE8F5", "#E4DDF0"], edgecolor=["#4F76A3", "#7D6AA3"], linewidth=0.7)

    ax.axhline(0, color="#999", lw=0.7)

    ax.set_xticks([0, 1]); ax.set_xticklabels(["full $E_k$", "branch $E_k-x$"], fontsize=8)

    ax.set_ylabel("mean pairwise cosine")

    ax.set_title("Collinearity, correctly measured", fontsize=10)

    fig.savefig(FIGDIR / "fig_branch_cosine.pdf")

    fig.savefig(FIGDIR / "fig_branch_cosine.png")

    plt.show()



print("figures:", sorted(p.name for p in FIGDIR.glob("*")))

"""),



CO(r"""

# ---------------------------------------------------------------- upload

api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=True)



# 1. machine-readable results

payload = dict(

    generated_utc=__import__("datetime").datetime.utcnow().isoformat() + "Z",

    split=SPLIT, n_images=len(paired), margin=MARGIN, n_boot=N_BOOT,

    reference=REF, reference_mae=float(np.nanmean(ref)),

    arms=rows,

    seed_floor=seed_stats,

    router_init=init_rows,

    router_trained=trained_rows,

    branch_cosine=(res if (model is not None and not err) else None),

)

p = Path("/kaggle/working/bootstrap_and_router.json")

p.write_text(json.dumps(payload, indent=2, default=float))

api.upload_file(repo_id=REPO, path_or_fileobj=str(p),

                path_in_repo=f"{OUT_DIR}/results.json",

                repo_type="dataset", token=HF_TOKEN)

print("uploaded", f"{OUT_DIR}/results.json")



# 2. figures

for f in sorted(FIGDIR.glob("*")):

    api.upload_file(repo_id=REPO, path_or_fileobj=str(f),

                    path_in_repo=f"figures/paper/{f.name}",

                    repo_type="dataset", token=HF_TOKEN)

    print("uploaded", f"figures/paper/{f.name}")



# 3. the per-image difference vectors, so the test is reproducible without a re-run

np.savez_compressed("/kaggle/working/per_image_mae.npz",

                    __stems__=np.array(PAIRED_STEMS),

                    **{a: v for a, v in per_image.items()})

api.upload_file(repo_id=REPO, path_or_fileobj="/kaggle/working/per_image_mae.npz",

                path_in_repo=f"{OUT_DIR}/per_image_mae.npz",

                repo_type="dataset", token=HF_TOKEN)

print("uploaded", f"{OUT_DIR}/per_image_mae.npz")

print("\ndone.")

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
