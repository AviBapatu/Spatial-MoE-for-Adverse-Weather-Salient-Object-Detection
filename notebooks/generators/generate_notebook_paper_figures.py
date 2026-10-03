"""Paper-figure notebook — the data-driven figures of paper/main.tex.

Regenerates ``notebooks/figures/moe-of-sod__paper_figures.ipynb`` and is the editable
source of truth for it: change a cell here and re-run this script to rebuild the
notebook. Never edit the ``.ipynb`` by hand.

Sources (all already on the Hub — nothing here retrains or re-evaluates):

* per-arm per-image real-test predictions, 554 PNGs per arm, at
  ``analysis_results/<EXPERIMENT_ID>/proxy_ablations/full_model/{idx:04d}_{weather}.png``;
* the reported (legacy) model's predictions at ``evaluation_results/best/test_real/none/``;
* the inference-time interventions beside them under ``proxy_ablations/``:
  ``no_entropy_fusion``, ``forced_expert_scale4`` and ``random_routing_seed{1,7,42,123}``;
* ground truth from the WXSOD dataset, which the notebook finds attached or fetches
  from the project's Google Drive id.

Usage:
    python notebooks/generators/generate_notebook_paper_figures.py
"""
import json

OUTPUT_NOTEBOOK = "notebooks/figures/moe-of-sod__paper_figures.ipynb"


def create_notebook(output_notebook: str = OUTPUT_NOTEBOOK) -> None:
    """Build the notebook cell list and write it to ``OUTPUT_NOTEBOOK``."""
    cells = []

    def add_markdown(text: str) -> None:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        })

    def add_code(text: str) -> None:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        })

    # ------------------------------------------------------------------ title
    add_markdown(
        r"""# Paper figures — SpatialMoE-SOD

Generates the data-driven figures of `paper/main.tex`. Everything else in the paper is
drawn in LaTeX (TikZ / pgfplots) inside `main.tex` and is not produced here.

| output (in `/kaggle/working/paper_figures/`) | figure | content |
|---|---|---|
| `fig_arms_grid.pdf` | Fig. 4 | the same scenes under all five real-world conditions, per arm |
| `fig_pr_curves.pdf` | Fig. 7 | precision–recall and $F_\beta$ versus threshold, per condition |
| `fig_stages.pdf` | Fig. 8 | coarse-to-fine decoder progression (needs a checkpoint) |
| `fig_interventions_grid.pdf` | *(not yet in the paper)* | the same scenes under the four inference-time interventions |

**This notebook is generated.** Do not edit the `.ipynb`; edit
`notebooks/generators/generate_notebook_paper_figures.py` and re-run it.

**Inputs.** Per-image prediction maps come from the private Hub repo — every trained arm
has all 554 real-test predictions archived there. Ground truth comes from the WXSOD
dataset: attach it to this notebook, or let the notebook fetch it. Add `HF_TOKEN` to
*Add-ons ▸ Secrets* first. No GPU, no training, no evaluation.

**How to run it.** *Run All*. Then download `paper_figures.zip` from the output pane and
unzip it into `paper/figures/`. `main.tex` shows a labelled placeholder box for any figure
whose file is absent, so it compiles before and after.

**Where the output goes.** The last cell publishes the figures to the Hub under
`figures/paper/`, beside the per-arm montages in `figures/qualitative/`. Pull them anywhere
with

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Avi2006/spatial-moe-results", repo_type="dataset",
                  allow_patterns="figures/paper/*")
```

Either route — the Hub folder or the zip in the Kaggle output pane — lands the same files in
`paper/figures/`. Set `UPLOAD_TO_HUB = False` in `01_config` to skip the push.
""")

    # ------------------------------------------------------------- 01_config
    add_markdown(r"""## 01_config

Run mode, credentials, paths, and the dataset id used for the ground truth.
""")
    add_code(
        r'''import os
import re
import sys
import glob
import json
import shutil
import zipfile
import subprocess
from collections import defaultdict
from pathlib import Path

# "ALL" runs every figure; the others run one of them, which is handy when a
# single figure needs re-doing without re-pulling the images.
RUN_MODE = "ALL"          # ALL | ARMS | CURVES | INTERVENTIONS | STAGES
UPLOAD_TO_HUB = True      # publish the figures to <HF_REPO_ID>/<PAPER_FIGS_REPO_DIR>
# Where the finished figures live on the Hub. Sits under the existing `figures/`
# tree, beside the per-arm montages in `figures/qualitative/`.
PAPER_FIGS_REPO_DIR = "figures/paper"

HF_REPO_ID = "Avi2006/spatial-moe-results"
# The reported model's checkpoint survives only at this revision; checkpoints/best.pth
# on main was overwritten later (see notebooks/evaluate/moe-of-sod__legacy_eval.ipynb).
LEGACY_REVISION = "f395682f3f1dfa49fecff5d32a412be8fbc8abfd"
# Same WXSOD archive the training notebooks pull; used only if the dataset is not
# attached to this notebook.
DATA_FILE_ID = "1SUtAaxOy3c8Vzb5Ojp4NzxRB1vmsTjHL"

WORK = Path("/kaggle/working")
ARTIFACTS = WORK / "hf_artifacts"
FIGS = WORK / "paper_figures"
for _d in (ARTIFACTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_REPO_ID"] = HF_REPO_ID

print("RUN_MODE   =", RUN_MODE)
print("HF_REPO_ID =", HF_REPO_ID)
print("figures will be written to", FIGS)
''')

    # --------------------------------------------------------- 02_repo_index
    add_markdown(r"""## 02_repo_index

List the Hub repo (one API call) and resolve, per arm, **which directory holds that arm's
per-image real-test predictions**. Nothing is downloaded yet.

Every trained arm was uploaded with all 554 real-test predictions under
`analysis_results/<EXPERIMENT_ID>/proxy_ablations/full_model/`. The reported model predates
that convention, so its predictions live at `evaluation_results/best/test_real/none/`.

The arm → experiment-id map duplicates the canonical list in
`scripts/generate_results_doc.py`; it is inlined so the notebook stands alone on Kaggle.
""")
    add_code(
        r'''from huggingface_hub import HfApi

ARM_EXP = {
    "Ref": "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO",
    "A1":  "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14",
    "A2":  "EXP_B4_E2_K2_S32_R1_L3_M_SPARSE",
    "A3":  "EXP_B4_E4_K1_S32_R1_L3_M_SPARSE",
    "A4":  "EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB",
    "A5":  "EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL",
    "A6":  "EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL",
    "A7":  "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE",
    "A8":  "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1",
    "A9":  "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43",
    "A10": "EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44",
    "B1":  "EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE",
    "B2":  "EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE",
}
# Column headers for the arm grid: arm code -> short configuration label.
ARM_LABEL = {
    "R":   "R\nE8 k2, 14 ep",
    "Ref": "Ref\nE8 k2, 8 ep",
    "A1":  "A1\n14 epochs",
    "A2":  "A2\nE=2",
    "A3":  "A3\nk=1",
    "A5":  "A5\ndense expert",
    "A6":  "A6\nno MoE",
    "A7":  "A7\ndense gate",
}
# The eight arms the paper's Fig. 4 shows, widest architectural spread first.
SHOWCASE = ["R", "Ref", "A1", "A2", "A3", "A5", "A6", "A7"]
# Conditions shown as rows, using the dataset's own label strings.
#
# These are all five real-world test labels, in the order the paper's weather-wise
# table uses (easiest to hardest by MAE). Only real-world conditions appear here
# because that is what the per-arm archive holds: every arm's predictions were
# uploaded for test_real only (554 images each). The synthetic test set
# (clean/rainasnow/snowafog/...) exists per arm nowhere, so a multi-arm grid cannot
# be built from it.
#
# Fig. 7 is not affected by this list -- it auto-covers whichever conditions are
# present, so it already reports all five.
CONDITIONS = ["snow", "fog", "rain", "dark", "light"]

REPO_FILES = HfApi().list_repo_files(HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
REPO_SET = set(REPO_FILES)
print(f"{HF_REPO_ID}: {len(REPO_FILES)} files")

LEGACY_DIR_CANDIDATES = [
    "evaluation_results/best/test_real/none",
    "analysis_results/none/legacy_best/test_real/none",
]


def dir_has_files(directory):
    prefix = directory + "/"
    return any(f.startswith(prefix) for f in REPO_SET)


def pred_dir(arm):
    """Repo directory holding this arm's per-image real-test predictions."""
    if arm == "R":
        for d in LEGACY_DIR_CANDIDATES:
            if dir_has_files(d):
                return d
        return None
    d = f"analysis_results/{ARM_EXP[arm]}/proxy_ablations/full_model"
    return d if dir_has_files(d) else None


PRED_DIR = {a: pred_dir(a) for a in ["R"] + list(ARM_EXP)}

INTERVENTION_DIR = {}
_base = f"analysis_results/{ARM_EXP['Ref']}/proxy_ablations"
for variant in ("full_model", "no_entropy_fusion", "forced_expert_scale4",
                "random_routing_seed42", "random_routing_seed7"):
    if dir_has_files(f"{_base}/{variant}"):
        INTERVENTION_DIR[variant] = f"{_base}/{variant}"


def stems_in(directory):
    """{stem: repo path} for the PNGs directly inside a repo directory."""
    if not directory:
        return {}
    prefix = directory + "/"
    out = {}
    for f in REPO_FILES:
        if f.startswith(prefix) and f.lower().endswith(".png"):
            name = f[len(prefix):]
            if "/" not in name:
                out[name[:-4]] = f
    return out


print("\n--- arm availability ---")
for arm in ["R"] + list(ARM_EXP):
    n = len(stems_in(PRED_DIR[arm]))
    print(f"  {arm:<4} {'OK ' if n else 'MISSING'} {n:5d} predictions   {PRED_DIR[arm] or '-'}")

print("\n--- inference-time interventions available (arm Ref) ---")
for v, d in INTERVENTION_DIR.items():
    print(f"  {v:<24} {len(stems_in(d)):5d}  {d}")

SHOWCASE = [a for a in SHOWCASE if stems_in(PRED_DIR[a])]
print(f"\nFig. 4 will use {len(SHOWCASE)} arms: {SHOWCASE}")
if len(SHOWCASE) < 8:
    print("NOTE: fewer than eight arms were found, so the grid shows what exists."
          " Adjust the caption of Fig. 4 if the count changes.")

# The stems are identical across arms; take them from the reported model.
ALL_STEMS = sorted(stems_in(PRED_DIR["R"]))
print(f"\nreported model: {len(ALL_STEMS)} real-test predictions")
''')

    # --------------------------------------------------------------- 03_data
    add_markdown(r"""## 03_data

Locate the WXSOD ground truth: use it if the dataset is attached to the notebook, else
fetch the project's archive from Google Drive and unpack it. Nothing is downloaded if a
valid root is already present.
""")
    add_code(
        r'''EXTRACT_PATH = "/kaggle/working/WXSDO_data"


def is_dataset_valid(path):
    """Return a WXSOD root under *path*, or None."""
    def check_root(r):
        reqs = [f"{s}/{part}" for s in ("train_sys", "test_sys", "test_real")
                for part in ("input", "gt")]
        return all(os.path.isdir(os.path.join(r, q)) for q in reqs)

    if not os.path.isdir(path):
        return None
    if check_root(path):
        return path
    for root, _dirs, _files in os.walk(path):
        if check_root(root):
            return root
    return None


DATA_ROOT = None

# 1. already attached to the notebook?
for candidate in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*"):
    found = is_dataset_valid(candidate)
    if found:
        DATA_ROOT = found
        print("using attached dataset:", DATA_ROOT)
        break

# 2. already downloaded earlier in this session
if DATA_ROOT is None:
    DATA_ROOT = is_dataset_valid(EXTRACT_PATH)
    if DATA_ROOT:
        print("using existing download:", DATA_ROOT)

# 3. fetch it
if DATA_ROOT is None:
    print("dataset not found; fetching from Google Drive...")
    ZIP_PATH = "/kaggle/working/dataset.zip"
    if not os.path.exists(ZIP_PATH):
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
        subprocess.run(["gdown", DATA_FILE_ID, "-O", ZIP_PATH], check=True)
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", ZIP_PATH, "-d", EXTRACT_PATH], check=True)
    DATA_ROOT = is_dataset_valid(EXTRACT_PATH)

if DATA_ROOT is None:
    print("WARNING: no WXSOD ground truth available. Attach the dataset, or the "
          "ground-truth-dependent figures (Fig. 4, Fig. 7, the intervention grid) "
          "cannot be built. Fig. 8 does not need it.")
else:
    print("DATA_ROOT =", DATA_ROOT)
    for split in ("test_real", "test_sys"):
        n_in = len([f for f in os.listdir(os.path.join(DATA_ROOT, split, "input"))
                    if not f.startswith(".")])
        print(f"  {split}: {n_in} inputs")
''')

    # --------------------------------------------------------------- 04_helpers
    add_markdown(r"""## 04_helpers

Image loading, path lookups, scene selection, and the Hub fetch helper. Downloads are
per-file, so only the PNGs a figure actually shows are transferred — not whole arm
directories. (Fig. 7 is the exception: it genuinely needs all 554 of the reported model's
predictions, which is about 30 MB.)
""")
    add_code(
        r'''import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

BETA2 = 0.3           # F_beta weighting used throughout the paper


def fetch(repo_path):
    """Download one repo file (cached) and return its local path, or None."""
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset",
                               filename=repo_path, token=HF_TOKEN,
                               local_dir=str(ARTIFACTS))
    except Exception as exc:                      # noqa: BLE001
        print("  fetch failed:", repo_path, type(exc).__name__)
        return None


def fetch_scenes(directory, stems):
    """{stem: local png path} for the given stems."""
    out = {}
    for s in stems:
        p = fetch(f"{directory}/{s}.png")
        if p:
            out[s] = p
    return out


def read_u8(path):
    """Read an image as a uint8 array in L mode, whatever its colour space."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def read_input(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_gt(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def weather_of(stem):
    """'0001_dark' -> 'dark'."""
    return stem.split("_", 1)[1] if "_" in stem else ""


def _with_ext(split, kind, stem):
    if DATA_ROOT is None:
        return None
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = os.path.join(DATA_ROOT, split, kind, stem + ext)
        if os.path.exists(p):
            return p
    return None


def gt_path(stem, split="test_real"):
    return _with_ext(split, "gt", stem)


def input_path(stem, split="test_real"):
    return _with_ext(split, "input", stem)


def pick_scenes(stems, conditions=CONDITIONS):
    """One stem per condition: the first in sorted order."""
    scenes = {}
    for cond in conditions:
        hits = sorted(s for s in stems if weather_of(s) == cond)
        if hits:
            scenes[cond] = hits[0]
    return scenes


# Plot style shared by every figure here, following the reference papers: dotted
# grid inside a full box of black spines, small sans-serif type, thin curves.
# It also satisfies the IOP guidance that colour must not be the only cue --
# series are distinguished by line style as well.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.4,
    "grid.color": "0.62",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "legend.fontsize": 5.5,
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.3,
    "lines.linewidth": 0.9,
})

SCENES = pick_scenes(ALL_STEMS)
print("scenes chosen (identical across arms):")
for cond, stem in sorted(SCENES.items()):
    print(f"  {cond:<6} {stem}   gt: {'found' if gt_path(stem) else 'MISSING'}")
''')

    # ----------------------------------------------------------- 05_fig_arms_grid
    add_markdown(r"""## 05_fig_arms_grid — Fig. 4

All five real-world conditions (snow, fog, rain, dark, light) across eight *different*
models. Columns: input, ground truth, then one column per arm. This is the ablation table
rendered as an image: the eight model columns are visually indistinguishable, which is the
paper's point. Edit `CONDITIONS` in `01_config` to show fewer rows.
""")
    add_code(
        r'''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if RUN_MODE not in ("ALL", "ARMS"):
    print("skipped (RUN_MODE =", RUN_MODE, ")")
elif not SHOWCASE or not SCENES:
    print("cannot build Fig. 4: no arms or no scenes available")
else:
    rows = [c for c in CONDITIONS if c in SCENES and gt_path(SCENES[c])]
    if not rows:
        print("cannot build Fig. 4: no ground truth for the chosen scenes")
    else:
        ncols = 2 + len(SHOWCASE)
        # download only these three scenes for each arm, not the whole 554-image directory
        arm_pngs = {a: fetch_scenes(PRED_DIR[a], [SCENES[c] for c in rows])
                    for a in SHOWCASE}

        fig, axes = plt.subplots(len(rows), ncols,
                                 figsize=(1.55 * ncols, 1.70 * len(rows)),
                                 gridspec_kw={"wspace": 0.05, "hspace": 0.12})
        if len(rows) == 1:
            axes = axes[None, :]

        headers = ["Input", "Ground truth"] + [ARM_LABEL.get(a, a) for a in SHOWCASE]
        for r, cond in enumerate(rows):
            stem = SCENES[cond]
            panels = [input_path(stem), gt_path(stem)]
            panels += [arm_pngs.get(a, {}).get(stem) for a in SHOWCASE]
            for c, path in enumerate(panels):
                ax = axes[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_linewidth(0.4); side.set_color("0.7")
                if path is None:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                            fontsize=5, color="0.6")
                elif c == 0:
                    ax.imshow(read_input(path))
                elif c == 1:
                    ax.imshow(read_gt(path), cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(read_u8(path), cmap="gray", vmin=0, vmax=255)
                if r == 0:
                    ax.set_title(headers[c], fontsize=6.5, pad=3)
            axes[r, 0].set_ylabel(cond, fontsize=7)

        # (a)-(n) under each column, as the reference papers label panels
        for c in range(ncols):
            axes[len(rows) - 1, c].text(
                0.5, -0.06, f"({chr(97 + c)})", transform=axes[len(rows) - 1, c].transAxes,
                ha="center", va="top", fontsize=6.5)
        fig.suptitle("Same scenes, eight different trained arms", fontsize=8.5, y=0.995)
        fig.savefig(FIGS / "fig_arms_grid.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(FIGS / "fig_arms_grid.png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        print("wrote fig_arms_grid.pdf / .png  (arms:", ", ".join(SHOWCASE), ")")
''')

    # ---------------------------------------------------------- 06_fig_pr_curves
    add_markdown(r"""## 06_fig_pr_curves — Fig. 7

Precision, recall and $F_\beta$ as the binarisation threshold sweeps 0–255, per real-world
condition, for the reported model.

The curves are accumulated from 256-bin histograms rather than by re-thresholding each
image 256 times: `num_pred(t)` and `tp(t)` are reverse cumulative sums over the histogram,
so every image is touched once. This is the only figure that needs all 554 predictions.
""")
    add_code(
        r'''def prf_curves(pred_u8, gt_bool):
    """Precision, recall and F_beta for thresholds 0..255 (three 256-length arrays)."""
    p = np.asarray(pred_u8).ravel()
    g = np.asarray(gt_bool).ravel().astype(bool)
    n_pos = int(np.count_nonzero(g))

    hist_all = np.bincount(p, minlength=256).astype(np.float64)
    hist_tp = np.bincount(p[g], minlength=256).astype(np.float64)

    n_pred = np.cumsum(hist_all[::-1])[::-1]        # pixels with value >= t
    tp = np.cumsum(hist_tp[::-1])[::-1]             # true positives at threshold t

    prec = np.divide(tp, n_pred, out=np.ones_like(tp), where=n_pred > 0)
    rec = (tp / n_pos) if n_pos else np.zeros_like(tp)
    denom = BETA2 * prec + rec
    fbeta = np.divide((1.0 + BETA2) * prec * rec, denom,
                      out=np.zeros_like(denom), where=denom > 0)
    return prec, rec, fbeta


if RUN_MODE not in ("ALL", "CURVES"):
    print("skipped (RUN_MODE =", RUN_MODE, ")")
elif DATA_ROOT is None:
    print("cannot build Fig. 7: no WXSOD ground truth available")
elif not ALL_STEMS:
    print("cannot build Fig. 7: no predictions for the reported model")
else:
    print(f"downloading {len(ALL_STEMS)} prediction maps for the reported model...")
    per_cond = defaultdict(lambda: {"P": [], "R": [], "F": []})
    used = defaultdict(int)
    skipped = 0
    for stem in ALL_STEMS:
        gp = gt_path(stem)
        if gp is None:
            skipped += 1
            continue
        png = fetch(f"{PRED_DIR['R']}/{stem}.png")
        if png is None:
            skipped += 1
            continue
        pred, gt = read_u8(png), read_gt(gp)
        if pred.shape != gt.shape:                  # reverse-geometry should make these equal
            gt = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize(
                (pred.shape[1], pred.shape[0]), Image.NEAREST)) > 127
        P, R, F = prf_curves(pred, gt)
        cond = weather_of(stem)
        per_cond[cond]["P"].append(P)
        per_cond[cond]["R"].append(R)
        per_cond[cond]["F"].append(F)
        used[cond] += 1
    print(f"scored {sum(used.values())} images ({skipped} skipped, no ground truth)")

    curves = {c: (np.nanmean(np.stack(d["P"]), axis=0),
                  np.nanmean(np.stack(d["R"]), axis=0),
                  np.nanmean(np.stack(d["F"]), axis=0),
                  used[c])
              for c, d in per_cond.items()}
    order = [c for c in ["snow", "fog", "rain", "dark", "light"] if c in curves]
    order += [c for c in sorted(curves) if c not in order]
    print("conditions:", {c: curves[c][3] for c in order})

    th = np.arange(256, dtype=np.float64)
    PANEL = "abcdefghijklmn"
    fig, axes = plt.subplots(2, len(order), figsize=(2.15 * len(order), 4.9),
                             gridspec_kw={"wspace": 0.36, "hspace": 0.52})
    if len(order) == 1:
        axes = axes[:, None]

    for c, cond in enumerate(order):
        P, R, F, n = curves[cond]

        # ---- top row: precision-recall, with an inset zoom -------------------
        ax = axes[0, c]
        ax.plot(R, P, color="black", ls="-", label="Ours")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Recall")
        if c == 0:
            ax.set_ylabel("Precision")
        ax.text(0.5, 0.06, cond, transform=ax.transAxes, ha="center",
                fontsize=7, fontweight="bold")
        if c == 0:
            ax.legend(loc="lower left")
        try:
            ins = ax.inset_axes([0.46, 0.34, 0.50, 0.46])
            ins.plot(R, P, color="black", ls="-", lw=0.8)
            ins.set_xlim(0.3, 1.0)
            ins.set_ylim(0.4, 1.02)
            ins.tick_params(labelsize=4, direction="in")
            ins.set_xticks([0.4, 0.6, 0.8, 1.0])
            ins.set_yticks([0.5, 0.7, 0.9])
            ax.indicate_inset_zoom(
                ins, edgecolor="black", lw=0.5, alpha=1.0,
                connectorprops={"ls": (0, (4, 2)), "lw": 0.5, "color": "black"})
        except Exception as exc:                    # noqa: BLE001
            print("inset unavailable:", exc)
        ax.text(0.5, -0.34, f"({PANEL[c]})", transform=ax.transAxes,
                ha="center", va="top", fontsize=7)

        # ---- bottom row: F_beta against threshold --------------------------
        ax = axes[1, c]
        best = int(np.nanargmax(F))
        ax.plot(th, F, color="black", ls="-")
        ax.axvline(th[best], color="0.45", lw=0.6, ls="--")
        ax.plot([th[best]], [F[best]], marker="o", ms=2.5, color="black")
        ax.annotate(f"{F[best]:.3f}\n@ t={int(th[best])}",
                    xy=(th[best], F[best]), xytext=(5, -20),
                    textcoords="offset points", fontsize=5)
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Threshold")
        if c == 0:
            ax.set_ylabel(r"$F_\beta$")
        ax.text(0.5, -0.34, f"({PANEL[len(order) + c]})", transform=ax.transAxes,
                ha="center", va="top", fontsize=7)

    fig.savefig(FIGS / "fig_pr_curves.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGS / "fig_pr_curves.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote fig_pr_curves.pdf / .png")
''')

    # -------------------------------------------------------- 07_fig_interventions
    add_markdown(r"""## 07_fig_interventions — extra, not yet referenced in the paper

The same scenes with the reference recipe perturbed at inference: the unperturbed model,
entropy fusion disabled, every token at the $1/4$ scale forced to one expert, and random
routing. All four are archived, so this costs four more image fetches per scene.

It is the visual counterpart of the paper's inference-time ablation table and makes the
same point as Fig. 4 — the mechanism is inert. It is **not** referenced by `main.tex`
yet; add a figure block if you want it in.
""")
    add_code(
        r'''if RUN_MODE not in ("ALL", "INTERVENTIONS"):
    print("skipped (RUN_MODE =", RUN_MODE, ")")
elif not INTERVENTION_DIR or not SCENES:
    print("cannot build the intervention grid: no intervention outputs found")
else:
    rows = [c for c in CONDITIONS if c in SCENES and gt_path(SCENES[c])]
    variants = [v for v in ("full_model", "no_entropy_fusion", "forced_expert_scale4",
                            "random_routing_seed42") if v in INTERVENTION_DIR]
    titles = {
        "full_model": "Full model",
        "no_entropy_fusion": "Entropy fusion off",
        "forced_expert_scale4": "Force 1/4 to one expert",
        "random_routing_seed42": "Random routing",
    }
    if not rows or not variants:
        print("cannot build the intervention grid: nothing to show")
    else:
        ncols = 2 + len(variants)
        fetched = {v: fetch_scenes(INTERVENTION_DIR[v], [SCENES[c] for c in rows])
                   for v in variants}

        fig, axes = plt.subplots(len(rows), ncols,
                                 figsize=(1.75 * ncols, 1.70 * len(rows)),
                                 gridspec_kw={"wspace": 0.05, "hspace": 0.12})
        if len(rows) == 1:
            axes = axes[None, :]
        for r, cond in enumerate(rows):
            stem = SCENES[cond]
            panels = [input_path(stem), gt_path(stem)] + \
                     [fetched[v].get(stem) for v in variants]
            for c, path in enumerate(panels):
                ax = axes[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_linewidth(0.4); side.set_color("0.7")
                if path is None:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                            fontsize=5, color="0.6")
                elif c == 0:
                    ax.imshow(read_input(path))
                elif c == 1:
                    ax.imshow(read_gt(path), cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(read_u8(path), cmap="gray", vmin=0, vmax=255)
                if r == 0:
                    label = ["Input", "Ground truth"] + [titles[v] for v in variants]
                    ax.set_title(label[c], fontsize=6.5, pad=3)
            axes[r, 0].set_ylabel(cond, fontsize=7)

        fig.suptitle("Reported model under inference-time interventions",
                     fontsize=8.5, y=0.995)
        fig.savefig(FIGS / "fig_interventions_grid.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(FIGS / "fig_interventions_grid.png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        print("wrote fig_interventions_grid.pdf / .png  (variants:",
              ", ".join(variants), ")")
''')

    # ------------------------------------------------------------- 08_fig_stages
    add_markdown(r"""## 08_fig_stages — Fig. 8

The decoder's coarse-to-fine progression. Three stages are captured with forward hooks —
the output of `fuse_16`, of `cross_attn_16_to_8`, and of `cross_attn_8_to_4` — plus the
final saliency logits, and the saliency head (a 1×1 convolution, so it works at any
resolution) is applied to each. That shows what the model would predict from that stage
alone.

The stages are captured this way rather than read from `DecoderOutput.aux_logits_*`
because those are `None` unless deep supervision is on, and the reported checkpoint
builds to 69,212,349 parameters — the deep-supervision-off variant. An aux-based figure
would have been empty for exactly the model the paper reports.

The input is put through `model_frame()`, which reproduces the dataset's
aspect-preserving resize plus reflect padding to 384×384. Feeding the model a raw image
instead both wastes memory (the 1/8-scale attention matrix grows with the image's own
area) and gives the wrong answer.

This is the only cell that needs the model rather than archived outputs, so it is the only
one that can fail for environmental reasons. **If it fails it does not write
`fig_stages.pdf`**, leaving the manuscript's placeholder in place — a missing figure is
better than a wrong one. If the GPU runs out of memory it retries on CPU.
""")
    add_code(
        r'''def fetch_code_zip():
    """Pull and unpack the code zip from the Hub; return the project root or None."""
    try:
        zip_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset",
                                   filename="code/spatial_moe_sod_code.zip",
                                   token=HF_TOKEN)
    except Exception as exc:                       # noqa: BLE001
        print("code zip unavailable:", type(exc).__name__, exc)
        return None
    root = WORK / "spatial_moe_sod"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    if not (root / "src").is_dir():                # zip may nest under one top dir
        subs = [p for p in root.iterdir() if p.is_dir()]
        if len(subs) == 1:
            root = subs[0]
    print("unpacked code to", root)
    return root


def fetch_reported_checkpoint():
    """The reported model's checkpoint, which lives only at LEGACY_REVISION."""
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset",
                               filename="checkpoints/best.pth",
                               revision=LEGACY_REVISION, token=HF_TOKEN)
        print("checkpoint:", path)
        return path
    except Exception as exc:                       # noqa: BLE001
        print("checkpoint unavailable:", type(exc).__name__, exc)
        return None


def restore_legacy_router_noise_keys(state_dict):
    """Rename pre-refactor router-noise keys onto the current module layout.

    Local copy of ``src.evaluate._restore_legacy_parameter_names``: the learned
    router noise moved from ``moe_4.noise_linear`` to
    ``moe_4.router_noise.noise_linear``, so a checkpoint saved before that refactor
    fails a strict load without this remap. It is duplicated here on purpose --
    importing ``src.evaluate`` would drag in cv2, albumentations and
    ``py_sod_metrics``, none of which this figure needs, and that import is what
    used to break the cell. Keep in step with the helper in ``src/evaluate.py``.
    """
    restored = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if len(parts) == 3 and parts[1] == "noise_linear" and parts[0].startswith("moe_"):
            key = f"{parts[0]}.router_noise.noise_linear.{parts[2]}"
        restored[key] = value
    return restored


def model_frame(img_u8, size=384):
    """Longest-side resize to ``size``, then reflect-pad to square.

    Mirrors ``src/dataset.PairedSODDataset._aspect_preserving_resize_pad``:
    PIL BILINEAR is cv2 INTER_LINEAR, and numpy's ``reflect`` is
    cv2 BORDER_REFLECT_101 (reflect without repeating the edge pixel). The model
    was trained on this frame, so feeding it a raw image both wastes memory and
    gives the wrong answer.
    """
    h, w = img_u8.shape[:2]
    scale = size / max(h, w)
    rh, rw = int(h * scale), int(w * scale)
    resized = np.asarray(Image.fromarray(img_u8).resize((rw, rh), Image.BILINEAR))
    ph, pw = size - rh, size - rw
    top, left = ph // 2, pw // 2
    padded = np.pad(resized, ((top, ph - top), (left, pw - left), (0, 0)),
                    mode="reflect")
    meta = {"pad_top": top, "pad_left": left, "resized_h": rh, "resized_w": rw,
            "orig_h": h, "orig_w": w}
    return padded, meta


def reverse_geometry(arr, meta):
    """Undo the reflect pad and resize -- mirrors ``src.evaluate.reverse_geometry``."""
    cropped = arr[meta["pad_top"]:meta["pad_top"] + meta["resized_h"],
                  meta["pad_left"]:meta["pad_left"] + meta["resized_w"]]
    return np.asarray(Image.fromarray((np.clip(cropped, 0, 1) * 255).astype(np.uint8))
                      .resize((meta["orig_w"], meta["orig_h"]), Image.BILINEAR)) / 255.0


def to_original(logits, meta, size=384):
    """sigmoid -> upscale to the model frame -> undo pad and resize."""
    a = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    if a.shape != (size, size):
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                       .resize((size, size), Image.BILINEAR)) / 255.0
    return reverse_geometry(a, meta)


if RUN_MODE not in ("ALL", "STAGES"):
    print("skipped (RUN_MODE =", RUN_MODE, ")")
else:
    root = fetch_code_zip()
    ckpt = fetch_reported_checkpoint() if root else None
    if not root or not ckpt or not SCENES:
        print("SKIPPING Fig. 8 — the file will not be written, so the manuscript's"
              " placeholder stays in place.")
    else:
        sys.path.insert(0, str(root))
        try:
            import inspect

            import torch
            from src.model import SpatialMoESODNet

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("device:", device)
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            model_cfg = (blob.get("config") or {}).get("model", {})
            print("model config keys:", sorted(model_cfg))

            # Build the model the way src/evaluate.py does -- NOT with **model_cfg.
            # The checkpoint stores the whole model config, but the constructor takes a
            # subset of it and renames two keys (deep_supervision -> use_deep_supervision,
            # top_k -> k). `backbone` in particular is read only by the experiment-ID
            # builder, so splatting the dict raises TypeError. Keep this mapping in step
            # with evaluate.py's model-construction block.
            if "num_experts" not in model_cfg:
                print("WARNING: checkpoint config has no num_experts; falling back to the"
                      " default of 6. src/evaluate.py asserts on this.")
            ctor_args = {
                "num_experts":          model_cfg.get("num_experts", 6),
                "k":                    model_cfg.get("top_k", 2),
                "gate_mode":            model_cfg.get("gate_mode", "renormalized"),
                "window_size":          model_cfg.get("window_size", 8),
                "moe_16_mode":          model_cfg.get("moe_16_mode", "sparse"),
                "moe_type":             model_cfg.get("moe_type", "sparse"),
                "use_deep_supervision": model_cfg.get("deep_supervision", False),
                "router_noise_enabled": model_cfg.get("router_noise_enabled", True),
                "router_noise_scale":   model_cfg.get("router_noise_scale", 1.0),
                "router_noise_min_std": model_cfg.get("router_noise_min_std", 0.05),
                # Skip the ImageNet download: the checkpoint supplies every weight,
                # and the load below is strict, so a random init is fully overwritten.
                "pretrained_backbone":  False,
            }
            accepted = set(inspect.signature(SpatialMoESODNet.__init__).parameters)
            unknown = sorted(set(ctor_args) - accepted)
            if unknown:
                raise RuntimeError(
                    f"SpatialMoESODNet does not accept {unknown}; src/model.py has "
                    "diverged from src/evaluate.py -- update this cell")
            consumed = {"num_experts", "top_k", "gate_mode", "window_size", "moe_16_mode",
                        "moe_type", "deep_supervision", "router_noise_enabled",
                        "router_noise_scale", "router_noise_min_std"}
            print("config keys the constructor ignores:",
                  sorted(set(model_cfg) - consumed))

            model = SpatialMoESODNet(**ctor_args)
            state = blob["model_state_dict"]
            try:
                model.load_state_dict(state)
            except RuntimeError:
                model.load_state_dict(restore_legacy_router_noise_keys(state))
            model = model.to(device).eval()
            print("parameters:", sum(p.numel() for p in model.parameters()))
            print("deep supervision:", model.decoder.use_deep_supervision)

            # Capture the three decoder fusion stages. This does NOT depend on deep
            # supervision: the reported checkpoint has it off, so DecoderOutput's
            # aux_logits_* are all None and an aux-based figure would be empty.
            captured = {}
            handles = [
                model.decoder.fuse_16.register_forward_hook(
                    lambda m, i, o, k="1/16": captured.__setitem__(k, o)),
                model.decoder.cross_attn_16_to_8.register_forward_hook(
                    lambda m, i, o, k="1/8": captured.__setitem__(k, o)),
                model.decoder.cross_attn_8_to_4.register_forward_hook(
                    lambda m, i, o, k="1/4": captured.__setitem__(k, o)),
            ]

            def stage_maps(x):
                """Saliency-head output at each stage, in original coordinates."""
                captured.clear()
                with torch.inference_mode():
                    dec, _ = model(x)
                    head = model.decoder.saliency_head      # 1x1 conv: any resolution
                    logits = [head(captured[k]) for k in ("1/16", "1/8", "1/4")]
                    logits.append(dec.saliency_logits)
                    return [to_original(t, meta) for t in logits]

            rows = [c for c in CONDITIONS if c in SCENES and gt_path(SCENES[c])]
            headers = ["Input", "Ground truth", "1/16", "1/8", "1/4", "full",
                       "binarised"]
            fig, axes = plt.subplots(len(rows), len(headers),
                                     figsize=(1.55 * len(headers), 1.95 * len(rows)),
                                     gridspec_kw={"wspace": 0.05, "hspace": 0.12})
            if len(rows) == 1:
                axes = axes[None, :]

            for r, cond in enumerate(rows):
                stem = SCENES[cond]
                ip, gp = input_path(stem), gt_path(stem)
                if ip is None or gp is None:
                    raise FileNotFoundError(f"inputs for scene {stem!r} not found")

                img = read_input(ip)                       # original resolution
                frame, meta = model_frame(img)             # 384x384, model's frame
                x = frame.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                x = torch.from_numpy(((x - mean) / std).transpose(2, 0, 1))[None].to(device)

                try:
                    sal = stage_maps(x)
                except torch.cuda.OutOfMemoryError:
                    print(f"  CUDA OOM on {stem}; moving the model to CPU for the "
                          "rest of the figure (slower, same result)")
                    torch.cuda.empty_cache()
                    model = model.to("cpu")
                    device = "cpu"
                    x = x.to("cpu")
                    sal = stage_maps(x)

                panels = [img, read_gt(gp)] + sal + [sal[-1] > 0.5]
                for c, panel in enumerate(panels):
                    ax = axes[r, c]
                    ax.set_xticks([]); ax.set_yticks([])
                    for side in ax.spines.values():
                        side.set_linewidth(0.4); side.set_color("0.7")
                    if c == 0:
                        ax.imshow(panel)
                    else:
                        ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
                    if r == 0:
                        ax.set_title(headers[c], fontsize=6.5, pad=3)
                axes[r, 0].set_ylabel(cond, fontsize=7)

            for h in handles:
                h.remove()

            # (a)-(g) under each column, matching the reference papers
            for c in range(len(headers)):
                axes[len(rows) - 1, c].text(
                    0.5, -0.06, f"({chr(97 + c)})",
                    transform=axes[len(rows) - 1, c].transAxes,
                    ha="center", va="top", fontsize=6.5)
            fig.suptitle("Coarse-to-fine decoder stages, reported model",
                         fontsize=9, y=0.995)
            fig.savefig(FIGS / "fig_stages.pdf", bbox_inches="tight", dpi=300)
            fig.savefig(FIGS / "fig_stages.png", bbox_inches="tight", dpi=200)
            plt.close(fig)
            print("wrote fig_stages.pdf / .png")
        except Exception as exc:                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("SKIPPING Fig. 8 — figure not written:", type(exc).__name__, exc)
            print("main.tex will show a placeholder for it until this runs cleanly.")
''')

    # -------------------------------------------------------------- 09_package
    add_markdown(r"""## 09_package

Report what was produced, zip it for download, and optionally push it to the Hub so the
figures are versioned alongside the artifacts they came from.
""")
    add_code(
        r'''produced = sorted(p.name for p in FIGS.glob("*") if p.suffix in (".pdf", ".png"))
print("figures produced:")
for name in produced:
    print("  ", name, f"({(FIGS / name).stat().st_size / 1024:.0f} KiB)")

expected = ["fig_arms_grid.pdf", "fig_pr_curves.pdf", "fig_stages.pdf"]
missing = [n for n in expected if n not in produced]
if missing:
    print("\nnot produced:", missing)
    print("main.tex prints a labelled placeholder for any figure whose file is absent,")
    print("so it still compiles; copy the rest into paper/figures/ meanwhile.")

# A small index so the Hub folder is self-describing, the way figures/README.md is.
rows = "\n".join(f"| `{n}` | {(FIGS / n).stat().st_size / 1024:.0f} KiB |"
                 for n in produced)
(FIGS / "README.md").write_text(
    "# Paper figures (generated)\n\n"
    "Produced by `notebooks/figures/moe-of-sod__paper_figures.ipynb`; rebuild that notebook\n"
    "with `notebooks/generators/generate_notebook_paper_figures.py`.\n\n"
    "The paper's other four figures (1, 2, 3, 6) are drawn in LaTeX inside `paper/main.tex`\n"
    "and are not here.\n\n"
    "| file | size |\n|---|---|\n" + rows + "\n"
)

archive = shutil.make_archive(str(WORK / "paper_figures"), "zip", str(FIGS))
print("\narchive:", archive)

if not UPLOAD_TO_HUB:
    print("\nUPLOAD_TO_HUB is False - nothing pushed."
          " Set it True in 01_config to publish.")
elif not produced:
    print("\nnothing to upload.")
else:
    from huggingface_hub import login, HfApi, create_repo

    print(f"\nUploading paper figures to {PAPER_FIGS_REPO_DIR} ...")
    login(token=HF_TOKEN)
    REPO_ID = os.environ.get("HF_REPO_ID", HF_REPO_ID)
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=True)

    api = HfApi()
    uploaded_url = api.upload_folder(
        folder_path=str(FIGS),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=PAPER_FIGS_REPO_DIR,
    )
    print("Upload successful")
    print("Destination:", uploaded_url)
    print("Files uploaded:")
    for root, _, files in os.walk(FIGS):
        for f in files:
            print("  ", os.path.relpath(os.path.join(root, f), FIGS))

    print("\nPull them anywhere with:")
    print("  from huggingface_hub import snapshot_download")
    print(f"  snapshot_download(repo_id='{REPO_ID}', repo_type='dataset',")
    print(f"                    allow_patterns='{PAPER_FIGS_REPO_DIR}/*')")
''')

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
