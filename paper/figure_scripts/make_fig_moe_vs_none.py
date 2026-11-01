#!/usr/bin/env python3
"""Build the matched no-mixture qualitative figure (Fig. 7).

The previous raster of this figure could not be traced to a checkpoint (no
generator in the repository, and its commit message says only "the full
model"), while its caption asserted 14 epochs and tied the second prediction
column to an 8-epoch arm.  This script instead composes a *matched* pair, both
8 epochs, from the two archived per-arm qualitative grids:

  full MoE : results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/qualitative/qualitative_grid.png
  no MoE   : results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/qualitative/qualitative_grid.png

Each source grid is 3x3 (rows = the same three real-world scenes; columns =
input, ground truth, prediction).  We take input + ground truth + prediction
from the full-MoE grid and the prediction column from the no-MoE grid, and
assert the two arms' real-test MAE against their own metrics JSONs before
writing anything.

Outputs: paper/figures/fig_moe_vs_none_matched.png
         paper/figure_scripts/verification_fig7.tsv
Requires numpy + ImageMagick.
Usage:   python3 paper/figure_scripts/make_fig_moe_vs_none.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig_moe_vs_none_matched.png"
TSV = Path(__file__).resolve().parent / "verification_fig7.tsv"

ARMS = {
    "full": {
        "grid": "results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/qualitative/qualitative_grid.png",
        "metrics": "results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/eval_results/best/test_real/hflip/"
                   "metrics_test_real_20260924_044224.json",
    },
    "none": {
        "grid": "results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/qualitative/qualitative_grid.png",
        "metrics": "results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/eval_results/best/test_real/hflip/"
                   "metrics_test_real_20260926_040401.json",
    },
}
HEADERS = ["Input", "Ground truth", "Full MoE", "No MoE"]


def magick(*args: str) -> None:
    subprocess.run(["magick", *args], check=True, capture_output=True)


def real_mae(path: str) -> float:
    return float(json.loads((ROOT / path).read_text())["global"]["MAE"])


def main() -> int:
    maes = {k: real_mae(v["metrics"]) for k, v in ARMS.items()}
    rows, problems = [], []
    for key, v in ARMS.items():
        ok = abs(maes[key] - 0.0195) <= 0.0002
        rows.append((key, maes[key], v["metrics"], "match" if ok else "MISMATCH"))
        if not ok:
            problems.append(f"{key}: real MAE {maes[key]:.4f} != 0.0195 (Table 5)")
        if not (ROOT / v["grid"]).exists():
            problems.append(f"{key}: grid missing: {v['grid']}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"full-MoE arm real MAE {maes['full']:.4f} | no-MoE arm real MAE {maes['none']:.4f} "
          f"(both 8 epochs, same recipe)")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        grids = {k: ROOT / v["grid"] for k, v in ARMS.items()}
        # crop 3x3 cells (800 px each) from each source grid
        for k, src in grids.items():
            for r in range(3):
                for c in range(3):
                    magick(str(src), "-crop", "800x800+%d+%d" % (c * 800, r * 800),
                           "+repage", str(td / f"{k}_{r}_{c}.png"))
        # header strips: short labels only (the epochs and MAEs live in the
        # caption), sized large so they survive scaling to \columnwidth
        widths = []
        for i, text in enumerate(HEADERS):
            dest = td / f"h{i}.png"
            magick("-background", "white", "-fill", "black", "-pointsize", "74",
                   "-gravity", "center", "-size", "600x90", f"caption:{text}", str(dest))
            widths.append(dest)
        cells = [widths]
        for r in range(3):
            cells.append([td / f"full_{r}_0.png", td / f"full_{r}_1.png",
                          td / f"full_{r}_2.png", td / f"none_{r}_2.png"])
        row_files = []
        for r, row in enumerate(cells):
            dest = td / f"row{r}.png"
            magick(*[str(p) for p in row], "-gravity", "north", "-background", "white",
                   "+append", "-bordercolor", "white", "-border", "3", str(dest))
            row_files.append(dest)
        magick(*[str(p) for p in row_files], "-background", "white", "-append",
               "-bordercolor", "white", "-border", "6", str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)}")

    TSV.write_text("figure\tvalue\tsource\tmatch\n" +
                   "\n".join(f"7\t{k} arm real-test MAE={v:.4f}\t{s}\t{m}"
                             for k, v, s, m in rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
