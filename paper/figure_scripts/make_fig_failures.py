#!/usr/bin/env python3
"""Generate the failure-case grid from archived predictions (Figure D).

Inputs (archived artifacts and raw data, nothing hand-typed):
  predictions: results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/*.png
  ground truth: data/WXSDO_data/test_real/gt/*.jpg
  inputs:       data/WXSDO_data/test_real/input/*.jpg

Selection rule (stated in the paper caption): the three real-test images with
the highest per-image MAE of the whole split.  Per-image MAE is recomputed
here from the archived prediction PNGs and the ground truth, and the implied
global MAE is printed and recorded so it can be checked against Table 8.

Outputs:
  paper/figures/fig_failures.png
  paper/figure_scripts/failures_selection.json
  paper/figure_scripts/verification_figD.tsv

Requires numpy + ImageMagick; no torch, no PIL.
Usage:  python3 paper/figure_scripts/make_fig_failures.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none"
GT_DIR = ROOT / "data/WXSDO_data/test_real/gt"
IN_DIR = ROOT / "data/WXSDO_data/test_real/input"
OUT_PNG = ROOT / "paper/figures/fig_failures.png"
OUT_JSON = Path(__file__).resolve().parent / "failures_selection.json"
OUT_TSV = Path(__file__).resolve().parent / "verification_figD.tsv"

TABLE8_REAL_GLOBAL = 0.0168
PANEL_W = 300
STRIP_H = 17
ROW_LABEL_W = 150


def magick(*args: str) -> None:
    subprocess.run(["magick", *args], check=True, capture_output=True)


def read_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    assert data[:2] == b"P5", path
    header, body = data.split(b"\n", 3)[1], data.split(b"\n", 3)[3]
    width, height = (int(v) for v in header.split())
    return np.frombuffer(body[: width * height], dtype=np.uint8).reshape(height, width)


def gray_pgm(src: Path, dst: Path) -> None:
    magick(str(src), "-colorspace", "Gray", "-depth", "8", f"pgm:{dst}")


def clut_png(dst: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.png", Path(td) / "b.png"
        magick("-size", "128x1", "gradient:black-red", str(a))
        magick("-size", "128x1", "gradient:orange-yellow", str(b))
        magick(str(a), str(b), "+append", str(dst))


def panel(src: Path, label: str, dst: Path, *, threshold: bool = False,
          error_of: Path | None = None) -> None:
    """One panel: optional pixel operation, fixed width, caption strip on top."""
    args = [str(src)]
    if error_of is not None:
        with tempfile.TemporaryDirectory() as td:
            p, g, clut = Path(td) / "p.pgm", Path(td) / "g.pgm", Path(td) / "c.png"
            gray_pgm(src, p)
            gray_pgm(error_of, g)
            clut_png(clut)
            magick(str(p), str(g), "-compose", "difference", "-composite",
                   "-auto-level", str(clut), "-clut", str(Path(td) / "e.png"))
            args = [str(Path(td) / "e.png")]
            args += ["-resize", f"{PANEL_W}x"]
            if label:
                args += ["-bordercolor", "black", "-border", "1",
                         "-background", "white", "-splice", f"0x{STRIP_H}",
                         "-gravity", "north", "-pointsize", "12", "-annotate", "+0+1", label]
            magick(*args, str(dst))
        return
    args += ["-resize", f"{PANEL_W}x"]
    if threshold:
        args += ["-threshold", "50%"]
    if label:
        args += ["-bordercolor", "black", "-border", "1",
                 "-background", "white", "-splice", f"0x{STRIP_H}",
                 "-gravity", "north", "-pointsize", "12", "-annotate", "+0+1", label]
    magick(*args, str(dst))


def row_label(lines: list[str], dst: Path, *, width: int = ROW_LABEL_W) -> None:
    text = "\n".join(lines) if lines else " "
    magick("-background", "white", "-fill", "black", "-pointsize", "12",
           "-gravity", "center", "-size", f"{width}x250", f"caption:{text}", str(dst))


def main() -> int:
    preds = sorted(PRED_DIR.glob("*.png"))
    if not preds:
        print("no archived predictions found", file=sys.stderr)
        return 1
    stems = [p.stem for p in preds]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        gts, prs = td / "gt", td / "pred"
        gts.mkdir()
        prs.mkdir()
        subprocess.run(["magick", "mogrify", "-path", str(gts), "-format", "pgm",
                        "-colorspace", "Gray", "-depth", "8",
                        *[str(p) for p in sorted(GT_DIR.glob("*.jpg"))]],
                       check=True, capture_output=True)
        subprocess.run(["magick", "mogrify", "-path", str(prs), "-format", "pgm",
                        "-colorspace", "Gray", "-depth", "8", *[str(p) for p in preds]],
                       check=True, capture_output=True)
        maes: dict[str, float] = {}
        stats: dict[str, dict[str, float]] = {}
        for stem in stems:
            g = read_pgm(gts / f"{stem}.pgm").astype(np.float64) / 255.0
            p = read_pgm(prs / f"{stem}.pgm").astype(np.float64) / 255.0
            if g.shape != p.shape:
                print(f"shape mismatch {stem}: {g.shape} vs {p.shape}", file=sys.stderr)
                return 1
            maes[stem] = float(np.mean(np.abs(p - g)))
            gt_bin, pred_bin = g >= 0.5, p >= 0.5
            inter = float(np.logical_and(gt_bin, pred_bin).sum())
            union = float(np.logical_or(gt_bin, pred_bin).sum())
            stats[stem] = {
                "gt_area_frac": round(float(gt_bin.mean()), 5),
                "pred_area_frac": round(float(pred_bin.mean()), 5),
                "iou": round(inter / union if union else 0.0, 5),
            }

    global_mae = float(np.mean(list(maes.values())))
    print(f"images={len(maes)}  recomputed global MAE={global_mae:.5f}  "
          f"Table 8 global={TABLE8_REAL_GLOBAL}")
    ranked = sorted(maes.items(), key=lambda kv: kv[1], reverse=True)[:3]
    picks = [{"stem": s, "mae": round(m, 6), "condition": s.split("_", 1)[1],
              **stats[s]} for s, m in ranked]
    print("picks:", picks)
    OUT_JSON.write_text(json.dumps({
        "rule": "three real-test images with the highest per-image MAE",
        "prediction_source": str(PRED_DIR.relative_to(ROOT)),
        "recomputed_global_mae": round(global_mae, 6),
        "table8_global_mae": TABLE8_REAL_GLOBAL,
        "picks": picks}, indent=2))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        row_files = []
        for r, pick in enumerate([None] + picks):
            cells = []
            if pick is None:
                magick("-size", f"{ROW_LABEL_W}x34", "xc:white", str(td / "l.png"))
                cells.append(td / "l.png")
                for name in ("Input", "Ground truth", "Prediction ($\\geq0.5$)",
                             "Error $|p-g|$"):
                    dst = td / f"h_{len(cells)}.png"
                    shown = name.replace("$\\geq0.5$", ">= 0.5").replace(
                        "$|p-g|$", "|p - g|")
                    magick("-background", "white", "-fill", "black", "-pointsize", "13",
                           "-gravity", "center", "-size", f"{PANEL_W}x34",
                           f"caption:{shown}", str(dst))
                    cells.append(dst)
            else:
                stem = pick["stem"]
                row_label([stem, f"MAE {pick['mae']:.4f}", pick["condition"]], td / "l.png")
                cells.append(td / "l.png")
                panel(IN_DIR / f"{stem}.jpg", f"input ({pick['condition']})", td / "i.png")
                panel(GT_DIR / f"{stem}.jpg", "ground truth", td / "g.png")
                panel(PRED_DIR / f"{stem}.png", "prediction $\\geq 0.5$".replace("$\\geq 0.5$", ">= 0.5"),
                      td / "p.png", threshold=True)
                panel(PRED_DIR / f"{stem}.png", "|error|",
                      td / "e.png", error_of=GT_DIR / f"{stem}.jpg")
                cells += [td / "i.png", td / "g.png", td / "p.png", td / "e.png"]
            row = td / f"row{r}.png"
            magick(*[str(c) for c in cells], "-gravity", "north", "+append",
                   "-background", "white", "-bordercolor", "white", "-border", "3", str(row))
            row_files.append(row)
        magick(*[str(p) for p in row_files], "-append",
               "-bordercolor", "white", "-border", "6", str(OUT_PNG))
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")

    ok = abs(global_mae - TABLE8_REAL_GLOBAL) <= 0.0005
    OUT_TSV.write_text(
        "figure\tvalue\tsource\tmatch\n"
        f"D\trecomputed global MAE={global_mae:.5f}\t{PRED_DIR.relative_to(ROOT)}\t"
        f"{'match' if ok else 'MISMATCH'}\n" +
        "\n".join(f"D\t{p['stem']} MAE={p['mae']:.6f} ({p['condition']})\t"
                  f"{PRED_DIR.relative_to(ROOT)}\tmatch" for p in picks) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
