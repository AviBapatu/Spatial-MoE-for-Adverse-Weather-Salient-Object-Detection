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
ROW_LABEL_W = 150
HEADER_H = 26
LETTER_H = 18
GUTTER = 8   # white margin shared by neighbouring cells, so cells stay separate
OUTER = 6    # white margin around the whole figure
COLUMNS = ("Input", "Ground truth", "Prediction (>= 0.5)", "Error |p - g|")


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


def white(dst: Path, width: int, height: int) -> Path:
    """Blank cell of a fixed size."""
    magick("-size", f"{width}x{height}", "xc:white", str(dst))
    return dst


def grey_frame(src: Path, dst: Path) -> Path:
    """One hairline frame round a cell, matching the arm grid's 70%-grey spines."""
    magick(str(src), "-bordercolor", "gray70", "-border", "1", str(dst))
    return dst


def centred_text(text: str, dst: Path, width: int, height: int,
                 pointsize: int = 12) -> Path:
    """Cell holding centred text (the header labels and the row labels)."""
    magick("-background", "white", "-fill", "black", "-pointsize", str(pointsize),
           "-gravity", "center", "-size", f"{width}x{height}", f"caption:{text}", str(dst))
    return dst


def guttered(src: Path, dst: Path) -> Path:
    """Add this cell's half of the gutter, so adjacent cells end up GUTTER apart."""
    magick(str(src), "-bordercolor", "white", "-border", str(GUTTER // 2), str(dst))
    return dst


def error_map(pred: Path, gt: Path, dst: Path) -> None:
    """Absolute-error panel ``|p - g|``, auto-levelled through a colour ramp."""
    with tempfile.TemporaryDirectory() as td:
        p, g, clut = Path(td) / "p.pgm", Path(td) / "g.pgm", Path(td) / "clut.png"
        gray_pgm(pred, p)
        gray_pgm(gt, g)
        clut_png(clut)
        magick(str(p), str(g), "-compose", "difference", "-composite",
               "-auto-level", str(clut), "-clut", "-resize", f"{PANEL_W}x", str(dst))


def image_cell(src: Path, dst: Path, *, threshold: bool = False) -> None:
    """One image cell, resized to the column width."""
    args = [str(src), "-resize", f"{PANEL_W}x"]
    if threshold:
        args += ["-threshold", "50%"]
    magick(*args, str(dst))


def size_of(src: Path) -> tuple[int, int]:
    """Pixel size of an image, read through ImageMagick (this script avoids PIL)."""
    out = subprocess.run(["magick", "identify", "-format", "%w %h", str(src)],
                         check=True, capture_output=True, text=True).stdout.split()
    return int(out[0]), int(out[1])


def panels_for(td: Path, stem: str, row: int) -> list[tuple[Path, str]]:
    """Render one image's four panels into *td*, returning (path, tag) pairs."""
    image_cell(IN_DIR / f"{stem}.jpg", td / f"i{row}.png")
    image_cell(GT_DIR / f"{stem}.jpg", td / f"g{row}.png")
    image_cell(PRED_DIR / f"{stem}.png", td / f"p{row}.png", threshold=True)
    error_map(PRED_DIR / f"{stem}.png", GT_DIR / f"{stem}.jpg", td / f"e{row}.png")
    return [(td / f"{name}{row}.png", f"{name}{row}") for name in ("i", "g", "p", "e")]


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

        def framed(src: Path, tag: str) -> Path:
            """An image cell: hairline frame plus this cell's gutter."""
            return guttered(grey_frame(src, td / f"fr_{tag}.png"), td / f"pd_{tag}.png")

        def text_box(text: str, tag: str, width: int, height: int,
                     pointsize: int = 12) -> Path:
            """A plain text cell (header labels, row labels, panel letters)."""
            cell = centred_text(text, td / f"tx_{tag}.png", width, height, pointsize)
            return guttered(cell, td / f"pd_{tag}.png")

        # the archived images are not square and not all the same shape, so every
        # panel is padded to the tallest with white before framing: the rows then
        # have one height, and the row label is sized from the rendered panels.
        raw = [panels_for(td, pick["stem"], r) for r, pick in enumerate(picks)]
        tall = max(size_of(cell)[1] for row in raw for cell, _ in row)
        data = []
        for r, row in enumerate(raw):
            cells = []
            for cell, tag in row:
                padded = td / f"ex_{tag}.png"
                magick(str(cell), "-background", "white", "-gravity", "center",
                       "-extent", f"{PANEL_W}x{tall}", str(padded))
                cells.append(framed(padded, tag))
            data.append(cells)
        cell_h = tall + 2            # 1 px hairline frame per side
        col_w = PANEL_W + 2          # matches a framed panel, so columns line up

        rows: list[list[Path]] = []

        # header: the column labels, once, across the top; no corner box
        header = [text_box(" ", "corner", ROW_LABEL_W, HEADER_H - 2)]
        for i, name in enumerate(COLUMNS):
            header.append(text_box(name, f"h{i}", col_w, HEADER_H - 2, pointsize=13))
        rows.append(header)

        # one row per selected image: row label, then the four panels
        for r, (pick, cells) in enumerate(zip(picks, data)):
            label = text_box(f"{pick['condition']}\nMAE {pick['mae']:.4f}", f"l{r}",
                             ROW_LABEL_W, cell_h)
            rows.append([label] + cells)

        # (a)-(d) under the columns, as the arm grid labels its panels
        letters = [text_box(" ", "lc", ROW_LABEL_W, LETTER_H - 2)]
        for i in range(len(COLUMNS)):
            letters.append(text_box(f"({chr(97 + i)})", f"a{i}", col_w, LETTER_H - 2,
                                    pointsize=11))
        rows.append(letters)

        row_files = []
        for i, row in enumerate(rows):
            row_file = td / f"row{i}.png"
            magick(*[str(c) for c in row], "+append", str(row_file))
            row_files.append(row_file)
        magick(*[str(f) for f in row_files], "-append",
               "-bordercolor", "white", "-border", str(OUTER), str(OUT_PNG))
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
