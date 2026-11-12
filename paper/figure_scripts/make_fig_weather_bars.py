#!/usr/bin/env python3
"""Generate Fig. 8 (weather-wise MAE bars, both splits in one axes) as pgfplots code.

Source: the per-condition metrics JSONs of the reported model ---
  real: results/legacy_8expert_best/none/legacy_best/test_real/none/metrics_test_real_20260925_193945.json
  sys:  results/legacy_8expert_best/none/legacy_best/test_sys/none/metrics_test_sys_20260925_194231.json
These are the artifacts behind Table 8 (tab:weather); every plotted value is
cross-checked against that table and the script aborts on mismatch.

The figure is one grouped-bar chart rather than two panels: each row is a
condition of the synthetic split and carries the real split's bar for that same
condition where it exists (the real split has no compound conditions).

Output: paper/figures/fig_weather_bars.tex
Usage:  python3 paper/figure_scripts/make_fig_weather_bars.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "paper" / "main.tex"
OUT = ROOT / "paper" / "figures" / "fig_weather_bars.tex"
REPORT = Path(__file__).resolve().parent / "verification_figB.tsv"

SOURCES = {
    "real": "results/legacy_8expert_best/none/legacy_best/test_real/none/"
            "metrics_test_real_20260925_193945.json",
    "synthetic": "results/legacy_8expert_best/none/legacy_best/test_sys/none/"
                 "metrics_test_sys_20260925_194231.json",
}
DISPLAY = {"rainafog": "Rain${+}$fog", "rainasnow": "Rain${+}$snow",
           "snowafog": "Snow${+}$fog"}
# Table 8 spells the compound conditions "Rain+fog"; the metrics JSONs key them
# as "rainafog" / "rainasnow" / "snowafog".
ALIASES = {"rain+fog": "rainafog", "rain+snow": "rainasnow", "snow+fog": "snowafog"}
SPLIT_TAG = {"synthetic": "syn", "real": "real"}
BAR_SHIFT = 3.2  # pt: shifts the pair apart along y; positive is the upper bar


def parse_table8() -> dict[str, list[tuple[str, int, float]]]:
    """Parse tab:weather from main.tex into per-split (label, N, MAE) rows."""
    block = MAIN.read_text().split("label{tab:weather}")[1].split("\\end{tabular}")[0]
    out: dict[str, list[tuple[str, int, float]]] = {"real": [], "synthetic": []}
    split = None
    for line in block.splitlines():
        if "Real test" in line:
            split = "real"
            continue
        if "Synthetic test" in line:
            split = "synthetic"
            continue
        m = re.match(r"\s*([A-Za-z\${}+^\\]+)\s*&\s*(\d+)\s*&\s*([\d.]+)\s*&", line)
        if not m or split is None:
            continue
        label = m.group(1).replace("${+}$", "+").replace("\\", "")
        out[split].append((label, int(m.group(2)), float(m.group(3))))
    return out


def load(split: str) -> dict[str, dict]:
    data = json.loads((ROOT / SOURCES[split]).read_text())
    return {k: {"mae": v["MAE"], "n": v["sample_count"]} for k, v in data["weather"].items()}


def normalise(name: str) -> str:
    key = name.replace(" ", "").lower()
    return ALIASES.get(key, key)


def verify(table8: dict, metrics: dict) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for split, entries in table8.items():
        by_name = {normalise(k): v for k, v in metrics[split].items()}
        for label, n, mae in entries:
            key = normalise(label)
            m = by_name.get(key)
            if m is None:
                problems.append(f"{split}/{label}: condition not in metrics JSON")
                continue
            ok_n, ok_mae = m["n"] == n, abs(m["mae"] - mae) < 5e-5
            rows.append({"figure": "B", "value": f"{split}/{label} MAE={mae:.4f} N={n}",
                         "source": SOURCES[split].split("/")[-1],
                         "match": "match" if (ok_n and ok_mae) else "MISMATCH"})
            if not ok_n:
                problems.append(f"{split}/{label}: N table={n} json={m['n']}")
            if not ok_mae:
                problems.append(f"{split}/{label}: MAE table={mae} json={m['mae']:.6f}")
    return rows, problems


def merge_rows(table8: dict) -> list[dict]:
    """Return one row per condition, each carrying whichever splits have it.

    Rows follow the synthetic table order (best first), since the synthetic
    split is the one that contains every condition; a condition that only the
    real split has would be appended.
    """
    rows: list[dict] = []
    index: dict[str, dict] = {}
    for split in ("synthetic", "real"):
        for label, n, mae in table8[split]:
            key = normalise(label)
            if key not in index:
                index[key] = {"label": label, "synthetic": None, "real": None}
                rows.append(index[key])
            index[key][split] = (n, mae)
    return rows


def row_label(row: dict) -> str:
    """Tick label: condition name, per-split image counts, and a no-real mark.

    The real split has five conditions and no compound ones, so four rows have
    no real counterpart at all; those carry the mark defined in the caption
    rather than a bar.
    """
    counts = [f"{row[split][0]} {SPLIT_TAG[split]}" for split in ("synthetic", "real")
              if row[split]]
    mark = "" if row["real"] else r"$^{\ast}$"
    return f"{row['label']}{mark} ($N$ " + ", ".join(counts) + ")"


def chart(rows: list[dict], xmax: float, xticks: list[float]) -> str:
    """Emit one axes holding both splits as paired horizontal bars.

    pgfplots' symbolic coordinates cannot contain parentheses, so plot against
    bare keys and put the display text in yticklabels.
    """
    keys = [f"C{i + 1}" for i in range(len(rows))]
    labels = {k: row_label(r) for k, r in zip(keys, rows)}
    y_order = list(reversed(keys))                    # best entry at the top
    lines = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  iopaxis, xbar, bar width=6.4pt,",
        "  width=0.80\\textwidth, height=64mm,",
        "  scaled x ticks=false,",
        "  xticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=3},",
        "  xlabel={MAE $\\downarrow$ (one shared axis, both splits from $0$)},"
        " xlabel style={font=\\small},",
        f"  xmin=0, xmax={xmax},",
        "  symbolic y coords={" + ",".join(y_order) + "},",
        "  ytick={" + ",".join(y_order) + "},",
        "  yticklabels={" + ",".join(labels[k] for k in y_order) + "},",
        "  yticklabel style={font=\\fontsize{6}{7}\\selectfont},",
        "  xtick={" + ",".join(f"{t:g}" for t in xticks) + "},",
        "  enlarge y limits=0.1,",
        "]",
    ]
    for split, shift, style in (("synthetic", BAR_SHIFT, "fill=teal!80, draw=teal!95"),
                                ("real", -BAR_SHIFT, "fill=teal!45, draw=teal!80")):
        present = [(k, row) for k, row in zip(keys, rows) if row[split]]
        if not present:
            continue
        lines.append(f"\\addplot[{style}, line width=0.3pt, bar shift={shift:g}pt]"
                     " coordinates {")
        lines += [f"  ({row[split][1]:.4f},{k})" for k, row in present]
        lines.append("};")
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(lines)


def main() -> int:
    table8 = parse_table8()
    metrics = {split: load(split) for split in SOURCES}
    rows, problems = verify(table8, metrics)
    print(f"parsed: real={len(table8['real'])} synthetic={len(table8['synthetic'])}")
    REPORT.write_text("figure\tvalue\tsource\tmatch\n" +
                      "\n".join(f"{r['figure']}\t{r['value']}\t{r['source']}\t{r['match']}"
                                for r in rows) + "\n")
    if problems:
        print("MISMATCHES:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"mismatches: NONE ({len(rows)} values checked; table written to {REPORT.name})")
    header = ("% Generated by paper/figure_scripts/make_fig_weather_bars.py\n"
              "% Data: paper/figure_scripts sources listed in that script "
              "(metrics JSONs behind Table 8). DO NOT EDIT BY HAND.\n")
    rows = merge_rows(table8)
    print(f"rows: {len(rows)} (synthetic={sum(1 for r in rows if r['synthetic'])}, "
          f"real={sum(1 for r in rows if r['real'])})")
    OUT.write_text(header + chart(rows, 0.028, [0, 0.01, 0.02]) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
