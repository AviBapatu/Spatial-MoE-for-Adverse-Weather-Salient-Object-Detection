#!/usr/bin/env python3
"""Generate Fig. 3 (two-panel nineteen-entry ranking) as pgfplots code.

Data source: paper/BENCHMARK_RESULTS.md (the WXSOD paper's Tables II/III,
transcribed) plus our measured row.  Every plotted value is cross-checked
against the appendix table (tab:full19) in paper/main.tex; the script aborts
on any mismatch and writes a verification table next to itself.

Output: paper/figures/fig_benchmark_panels.tex (tikzpicture bodies only; the
figure caption stays in main.tex).

Usage:  python3 paper/figure_scripts/make_fig_benchmark_panels.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "paper" / "BENCHMARK_RESULTS.md"
MAIN = ROOT / "paper" / "main.tex"
OUT = ROOT / "paper" / "figures" / "fig_benchmark_panels.tex"
REPORT = Path(__file__).resolve().parent / "verification_figA.tsv"

OURS = {"synthetic": (0.0192, 0.9140), "real": (0.0168, 0.9151)}
SPLIT_OF_HEADING = {"## Table II — synthesized test set (1,500 images)": "synthetic",
                    "## Table III — real test set (554 images)": "real"}


def parse_benchmark() -> dict[str, dict[str, tuple[float, float]]]:
    """Return {'synthetic'|'real': {method: (MAE, S_phi)}} from the markdown."""
    tables: dict[str, dict[str, tuple[float, float]]] = {}
    split = None
    for line in BENCH.read_text().splitlines():
        if line.startswith("## Table"):
            split = SPLIT_OF_HEADING[line.strip()]
            tables[split] = {}
            continue
        if split is None or not line.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("Method",) or set(cells[1]) == {"-"}:
            continue
        if not re.fullmatch(r"[\d.]+", cells[1]):
            continue
        name = re.sub(r"\s*\(.*\)\s*$", "", cells[0]).strip()
        tables[split][name] = (float(cells[1]), float(cells[2]))
    return tables


def parse_table14() -> dict[str, dict[str, tuple[float, float]]]:
    """Parse the appendix table (tab:full19) out of main.tex.

    Splits each row on ``&`` rather than pattern-matching the label, so a
    changed method-cell decoration (italic note, superscript dagger) cannot
    silently drop a row from the check.
    """
    tex = MAIN.read_text()
    block = tex.split("label{tab:full19}")[1].split("\\end{tabular}")[0]
    rows: dict[str, dict[str, tuple[float, float]]] = {}
    for line in block.splitlines():
        if "&" not in line or line.lstrip().startswith(("\\midrule", "\\toprule",
                                                       "\\bottomrule", "\\cmidrule",
                                                       "Method", "&")):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 5:
            continue
        name = re.sub(r"^\\textbf\{|\}$", "", cells[0])
        name = re.split(r"~|\\cite|\\textit|\$", name)[0].replace("}", "").strip()
        nums = []
        for cell in cells[1:5]:
            m = re.search(r"([\d.]+)", cell)
            if not m:
                nums = []
                break
            nums.append(float(m.group(1)))
        if len(nums) == 4 and name:
            rows[name] = {"synthetic": (nums[0], nums[1]),
                          "real": (nums[2], nums[3])}
    return rows


def verify(tables: dict, t14: dict) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for split, values in tables.items():
        for name, mae_s in values.items():
            src = f"{BENCH.name}:{split}"
            t14_v = t14.get(name, {}).get(split)
            ok = t14_v == mae_s
            rows.append({"figure": "A", "value": f"{name}/{split} MAE={mae_s[0]:.4f} S={mae_s[1]:.4f}",
                         "source": src, "match": "match" if ok else "MISMATCH"})
            if not ok:
                problems.append(f"{name}/{split}: benchmark={mae_s} table14={t14_v}")
        t14_ours = t14.get("Ours", {}).get(split)
        ok = t14_ours == OURS[split]
        rows.append({"figure": "A", "value": f"Ours/{split} MAE={OURS[split][0]:.4f} S={OURS[split][1]:.4f}",
                     "source": "results/legacy_8expert_best + Table 14", "match": "match" if ok else "MISMATCH"})
        if not ok:
            problems.append(f"Ours/{split}: ours={OURS[split]} table14={t14_ours}")
    return rows, problems


def panel(split: str, rows: dict[str, tuple[float, float]], xmax: float,
          xticks: list[float], label: str) -> str:
    ordered = sorted(rows.items(), key=lambda kv: kv[1][0])  # lowest MAE first
    y_order = [name for name, _ in reversed(ordered)]        # best at the top
    lines = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  iopaxis, xbar, bar shift=0pt, bar width=5pt,",
        "  width=0.47\\textwidth, height=84mm,",
        "  scaled x ticks=false,",
        "  xticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=3},",
        f"  xlabel={{{label}}},",
        "  xlabel style={font=\\small},",
        f"  xmin=0, xmax={xmax},",
        "  symbolic y coords={" + ",".join(y_order) + "},",
        "  ytick={" + ",".join(y_order) + "},",
        "  yticklabel style={font=\\fontsize{5.6}{6.4}\\selectfont},",
        "  xtick={" + ",".join(f"{t:g}" for t in xticks) + "},",
        "  enlarge y limits=0.03,",
        "  nodes near coords, point meta=explicit symbolic,",
        "  every node near coord/.append style={font=\\fontsize{4.8}{5.6}\\selectfont,"
        " anchor=west, xshift=1.0pt},",
        "]",
    ]
    grey = [(n, v[0]) for n, v in ordered if n not in ("WFANet", "Ours")]
    lines.append("\\addplot[fill=gray!35, draw=gray!60, line width=0.3pt] coordinates {")
    lines += [f"  ({mae:.4f},{name}) [{mae:.4f}]" for name, mae in grey]
    lines.append("};")
    for name, style in (("WFANet", "orange!60, draw=orange!85"),
                        ("Ours", "violet!50, draw=violet!75")):
        mae = rows[name][0]
        lines.append(f"\\addplot[fill={style}, line width=0.3pt] coordinates {{")
        lines.append(f"  ({mae:.4f},{name}) [{mae:.4f}]")
        lines.append("};")
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(lines)


def main() -> int:
    tables = parse_benchmark()
    t14 = parse_table14()
    rows, problems = verify(tables, t14)
    print(f"parsed: synthetic={len(tables['synthetic'])} real={len(tables['real'])} "
          f"table14={len(t14)}")
    REPORT.write_text("figure\tvalue\tsource\tmatch\n" +
                      "\n".join(f"{r['figure']}\t{r['value']}\t{r['source']}\t{r['match']}"
                                for r in rows) + "\n")
    if problems:
        print("MISMATCHES:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"mismatches: NONE ({len(rows)} values checked; table written to {REPORT.name})")
    header = ("% Generated by paper/figure_scripts/make_fig_benchmark_panels.py\n"
              "% Data: paper/BENCHMARK_RESULTS.md, verified against tab:full19 in main.tex.\n"
              "% DO NOT EDIT BY HAND.\n")
    syn = dict(tables["synthetic"], Ours=OURS["synthetic"])
    real = dict(tables["real"], Ours=OURS["real"])
    left = panel("synthetic", syn, 0.098, [0.02, 0.04, 0.06, 0.08],
                 "MAE on synthetic test (1{,}500 images) $\\downarrow$")
    right = panel("real", real, 0.068, [0.02, 0.04, 0.06],
                  "MAE on real test (554 images) $\\downarrow$")
    OUT.write_text(header + left + "\n\\hfill\n" + right + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
