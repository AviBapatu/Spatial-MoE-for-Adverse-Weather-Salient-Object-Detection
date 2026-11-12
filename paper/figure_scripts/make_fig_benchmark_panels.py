#!/usr/bin/env python3
"""Generate Fig. 3 (nineteen-entry ranking, both splits in one axes) as pgfplots code.

Data source: paper/BENCHMARK_RESULTS.md (the WXSOD paper's Tables II/III,
transcribed) plus our measured row.  Every plotted value is cross-checked
against the appendix table (tab:full19) in paper/main.tex; the script aborts
on any mismatch and writes a verification table next to itself.

The figure is a single grouped-bar chart: one row per method, carrying a
synthetic bar and a real bar on one shared axis, so the two splits can be read
against each other instead of across two panels.

Output: paper/figures/fig_benchmark_panels.tex (tikzpicture body only; the
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


#: (synthetic fill, synthetic draw, real fill, real draw) for each method class.
#: The two shades of a class differ by roughly a factor of two in luminance, and
#: the pale shade is deliberately not a near-white tint so it still reads as a
#: filled bar on the page. Greys are mixed from black/white directly because
#: xcolor's "gray!n" is a tint of mid-grey and so stays light at low n.
CLASS_STYLE = {
    "published": ("black!55", "black!80", "black!25", "black!55"),
    "WFANet": ("orange!85", "orange!95", "orange!55", "orange!85"),
    "Ours": ("violet!85", "violet!95", "violet!50", "violet!85"),
}
BAR_SHIFT = 2.2  # pt: shifts the pair apart along y; positive is the upper bar


def style_of(name: str) -> str:
    """Return the colour class of *name*, calling out WFANet and our own row."""
    return name if name in ("WFANet", "Ours") else "published"


def ylabel(name: str) -> str:
    """Colour the two called-out row labels to match their bars."""
    if name == "WFANet":
        return r"\textcolor{orange!85!black}{\textbf{WFANet}}"
    if name == "Ours":
        return r"\textcolor{violet!85!black}{\textbf{Ours}}"
    return name


def combined(synthetic: dict[str, tuple[float, float]],
             real: dict[str, tuple[float, float]]) -> str:
    """Emit one axes holding both splits as paired horizontal bars.

    Rows are ordered by synthetic MAE (best at the top), which is the split the
    paper leads on; the real bars therefore are not rank-ordered.
    """
    ordered = sorted(synthetic.items(), key=lambda kv: kv[1][0])  # lowest MAE first
    y_order = [name for name, _ in reversed(ordered)]             # best at the top
    lines = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  iopaxis, xbar, bar width=4.4pt,",
        "  width=0.88\\textwidth, height=105mm,",
        "  scaled x ticks=false,",
        "  xticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=3},",
        "  xlabel={MAE $\\downarrow$ (one shared axis, both splits from $0$)},",
        "  xlabel style={font=\\small},",
        "  xmin=0, xmax=0.098,",
        "  symbolic y coords={" + ",".join(y_order) + "},",
        "  ytick={" + ",".join(y_order) + "},",
        "  yticklabels={" + ",".join(ylabel(name) for name in y_order) + "},",
        "  yticklabel style={font=\\fontsize{5.6}{6.4}\\selectfont},",
        "  xtick={0.02,0.04,0.06,0.08},",
        # relative enlargement: 0.05 of the 18-unit range is ~1 row, which is what
        # it takes to clear the outer bar pairs from the top and bottom frame lines
        "  enlarge y limits=0.05,",
        "]",
    ]
    for split, values, shift in (("synthetic", synthetic, BAR_SHIFT),
                                 ("real", real, -BAR_SHIFT)):
        idx = 0 if split == "synthetic" else 2
        for cls in ("published", "WFANet", "Ours"):
            members = [name for name in y_order if style_of(name) == cls]
            if not members:
                continue
            fill, draw = CLASS_STYLE[cls][idx], CLASS_STYLE[cls][idx + 1]
            lines.append(f"\\addplot[fill={fill}, draw={draw}, line width=0.3pt,"
                         f" bar shift={shift:g}pt] coordinates {{")
            lines += [f"  ({values[name][0]:.4f},{name})" for name in members]
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
    OUT.write_text(header + combined(syn, real) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
