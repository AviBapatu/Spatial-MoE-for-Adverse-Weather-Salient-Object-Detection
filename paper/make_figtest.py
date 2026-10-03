"""Generate paper/_figtest.tex -- a standalone harness for the two TikZ diagrams.

The harness exists so that a TikZ failure can be debugged without a full
latexmk run of main.tex: it carries the complete preamble of main.tex plus the
two diagrams, lifted verbatim, with the float wrappers and captions removed.

Because the diagrams are extracted rather than transcribed, they cannot drift
out of sync with the manuscript. Re-run this after editing either diagram:

    python3 paper/make_figtest.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE / "main.tex"
OUT = HERE / "_figtest.tex"

BANNER = r"""% ============================================================
%  _figtest.tex  --  STANDALONE DIAGRAM HARNESS
%
%  This file is GENERATED. Do not edit it by hand; regenerate with:
%
%      python3 paper/make_figtest.py
%
%  It contains the complete preamble of main.tex followed by the two TikZ
%  diagrams, lifted VERBATIM from main.tex, with the float wrappers and
%  captions removed.
%
%  Purpose: if a full latexmk run of main.tex fails inside a TikZ picture,
%  compile this file instead. A failure here is isolated to the diagram, and
%  the reported line number maps directly onto the diagram source.
% ============================================================
"""

BLOCK_RE = re.compile(
    re.escape(r"\begin{figure}") + r".*?" + re.escape(r"\end{figure}"),
    re.S,
)


def figure_body(src: str, label: str) -> str:
    """Return one figure* block's contents, minus the wrapper and caption."""
    for block in BLOCK_RE.findall(src):
        if r"\label{" + label + "}" not in block:
            continue
        body: list[str] = []
        for line in block.splitlines():
            if line.startswith(r"\begin{figure}"):
                continue
            if line.lstrip().startswith(r"\caption"):
                break
            body.append(line)
        return "\n".join(body).strip()
    raise SystemExit(f"figure block with label {label!r} not found in {MAIN}")


def main() -> int:
    src = MAIN.read_text()
    preamble = src.split(r"\begin{document}")[0]

    parts = [
        BANNER,
        preamble,
        r"\begin{document}",
        # The diagrams are sized in units of \textwidth for the two-column
        # manuscript. Switching the harness to one column keeps them inside
        # the page here, so the only diagnostics that appear are real ones.
        r"\onecolumn",
        r"\section*{Figure 1 --- Architecture}",
        r"\noindent",
        figure_body(src, "fig:arch") + "\n",
        r"\clearpage",
        r"\section*{Figure 2 --- Routing mechanism}",
        r"\noindent",
        figure_body(src, "fig:router") + "\n",
        r"\end{document}",
    ]
    OUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
