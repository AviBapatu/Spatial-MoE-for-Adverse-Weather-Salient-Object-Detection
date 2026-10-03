# Paper

**IOP journal revision (2026-10-02).** The paper was a 6-page IEEE-style conference
submission; it is now an **IOP journal manuscript** built on the publisher's own class,
`iopjournal.cls`, supplied in `References/Template/`. The class dictates the layout — A4,
single column, 153 mm text width, ragged right, 8 pt captions and 8 pt tabular — so the
preamble is deliberately thin and must not reintroduce `geometry`, `caption`, `titlesec` or
`authblk`, which would fight it.

| file | what it is |
|---|---|
| `main.tex` | the LaTeX source. `latexmk -pdf main.tex` (runs pdflatex + bibtex for you). |
| `main.pdf` | the build output of that command. |
| **`SpatialMoE-SOD.pdf`** | **the compiled paper as a distributable file** — this is the one to send or upload. |
| `iopjournal.cls` | IOP's class file. **Required to build**; copied here from `References/Template/`. |
| `orcid.pdf` | the ORCID glyph the class's `\orcid{}` command includes. |
| `references.bib` | 24 entries, every one cited. |
| `figures/` | see `figures/README.md` — one raster figure and three generated ones. |
| `make_figtest.py` | regenerates `_figtest.tex`, a standalone harness holding just the two TikZ diagrams. |
| `_figtest.tex` | **generated.** Compile this, not `main.tex`, when a diagram breaks. |
| `References/` | the two reference papers and the IOP template this revision was styled against. |
| `FINAL_PAPER_PLAN.md` | the 6-page conference plan — superseded, kept as history. |
| `REDESIGN_PLAN.md` | the conference-era layout redesign — superseded. |
| `main_FINAL_7page.pdf` | the previous conference build, kept for reference. |

The manuscript carries **16 tables and 8 figures**:

- four figures are drawn in LaTeX inside `main.tex` — the architecture and routing diagrams in
  TikZ, the real-split ranking and the weather-wise panels in pgfplots;
- one is a raster image already in `figures/` (`fig_qualitative_moe_vs_none.png`);
- three are produced by the Kaggle notebook
  `../notebooks/figures/moe-of-sod__paper_figures.ipynb` and must be copied into `figures/`
  before they appear: `fig_arms_grid.pdf`, `fig_pr_curves.pdf`, `fig_stages.pdf`.

`main.tex` compiles whether or not those three files exist. The `\figor` macro tests for the
file and, when it is missing, typesets a labelled placeholder box in its place — so a missing
figure is visible in the PDF rather than being a build error.

## Figure and plot style

Styled against the two reference papers in `References/`: the professor's own CGAN paper
(*The Visual Computer* 2022) and Wang et al. (*Phys. Scr.* 99, 2024).

- **Diagrams.** Pastel fills with a single dark outline weight on every box, black connectors,
  and **red reserved for the refinement/upsampling direction** — matching the red "Upsampling"
  arrow key in the CGAN architecture figure. Box colours are defined once in the `\tikzset` in
  `main.tex`, so Figures 1 and 2 cannot drift apart.
- **Plots.** One `iopaxis` pgfplots style: a dotted light-grey grid inside a full box of black
  spines, small sans-serif tick labels, thin curves. Panels are labelled `(a)`, `(b)`, … beneath
  the panel, as the reference papers do.
- **Colour is never the only cue.** The IOP author guidelines ask specifically for this, and the
  notebook's plots carry line styles and markers as well as colour.

Supporting documents — the authoritative claim list, every measured number with its source
file, the benchmark comparison, and the ablation design — live in `../docs/research/`.
