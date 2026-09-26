# Paper

| file | what it is |
|---|---|
| `main.tex` | the LaTeX source. `pdflatex → bibtex → pdflatex ×2`, or `latexmk -pdf main.tex`. |
| `main.pdf` | the build output of that command. |
| **`SpatialMoE-SOD.pdf`** | **the compiled paper as a distributable file** — this is the one to send or upload. |
| `references.bib` | 20 entries, every one cited. |
| `figures/` | the qualitative MoE-versus-no-MoE figure and an index of the per-arm grids under `../results/`. |

The paper is 7 pages with 13 tables and 4 figures. Three of the four figures are drawn in
TikZ directly in `main.tex`; only Figure 4 is a raster image, and it is the one file in
`figures/`.

Supporting documents — the authoritative claim list, every measured number with its source
file, the benchmark comparison, and the ablation design — live in `../docs/research/`.
