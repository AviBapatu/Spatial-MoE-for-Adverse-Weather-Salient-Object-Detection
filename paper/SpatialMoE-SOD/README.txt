Compile: pdflatex main && bibtex main && pdflatex main && pdflatex main
Put your own notebook figures in figures/ (same names as before):
  fig_architecture.pdf, fig_routing.pdf, fig_arms_grid.pdf, fig_pr_curves.pdf,
  fig_stages.pdf, fig_qualitative_moe_vs_none.png
(not included: they were not in the upload). main_preview.pdf was built with
low-res stand-ins for those six, so judge the final look from your own compile.
Crop fractions for fig_arms_grid / fig_stages / fig_pr_curves are the \cropband
arguments in main.tex (left, bottom, right, top fractions to trim).
