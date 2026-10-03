# Archived architecture figures

The TikZ source of two architecture figures that were drawn for `paper/main.tex`
and then replaced. Kept so the work is not lost; neither is referenced by the
manuscript any more.

| file | what it was |
|---|---|
| `fig_arch_lane_version.tex` | the second SpatialMoE-SOD overview: three vertical lanes, one per pyramid scale, each showing features → router → top-2 experts → entropy → fusion, with the decoder and heads to the right |
| `fig_routing_flow_version.tex` | the single-token routing diagram: a full-width vertical flow from token to decoder feature, with the softmax-over-selected-pairs and the full eight-way entropy formula written out |

Both were built on the shared `\tikzset` styles in `main.tex` (`moeblock`,
`moeback`, `moerout`, `moeexp`, `moeent`, `moefuse`, `moedec`, `moehead`,
`moeflow`, `moeup`, `\swatch`, `\arrowkey`, `\moelegend`). To reuse one, copy the
block into `main.tex` and make sure that preamble block is still present.

The current Figure 1 is `../fig_architecture.pdf`, whose source is
`../../fig_architecture.tex` — a standalone file with no dependency on the
`main.tex` preamble.
