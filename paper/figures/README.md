# Figures

| file | what it shows |
|---|---|
| `fig_qualitative_moe_vs_none.png` | **Figure 4 of the paper.** Four columns on the same three real-world scenes (fog, low-light, snow): input, ground truth, the full model, and the identical architecture with the mixture-of-experts removed. The two prediction columns are indistinguishable — the ablation result as a picture. 3200x2495. |
| `qualitative/qualitative_grid.png` | Per-arm grids, one directory per run under `../../results/<experiment_id>/qualitative/`. Each is a 3x3 grid: input / ground truth / prediction, on one scene per real-world weather condition. |

## Where the per-arm grids live

```
results/<experiment_id>/qualitative/qualitative_grid.png
```

One file per evaluated arm, 1800-2400 px square, ~1.5 MB. They are the qualitative
evidence behind Table 5 (the ablation table): comparing, for example,

```
results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/qualitative/qualitative_grid.png     (full model)
results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/qualitative/qualitative_grid.png (no MoE)
```

shows the two models produce near-identical masks on the same scenes, which is what the
0.0195-against-0.0195 result in the ablation table looks like.

Each grid carries its own column headers (`Input (<condition>)`, `Ground truth`,
`Prediction`). The one duplicate label to be aware of: in the composite figure above, columns
3 and 4 are two *different* models but both read "Prediction" inside the grid, because that is
how the evaluation tool renders a single model's output. The paper's caption disambiguates
them and the large header row is drawn on top.

Regenerate with the notebook's qualitative cell (`## 15_qualitative` in the generated
notebooks), not by editing these files by hand.
