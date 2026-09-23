# Results

Every number in this file is read from a result file on disk and names its source. Nothing here is transcribed by hand: `RESULTS.md` is generated from the JSON under `results/`, so it can be regenerated after new evaluations land.

All evaluations are single-pass (no test-time augmentation) and scored in **original image coordinates** — predictions are reverse-geometried out of the 384x384 padded model frame before metrics are computed. Flip-average TTA (`--tta hflip`) is implemented and defaults to on for *new* runs; every table below predates that, so none of it is TTA-boosted.

## 1. Main results

Two test sets: `test_sys` (1500 synthetic adverse-weather images) and `test_real` (554 real-world images). MAE lower is better; S-measure, E-measure and F-measure higher is better.

| run | split | MAE | S_measure | E_adaptive | F_adaptive | n | source |
|---|---|---|---|---|---|---|---|
| E8 legacy (old code, renormalized gate, eff batch 32) | test_real | 0.0168 | 0.9151 | 0.9530 | 0.8631 | 554 | `results/legacy/legacy_8expert/evaluation_results/metrics_test_real_20260901_173227.json` |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | 0.0192 | 0.9139 | 0.9591 | 0.8822 | 1500 | `results/legacy/legacy_8expert/evaluation_results/metrics_test_sys_20260901_172914.json` |
| E4 K2 (dense gate, eff batch 40) | test_real | 0.0197 | 0.9043 | 0.9446 | 0.8397 | 554 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_real_20260923_060823.json` |
| E4 K2 (dense gate, eff batch 40) | test_sys | 0.0215 | 0.9040 | 0.9526 | 0.8593 | 1500 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_sys_20260923_060508.json` |
| E2 K1 (dense gate, eff batch 40) | test_real | 0.0199 | 0.9034 | 0.9442 | 0.8381 | 554 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_real_20260923_060513.json` |
| E2 K1 (dense gate, eff batch 40) | test_sys | 0.0215 | 0.9041 | 0.9520 | 0.8590 | 1500 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_sys_20260923_060210.json` |

**Provenance and confounds — read before comparing rows.**

- The **E8 legacy** numbers come from a different code version and a different recipe (renormalized gate, router noise on, load-balance weights 0.08/0.15/0.1, effective batch 32). It is the historical best and the reproduction target, not a like-for-like comparison against the rows below it.
- The **E4** and **E2** rows were trained under dense gates, router noise off, load-balance weights 0.016/0.03/0.02, and effective batch 40. They differ from the E8 row in five ways at once, so a performance gap between them and E8 cannot be attributed to expert count.
- The 2026-09 experiment set (E8 k=2 renormalized and dense, seed repeats, and E4 at the E8 recipe) is training now; those rows get added when they are evaluated.

### Extended metrics

| run | split | E_mean | E_max | F_mean | F_max | boundary_MAE | boundary_F1 |
|---|---|---|---|---|---|---|---|
| E8 legacy (old code, renormalized gate, eff batch 32) | test_real | 0.9551 | 0.9608 | 0.8747 | 0.8936 | 0.6334 | 0.3601 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | 0.9585 | 0.9633 | 0.8886 | 0.9015 | 0.4817 | 0.5472 |
| E4 K2 (dense gate, eff batch 40) | test_real | 0.9483 | 0.9559 | 0.8561 | 0.8804 | 0.7330 | 0.2768 |
| E4 K2 (dense gate, eff batch 40) | test_sys | 0.9522 | 0.9592 | 0.8692 | 0.8862 | 0.5685 | 0.4771 |
| E2 K1 (dense gate, eff batch 40) | test_real | 0.9469 | 0.9546 | 0.8535 | 0.8794 | 0.7345 | 0.2739 |
| E2 K1 (dense gate, eff batch 40) | test_sys | 0.9516 | 0.9588 | 0.8687 | 0.8862 | 0.5676 | 0.4782 |

## 2. Weather-wise breakdown

Per-condition MAE / S-measure / F-measure. Conditions with fewer than 10 samples are marked; treat them as anecdote, not evidence.

| run | split | condition | n | MAE | S-measure | F-measure |
|---|---|---|---|---|---|---|
| E8 legacy (old code, renormalized gate, eff batch 32) | test_real | dark | 125 | 0.0191 | 0.9072 | 0.8594 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_real | fog | 126 | 0.0148 | 0.9179 | 0.8520 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_real | light | 93 | 0.0243 | 0.8897 | 0.8144 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_real | rain | 120 | 0.0165 | 0.9157 | 0.8754 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_real | snow | 90 | 0.0094 | 0.9478 | 0.9173 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | clean | 167 | 0.0142 | 0.9329 | 0.9007 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | dark | 156 | 0.0199 | 0.9049 | 0.8682 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | fog | 166 | 0.0166 | 0.9308 | 0.9078 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | light | 166 | 0.0215 | 0.9091 | 0.8846 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | rain | 179 | 0.0200 | 0.9173 | 0.8867 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | rainafog | 172 | 0.0237 | 0.8902 | 0.8464 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | rainasnow | 166 | 0.0194 | 0.9152 | 0.8863 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | snow | 172 | 0.0189 | 0.9192 | 0.8877 |
| E8 legacy (old code, renormalized gate, eff batch 32) | test_sys | snowafog | 156 | 0.0189 | 0.9051 | 0.8707 |
| E4 K2 (dense gate, eff batch 40) | test_real | dark | 125 | 0.0216 | 0.8991 | 0.8392 |
| E4 K2 (dense gate, eff batch 40) | test_real | fog | 126 | 0.0168 | 0.9051 | 0.8289 |
| E4 K2 (dense gate, eff batch 40) | test_real | light | 93 | 0.0269 | 0.8804 | 0.7934 |
| E4 K2 (dense gate, eff batch 40) | test_real | rain | 120 | 0.0204 | 0.9043 | 0.8565 |
| E4 K2 (dense gate, eff batch 40) | test_real | snow | 90 | 0.0128 | 0.9349 | 0.8811 |
| E4 K2 (dense gate, eff batch 40) | test_sys | clean | 167 | 0.0178 | 0.9206 | 0.8673 |
| E4 K2 (dense gate, eff batch 40) | test_sys | dark | 156 | 0.0219 | 0.9002 | 0.8528 |
| E4 K2 (dense gate, eff batch 40) | test_sys | fog | 166 | 0.0192 | 0.9235 | 0.8903 |
| E4 K2 (dense gate, eff batch 40) | test_sys | light | 166 | 0.0228 | 0.9015 | 0.8654 |
| E4 K2 (dense gate, eff batch 40) | test_sys | rain | 179 | 0.0206 | 0.9128 | 0.8719 |
| E4 K2 (dense gate, eff batch 40) | test_sys | rainafog | 172 | 0.0258 | 0.8809 | 0.8241 |
| E4 K2 (dense gate, eff batch 40) | test_sys | rainasnow | 166 | 0.0209 | 0.9037 | 0.8612 |
| E4 K2 (dense gate, eff batch 40) | test_sys | snow | 172 | 0.0222 | 0.9011 | 0.8541 |
| E4 K2 (dense gate, eff batch 40) | test_sys | snowafog | 156 | 0.0223 | 0.8910 | 0.8463 |
| E2 K1 (dense gate, eff batch 40) | test_real | dark | 125 | 0.0223 | 0.8962 | 0.8366 |
| E2 K1 (dense gate, eff batch 40) | test_real | fog | 126 | 0.0172 | 0.9029 | 0.8268 |
| E2 K1 (dense gate, eff batch 40) | test_real | light | 93 | 0.0266 | 0.8854 | 0.7930 |
| E2 K1 (dense gate, eff batch 40) | test_real | rain | 120 | 0.0205 | 0.9032 | 0.8532 |
| E2 K1 (dense gate, eff batch 40) | test_real | snow | 90 | 0.0128 | 0.9328 | 0.8827 |
| E2 K1 (dense gate, eff batch 40) | test_sys | clean | 167 | 0.0178 | 0.9193 | 0.8655 |
| E2 K1 (dense gate, eff batch 40) | test_sys | dark | 156 | 0.0214 | 0.8998 | 0.8538 |
| E2 K1 (dense gate, eff batch 40) | test_sys | fog | 166 | 0.0192 | 0.9250 | 0.8911 |
| E2 K1 (dense gate, eff batch 40) | test_sys | light | 166 | 0.0235 | 0.9016 | 0.8660 |
| E2 K1 (dense gate, eff batch 40) | test_sys | rain | 179 | 0.0198 | 0.9150 | 0.8731 |
| E2 K1 (dense gate, eff batch 40) | test_sys | rainafog | 172 | 0.0268 | 0.8786 | 0.8191 |
| E2 K1 (dense gate, eff batch 40) | test_sys | rainasnow | 166 | 0.0204 | 0.9032 | 0.8589 |
| E2 K1 (dense gate, eff batch 40) | test_sys | snow | 172 | 0.0221 | 0.9027 | 0.8572 |
| E2 K1 (dense gate, eff batch 40) | test_sys | snowafog | 156 | 0.0225 | 0.8902 | 0.8454 |

## 3. Proxy ablations

Each arm perturbs the *trained* model at inference time, so this measures how much each component contributes to the final prediction rather than what training does. `random_routing` reports the mean over four seeds.

| model | arm | MAE | S-measure | source |
|---|---|---|---|---|
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | full_model | 0.0199 | 0.9034 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | no_entropy_fusion | 0.0199 | 0.9033 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | forced_expert_scale4 | 0.0200 | 0.9031 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | random_routing (4 seeds) | 0.0204 | 0.9006 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | full_model | 0.0197 | 0.9043 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | no_entropy_fusion | 0.0200 | 0.9016 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | forced_expert_scale4 | 0.0200 | 0.9036 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | random_routing (4 seeds) | 0.0204 | 0.8994 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `legacy` | full_model | 0.0168 | 0.9151 | `results/legacy/legacy_8expert/evaluation_results/proxy_ablation_results.json` |
| `legacy` | no_entropy_fusion | 0.0169 | 0.9144 | `results/legacy/legacy_8expert/evaluation_results/proxy_ablation_results.json` |
| `legacy` | random_routing | 0.0168 | 0.9154 | `results/legacy/legacy_8expert/evaluation_results/proxy_ablation_results.json` |
| `legacy` | forced_expert_scale4 | 0.0169 | 0.9147 | `results/legacy/legacy_8expert/evaluation_results/proxy_ablation_results.json` |
| `proxy_ablation_results.json` | full_model | 0.0209 | 0.8966 | `results/proxy_ablation_results.json` |
| `proxy_ablation_results.json` | no_entropy_fusion | 0.0207 | 0.8971 | `results/proxy_ablation_results.json` |
| `proxy_ablation_results.json` | random_routing | 0.0200 | 0.9015 | `results/proxy_ablation_results.json` |
| `proxy_ablation_results.json` | forced_expert_scale4 | 0.0209 | 0.8965 | `results/proxy_ablation_results.json` |

## 4. Routing behaviour

Per-token routing entropy, averaged over the test set. For reference, the maximum entropy of an E-way gate distribution is ln(E): ln(2)=0.693, ln(4)=1.386, ln(8)=2.079.

| model | source |
|---|---|
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | real: scale_16=0.6919, scale_4=0.6931, scale_8=0.6931 |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | real: scale_16=1.3829, scale_4=1.3863, scale_8=1.3863 |
| `legacy` | real: scale_16=0.6924, scale_4=0.6930, scale_8=0.6931 |

Expert usage (hard assignment fraction, %) at the final logged epoch, per scale. Values near 100/E are uniform; values near 0 mean the expert is effectively dead.

| run | epoch | scale | usage % | dead (<2%) |
|---|---|---|---|---|
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_4 | 48.3, 51.7 | 0 |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_8 | 51.6, 48.4 | 0 |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_16 | 49.2, 50.8 | 0 |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_4 | 25.0, 24.9, 24.8, 25.4 | 0 |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_8 | 24.6, 25.6, 24.8, 25.0 | 0 |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_16 | 20.3, 26.3, 26.8, 26.6 | 0 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_full_moe` | 7 | moe_4 | 23.3, 4.5, 5.5, 6.6, 1.9, 22.2, 30.7, 5.4 | 1 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_full_moe` | 7 | moe_8 | 1.0, 6.8, 3.2, 3.1, 16.3, 0.6, 38.2, 30.8 | 2 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_full_moe` | 7 | moe_16 | 7.8, 8.1, 10.6, 10.9, 5.5, 26.3, 12.2, 18.6 | 0 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_moe16_dense` | 7 | moe_4 | 21.4, 3.6, 3.4, 7.3, 1.7, 24.9, 32.7, 5.1 | 1 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_moe16_dense` | 7 | moe_8 | 0.7, 3.6, 2.4, 9.4, 12.6, 0.9, 32.7, 37.6 | 2 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_moe16_dense` | 7 | moe_16 | 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 | 8 |

## 5. What is missing

Stated plainly so no claim outruns the evidence:

- **The mixture-vs-dense control has never been run.** `M_DENSE` and `M_NONE` have zero runs under `results/`. The configurations exist (`experiments/v_4expert_densecontrol.json`, `experiments/v_4expert_nonecontrol.json`). Without this arm the paper cannot claim the mixture contributes anything over a matched dense model.
- **No external baselines.** Every number in this file is this project's own. A comparison table against prior methods needs either published WXSOD numbers or re-implementations.
- **Single seed per configuration** except the seed repeats now training. There is no historical variance estimate, so a difference smaller than the seed spread cannot be interpreted.
- **The backbone axis is not testable as configured.** `config.model.backbone` does not reach the model — `src/model.py` hard-codes `pvt_v2_b4` — so the generated ablation arms that vary it would train identical architectures. Do not report a backbone comparison until this is wired through.

## 6. Source files

- `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_real_20260923_060513.json`
- `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_sys_20260923_060210.json`
- `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_real_20260923_060823.json`
- `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_sys_20260923_060508.json`
- `results/legacy/legacy_8expert/evaluation_results/metrics_test_real_20260901_173227.json`
- `results/legacy/legacy_8expert/evaluation_results/metrics_test_sys_20260901_172914.json`
