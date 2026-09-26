# Results

Every number in this file is read from a result file on disk and names its source. Nothing here is transcribed by hand: `RESULTS.md` is generated from the JSON under `results/`, so it can be regenerated after new evaluations land.

All evaluations are scored in **original image coordinates** — predictions are reverse-geometried out of the 384x384 padded model frame before metrics are computed. The S32 arms use flip-average TTA (`--tta hflip`); the legacy and S40 rows are single-pass and are labelled accordingly. TTA is worth about 0.0001 MAE, so it does not move any comparison here.

## 1. Main results

Two test sets: `test_sys` (1500 synthetic adverse-weather images) and `test_real` (554 real-world images). MAE lower is better; S-measure, E-measure and F-measure higher is better.

| run | split | MAE | S_measure | E_adaptive | F_adaptive | n | source |
|---|---|---|---|---|---|---|---|
| REPORTED: E8 k=2, 14 epochs (certified) | test_real | 0.0168 | 0.9151 | 0.9530 | 0.8631 | 554 | `results/legacy_8expert_best/none/legacy_best/test_real/none/metrics_test_real_20260925_193945.json` |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | 0.0192 | 0.9140 | 0.9591 | 0.8822 | 1500 | `results/legacy_8expert_best/none/legacy_best/test_sys/none/metrics_test_sys_20260925_194231.json` |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_real | 0.0195 | 0.9076 | 0.9402 | 0.8308 | 554 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/eval_results/best/test_real/hflip/metrics_test_real_20260924_044224.json` |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | 0.0214 | 0.9085 | 0.9493 | 0.8551 | 1500 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/eval_results/best/test_sys/hflip/metrics_test_sys_20260924_043703.json` |
| E8 k=2, 8 ep, seed 43 | test_real | 0.0199 | 0.9048 | 0.9380 | 0.8254 | 554 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43/eval_results/best/test_real/hflip/metrics_test_real_20260924_044342.json` |
| E8 k=2, 8 ep, seed 43 | test_sys | 0.0210 | 0.9086 | 0.9481 | 0.8513 | 1500 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43/eval_results/best/test_sys/hflip/metrics_test_sys_20260924_043819.json` |
| E8 k=2, 8 ep, seed 44 | test_real | 0.0191 | 0.9091 | 0.9412 | 0.8330 | 554 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44/eval_results/best/test_real/hflip/metrics_test_real_20260924_041616.json` |
| E8 k=2, 8 ep, seed 44 | test_sys | 0.0213 | 0.9083 | 0.9513 | 0.8574 | 1500 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44/eval_results/best/test_sys/hflip/metrics_test_sys_20260924_041058.json` |
| E8 k=2, 14 epochs | test_real | 0.0174 | 0.9169 | 0.9506 | 0.8495 | 554 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14/eval_results/best/test_real/hflip/metrics_test_real_20260926_015530.json` |
| E8 k=2, 14 epochs | test_sys | 0.0202 | 0.9141 | 0.9544 | 0.8685 | 1500 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14/eval_results/best/test_sys/hflip/metrics_test_sys_20260926_020154.json` |
| E8 k=2, dense gate | test_real | 0.0199 | 0.9060 | 0.9394 | 0.8276 | 554 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE/eval_results/best/test_real/hflip/metrics_test_real_20260924_044906.json` |
| E8 k=2, dense gate | test_sys | 0.0212 | 0.9084 | 0.9494 | 0.8536 | 1500 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE/eval_results/best/test_sys/hflip/metrics_test_sys_20260924_044342.json` |
| E8 k=2, routing at 1/4 only | test_real | 0.0195 | 0.9077 | 0.9425 | 0.8346 | 554 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1/eval_results/best/test_real/hflip/metrics_test_real_20260926_040405.json` |
| E8 k=2, routing at 1/4 only | test_sys | 0.0212 | 0.9078 | 0.9496 | 0.8564 | 1500 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1/eval_results/best/test_sys/hflip/metrics_test_sys_20260926_040935.json` |
| E2 k=2, 8 ep | test_real | 0.0193 | 0.9087 | 0.9434 | 0.8367 | 554 | `results/EXP_B4_E2_K2_S32_R1_L3_M_SPARSE/eval_results/best/test_real/hflip/metrics_test_real_20260925_225059.json` |
| E2 k=2, 8 ep | test_sys | 0.0210 | 0.9088 | 0.9509 | 0.8582 | 1500 | `results/EXP_B4_E2_K2_S32_R1_L3_M_SPARSE/eval_results/best/test_sys/hflip/metrics_test_sys_20260925_225624.json` |
| E4 k=1, 8 ep | test_real | 0.0212 | 0.8985 | 0.9251 | 0.7974 | 554 | `results/EXP_B4_E4_K1_S32_R1_L3_M_SPARSE/eval_results/best/test_real/hflip/metrics_test_real_20260925_231219.json` |
| E4 k=1, 8 ep | test_sys | 0.0231 | 0.9036 | 0.9365 | 0.8281 | 1500 | `results/EXP_B4_E4_K1_S32_R1_L3_M_SPARSE/eval_results/best/test_sys/hflip/metrics_test_sys_20260925_231817.json` |
| E4 k=2, load-balance weight 0 | test_real | 0.0196 | 0.9067 | 0.9404 | 0.8289 | 554 | `results/EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB/eval_results/best/test_real/hflip/metrics_test_real_20260925_231534.json` |
| E4 k=2, load-balance weight 0 | test_sys | 0.0212 | 0.9083 | 0.9501 | 0.8556 | 1500 | `results/EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB/eval_results/best/test_sys/hflip/metrics_test_sys_20260925_232119.json` |
| E4 k=2, dense expert type | test_real | 0.0191 | 0.9086 | 0.9431 | 0.8351 | 554 | `results/EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL/eval_results/best/test_real/hflip/metrics_test_real_20260926_040352.json` |
| E4 k=2, dense expert type | test_sys | 0.0212 | 0.9080 | 0.9504 | 0.8581 | 1500 | `results/EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL/eval_results/best/test_sys/hflip/metrics_test_sys_20260926_040946.json` |
| E4 k=2, no MoE at all | test_real | 0.0195 | 0.9066 | 0.9401 | 0.8304 | 554 | `results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/eval_results/best/test_real/hflip/metrics_test_real_20260926_040401.json` |
| E4 k=2, no MoE at all | test_sys | 0.0210 | 0.9084 | 0.9493 | 0.8553 | 1500 | `results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/eval_results/best/test_sys/hflip/metrics_test_sys_20260926_040952.json` |
| S40: E2 k=1, dense gate, batch 40 | test_real | 0.0199 | 0.9034 | 0.9442 | 0.8381 | 554 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_real_20260923_060513.json` |
| S40: E2 k=1, dense gate, batch 40 | test_sys | 0.0215 | 0.9041 | 0.9520 | 0.8590 | 1500 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_sys_20260923_060210.json` |
| S40: E4 k=2, dense gate, batch 40 | test_real | 0.0197 | 0.9043 | 0.9446 | 0.8397 | 554 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_real_20260923_060823.json` |
| S40: E4 k=2, dense gate, batch 40 | test_sys | 0.0215 | 0.9040 | 0.9526 | 0.8593 | 1500 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_sys_20260923_060508.json` |

**Provenance and confounds — read before comparing rows.**

**Three groups, not one table.** Compare within a group only.

- **REPORTED** — the 14-epoch E8 model. This is the headline row and the one the paper reports. It was trained under the older recipe (SSIM/boundary/Z-loss off, deep supervision off, effective batch 32) and has been re-evaluated under current code, reproducing its recorded MAE/S exactly.
- **S32 arms** — one recipe, one comparable family: renormalized gate, router noise on, load-balance weights 0.08/0.15/0.10, effective batch 32, 8 epochs unless stated. Every design-choice comparison in the paper must be made inside this group.
- **S40 arms** — an earlier recipe generation: effective batch 40 (2578 vs 3222 optimizer steps), different loss weights, and router noise off in the dense-gate family. Not comparable row-for-row with the S32 arms.

### Extended metrics

| run | split | E_mean | E_max | F_mean | F_max | boundary_MAE | boundary_F1 |
|---|---|---|---|---|---|---|---|
| REPORTED: E8 k=2, 14 epochs (certified) | test_real | 0.9551 | 0.9608 | 0.8747 | 0.8936 | 0.6334 | 0.3601 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | 0.9585 | 0.9633 | 0.8886 | 0.9016 | 0.4817 | 0.5472 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_real | 0.9467 | 0.9567 | 0.8543 | 0.8854 | 0.7211 | 0.2876 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | 0.9516 | 0.9617 | 0.8715 | 0.8925 | 0.5534 | 0.4947 |
| E8 k=2, 8 ep, seed 43 | test_real | 0.9452 | 0.9552 | 0.8498 | 0.8822 | 0.7345 | 0.2747 |
| E8 k=2, 8 ep, seed 43 | test_sys | 0.9514 | 0.9609 | 0.8688 | 0.8920 | 0.5559 | 0.4914 |
| E8 k=2, 8 ep, seed 44 | test_real | 0.9478 | 0.9583 | 0.8565 | 0.8854 | 0.7157 | 0.2922 |
| E8 k=2, 8 ep, seed 44 | test_sys | 0.9516 | 0.9604 | 0.8714 | 0.8911 | 0.5525 | 0.4945 |
| E8 k=2, 14 epochs | test_real | 0.9554 | 0.9631 | 0.8719 | 0.8963 | 0.6675 | 0.3373 |
| E8 k=2, 14 epochs | test_sys | 0.9536 | 0.9617 | 0.8812 | 0.8975 | 0.5067 | 0.5348 |
| E8 k=2, dense gate | test_real | 0.9460 | 0.9566 | 0.8525 | 0.8854 | 0.7317 | 0.2809 |
| E8 k=2, dense gate | test_sys | 0.9508 | 0.9608 | 0.8692 | 0.8917 | 0.5612 | 0.4895 |
| E8 k=2, routing at 1/4 only | test_real | 0.9495 | 0.9592 | 0.8584 | 0.8869 | 0.7221 | 0.2881 |
| E8 k=2, routing at 1/4 only | test_sys | 0.9503 | 0.9588 | 0.8705 | 0.8889 | 0.5554 | 0.4929 |
| E2 k=2, 8 ep | test_real | 0.9491 | 0.9583 | 0.8584 | 0.8860 | 0.7186 | 0.2912 |
| E2 k=2, 8 ep | test_sys | 0.9519 | 0.9611 | 0.8722 | 0.8922 | 0.5521 | 0.4968 |
| E4 k=1, 8 ep | test_real | 0.9376 | 0.9560 | 0.8293 | 0.8801 | 0.7688 | 0.2390 |
| E4 k=1, 8 ep | test_sys | 0.9440 | 0.9592 | 0.8514 | 0.8878 | 0.5874 | 0.4577 |
| E4 k=2, load-balance weight 0 | test_real | 0.9471 | 0.9571 | 0.8534 | 0.8860 | 0.7260 | 0.2835 |
| E4 k=2, load-balance weight 0 | test_sys | 0.9511 | 0.9604 | 0.8706 | 0.8917 | 0.5544 | 0.4941 |
| E4 k=2, dense expert type | test_real | 0.9492 | 0.9581 | 0.8583 | 0.8868 | 0.7174 | 0.2915 |
| E4 k=2, dense expert type | test_sys | 0.9512 | 0.9602 | 0.8715 | 0.8902 | 0.5540 | 0.4941 |
| E4 k=2, no MoE at all | test_real | 0.9479 | 0.9581 | 0.8541 | 0.8841 | 0.7250 | 0.2835 |
| E4 k=2, no MoE at all | test_sys | 0.9521 | 0.9615 | 0.8708 | 0.8909 | 0.5557 | 0.4907 |
| S40: E2 k=1, dense gate, batch 40 | test_real | 0.9469 | 0.9546 | 0.8535 | 0.8794 | 0.7345 | 0.2739 |
| S40: E2 k=1, dense gate, batch 40 | test_sys | 0.9516 | 0.9588 | 0.8687 | 0.8862 | 0.5676 | 0.4782 |
| S40: E4 k=2, dense gate, batch 40 | test_real | 0.9483 | 0.9559 | 0.8561 | 0.8804 | 0.7330 | 0.2768 |
| S40: E4 k=2, dense gate, batch 40 | test_sys | 0.9522 | 0.9592 | 0.8692 | 0.8862 | 0.5685 | 0.4771 |

## 2. Weather-wise breakdown

Per-condition MAE / S-measure / F-measure. Conditions with fewer than 10 samples are marked; treat them as anecdote, not evidence.

| run | split | condition | n | MAE | S-measure | F-measure |
|---|---|---|---|---|---|---|
| REPORTED: E8 k=2, 14 epochs (certified) | test_real | dark | 125 | 0.0191 | 0.9072 | 0.8594 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_real | fog | 126 | 0.0148 | 0.9179 | 0.8521 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_real | light | 93 | 0.0243 | 0.8897 | 0.8145 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_real | rain | 120 | 0.0165 | 0.9157 | 0.8754 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_real | snow | 90 | 0.0094 | 0.9478 | 0.9173 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | clean | 167 | 0.0142 | 0.9329 | 0.9007 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | dark | 156 | 0.0199 | 0.9049 | 0.8682 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | fog | 166 | 0.0166 | 0.9308 | 0.9078 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | light | 166 | 0.0215 | 0.9091 | 0.8846 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | rain | 179 | 0.0200 | 0.9173 | 0.8867 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | rainafog | 172 | 0.0237 | 0.8903 | 0.8464 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | rainasnow | 166 | 0.0194 | 0.9152 | 0.8863 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | snow | 172 | 0.0189 | 0.9192 | 0.8877 |
| REPORTED: E8 k=2, 14 epochs (certified) | test_sys | snowafog | 156 | 0.0189 | 0.9051 | 0.8707 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_real | dark | 125 | 0.0221 | 0.9005 | 0.8296 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_real | fog | 126 | 0.0171 | 0.9058 | 0.8135 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_real | light | 93 | 0.0262 | 0.8875 | 0.7841 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_real | rain | 120 | 0.0193 | 0.9088 | 0.8512 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_real | snow | 90 | 0.0127 | 0.9392 | 0.8774 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | clean | 167 | 0.0177 | 0.9225 | 0.8594 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | dark | 156 | 0.0214 | 0.9025 | 0.8443 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | fog | 166 | 0.0193 | 0.9272 | 0.8837 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | light | 166 | 0.0252 | 0.9038 | 0.8574 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | rain | 179 | 0.0198 | 0.9179 | 0.8699 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | rainafog | 172 | 0.0249 | 0.8870 | 0.8255 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | rainasnow | 166 | 0.0207 | 0.9078 | 0.8581 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | snow | 172 | 0.0219 | 0.9095 | 0.8538 |
| E8 k=2, 8 ep, batch 32 (REPRO) | test_sys | snowafog | 156 | 0.0218 | 0.8968 | 0.8421 |
| E8 k=2, 8 ep, seed 43 | test_real | dark | 125 | 0.0222 | 0.8994 | 0.8240 |
| E8 k=2, 8 ep, seed 43 | test_real | fog | 126 | 0.0175 | 0.9025 | 0.8067 |
| E8 k=2, 8 ep, seed 43 | test_real | light | 93 | 0.0259 | 0.8832 | 0.7804 |
| E8 k=2, 8 ep, seed 43 | test_real | rain | 120 | 0.0197 | 0.9087 | 0.8498 |
| E8 k=2, 8 ep, seed 43 | test_real | snow | 90 | 0.0142 | 0.9323 | 0.8674 |
| E8 k=2, 8 ep, seed 43 | test_sys | clean | 167 | 0.0171 | 0.9243 | 0.8606 |
| E8 k=2, 8 ep, seed 43 | test_sys | dark | 156 | 0.0216 | 0.9023 | 0.8389 |
| E8 k=2, 8 ep, seed 43 | test_sys | fog | 166 | 0.0196 | 0.9254 | 0.8790 |
| E8 k=2, 8 ep, seed 43 | test_sys | light | 166 | 0.0233 | 0.9039 | 0.8564 |
| E8 k=2, 8 ep, seed 43 | test_sys | rain | 179 | 0.0203 | 0.9168 | 0.8626 |
| E8 k=2, 8 ep, seed 43 | test_sys | rainafog | 172 | 0.0245 | 0.8880 | 0.8170 |
| E8 k=2, 8 ep, seed 43 | test_sys | rainasnow | 166 | 0.0206 | 0.9068 | 0.8521 |
| E8 k=2, 8 ep, seed 43 | test_sys | snow | 172 | 0.0207 | 0.9117 | 0.8573 |
| E8 k=2, 8 ep, seed 43 | test_sys | snowafog | 156 | 0.0219 | 0.8972 | 0.8365 |
| E8 k=2, 8 ep, seed 44 | test_real | dark | 125 | 0.0222 | 0.9022 | 0.8305 |
| E8 k=2, 8 ep, seed 44 | test_real | fog | 126 | 0.0165 | 0.9090 | 0.8239 |
| E8 k=2, 8 ep, seed 44 | test_real | light | 93 | 0.0242 | 0.8916 | 0.7849 |
| E8 k=2, 8 ep, seed 44 | test_real | rain | 120 | 0.0192 | 0.9094 | 0.8514 |
| E8 k=2, 8 ep, seed 44 | test_real | snow | 90 | 0.0131 | 0.9365 | 0.8742 |
| E8 k=2, 8 ep, seed 44 | test_sys | clean | 167 | 0.0179 | 0.9230 | 0.8707 |
| E8 k=2, 8 ep, seed 44 | test_sys | dark | 156 | 0.0212 | 0.9039 | 0.8510 |
| E8 k=2, 8 ep, seed 44 | test_sys | fog | 166 | 0.0192 | 0.9264 | 0.8863 |
| E8 k=2, 8 ep, seed 44 | test_sys | light | 166 | 0.0231 | 0.9043 | 0.8648 |
| E8 k=2, 8 ep, seed 44 | test_sys | rain | 179 | 0.0204 | 0.9159 | 0.8683 |
| E8 k=2, 8 ep, seed 44 | test_sys | rainafog | 172 | 0.0252 | 0.8855 | 0.8221 |
| E8 k=2, 8 ep, seed 44 | test_sys | rainasnow | 166 | 0.0213 | 0.9079 | 0.8531 |
| E8 k=2, 8 ep, seed 44 | test_sys | snow | 172 | 0.0205 | 0.9108 | 0.8551 |
| E8 k=2, 8 ep, seed 44 | test_sys | snowafog | 156 | 0.0226 | 0.8964 | 0.8444 |
| E8 k=2, 14 epochs | test_real | dark | 125 | 0.0201 | 0.9086 | 0.8491 |
| E8 k=2, 14 epochs | test_real | fog | 126 | 0.0150 | 0.9160 | 0.8354 |
| E8 k=2, 14 epochs | test_real | light | 93 | 0.0238 | 0.8969 | 0.8067 |
| E8 k=2, 14 epochs | test_real | rain | 120 | 0.0167 | 0.9202 | 0.8646 |
| E8 k=2, 14 epochs | test_real | snow | 90 | 0.0113 | 0.9460 | 0.8941 |
| E8 k=2, 14 epochs | test_sys | clean | 167 | 0.0161 | 0.9287 | 0.8811 |
| E8 k=2, 14 epochs | test_sys | dark | 156 | 0.0200 | 0.9075 | 0.8570 |
| E8 k=2, 14 epochs | test_sys | fog | 166 | 0.0175 | 0.9322 | 0.8973 |
| E8 k=2, 14 epochs | test_sys | light | 166 | 0.0232 | 0.9101 | 0.8713 |
| E8 k=2, 14 epochs | test_sys | rain | 179 | 0.0195 | 0.9200 | 0.8782 |
| E8 k=2, 14 epochs | test_sys | rainafog | 172 | 0.0247 | 0.8921 | 0.8356 |
| E8 k=2, 14 epochs | test_sys | rainasnow | 166 | 0.0202 | 0.9124 | 0.8678 |
| E8 k=2, 14 epochs | test_sys | snow | 172 | 0.0197 | 0.9186 | 0.8716 |
| E8 k=2, 14 epochs | test_sys | snowafog | 156 | 0.0203 | 0.9041 | 0.8552 |
| E8 k=2, dense gate | test_real | dark | 125 | 0.0216 | 0.8999 | 0.8243 |
| E8 k=2, dense gate | test_real | fog | 126 | 0.0173 | 0.9063 | 0.8185 |
| E8 k=2, dense gate | test_real | light | 93 | 0.0262 | 0.8880 | 0.7905 |
| E8 k=2, dense gate | test_real | rain | 120 | 0.0206 | 0.9052 | 0.8412 |
| E8 k=2, dense gate | test_real | snow | 90 | 0.0139 | 0.9337 | 0.8653 |
| E8 k=2, dense gate | test_sys | clean | 167 | 0.0175 | 0.9219 | 0.8571 |
| E8 k=2, dense gate | test_sys | dark | 156 | 0.0209 | 0.9046 | 0.8469 |
| E8 k=2, dense gate | test_sys | fog | 166 | 0.0192 | 0.9268 | 0.8843 |
| E8 k=2, dense gate | test_sys | light | 166 | 0.0240 | 0.9032 | 0.8612 |
| E8 k=2, dense gate | test_sys | rain | 179 | 0.0204 | 0.9160 | 0.8614 |
| E8 k=2, dense gate | test_sys | rainafog | 172 | 0.0250 | 0.8876 | 0.8231 |
| E8 k=2, dense gate | test_sys | rainasnow | 166 | 0.0206 | 0.9076 | 0.8507 |
| E8 k=2, dense gate | test_sys | snow | 172 | 0.0215 | 0.9096 | 0.8539 |
| E8 k=2, dense gate | test_sys | snowafog | 156 | 0.0215 | 0.8971 | 0.8436 |
| E8 k=2, routing at 1/4 only | test_real | dark | 125 | 0.0222 | 0.8987 | 0.8271 |
| E8 k=2, routing at 1/4 only | test_real | fog | 126 | 0.0163 | 0.9103 | 0.8277 |
| E8 k=2, routing at 1/4 only | test_real | light | 93 | 0.0257 | 0.8870 | 0.7873 |
| E8 k=2, routing at 1/4 only | test_real | rain | 120 | 0.0201 | 0.9077 | 0.8512 |
| E8 k=2, routing at 1/4 only | test_real | snow | 90 | 0.0132 | 0.9378 | 0.8812 |
| E8 k=2, routing at 1/4 only | test_sys | clean | 167 | 0.0176 | 0.9223 | 0.8629 |
| E8 k=2, routing at 1/4 only | test_sys | dark | 156 | 0.0210 | 0.9029 | 0.8482 |
| E8 k=2, routing at 1/4 only | test_sys | fog | 166 | 0.0192 | 0.9269 | 0.8867 |
| E8 k=2, routing at 1/4 only | test_sys | light | 166 | 0.0227 | 0.9048 | 0.8658 |
| E8 k=2, routing at 1/4 only | test_sys | rain | 179 | 0.0211 | 0.9127 | 0.8631 |
| E8 k=2, routing at 1/4 only | test_sys | rainafog | 172 | 0.0253 | 0.8868 | 0.8213 |
| E8 k=2, routing at 1/4 only | test_sys | rainasnow | 166 | 0.0210 | 0.9069 | 0.8560 |
| E8 k=2, routing at 1/4 only | test_sys | snow | 172 | 0.0204 | 0.9111 | 0.8600 |
| E8 k=2, routing at 1/4 only | test_sys | snowafog | 156 | 0.0224 | 0.8944 | 0.8434 |
| E2 k=2, 8 ep | test_real | dark | 125 | 0.0219 | 0.9006 | 0.8329 |
| E2 k=2, 8 ep | test_real | fog | 126 | 0.0165 | 0.9087 | 0.8257 |
| E2 k=2, 8 ep | test_real | light | 93 | 0.0254 | 0.8930 | 0.7976 |
| E2 k=2, 8 ep | test_real | rain | 120 | 0.0199 | 0.9066 | 0.8496 |
| E2 k=2, 8 ep | test_real | snow | 90 | 0.0127 | 0.9392 | 0.8807 |
| E2 k=2, 8 ep | test_sys | clean | 167 | 0.0173 | 0.9232 | 0.8674 |
| E2 k=2, 8 ep | test_sys | dark | 156 | 0.0208 | 0.9039 | 0.8499 |
| E2 k=2, 8 ep | test_sys | fog | 166 | 0.0191 | 0.9277 | 0.8880 |
| E2 k=2, 8 ep | test_sys | light | 166 | 0.0223 | 0.9058 | 0.8660 |
| E2 k=2, 8 ep | test_sys | rain | 179 | 0.0201 | 0.9175 | 0.8702 |
| E2 k=2, 8 ep | test_sys | rainafog | 172 | 0.0253 | 0.8843 | 0.8221 |
| E2 k=2, 8 ep | test_sys | rainasnow | 166 | 0.0203 | 0.9082 | 0.8584 |
| E2 k=2, 8 ep | test_sys | snow | 172 | 0.0216 | 0.9093 | 0.8547 |
| E2 k=2, 8 ep | test_sys | snowafog | 156 | 0.0219 | 0.8981 | 0.8462 |
| E4 k=1, 8 ep | test_real | dark | 125 | 0.0238 | 0.8913 | 0.7948 |
| E4 k=1, 8 ep | test_real | fog | 126 | 0.0186 | 0.8979 | 0.7828 |
| E4 k=1, 8 ep | test_real | light | 93 | 0.0279 | 0.8752 | 0.7441 |
| E4 k=1, 8 ep | test_real | rain | 120 | 0.0210 | 0.9015 | 0.8235 |
| E4 k=1, 8 ep | test_real | snow | 90 | 0.0147 | 0.9294 | 0.8418 |
| E4 k=1, 8 ep | test_sys | clean | 167 | 0.0191 | 0.9154 | 0.8362 |
| E4 k=1, 8 ep | test_sys | dark | 156 | 0.0234 | 0.8987 | 0.8154 |
| E4 k=1, 8 ep | test_sys | fog | 166 | 0.0217 | 0.9198 | 0.8580 |
| E4 k=1, 8 ep | test_sys | light | 166 | 0.0252 | 0.9008 | 0.8345 |
| E4 k=1, 8 ep | test_sys | rain | 179 | 0.0223 | 0.9114 | 0.8435 |
| E4 k=1, 8 ep | test_sys | rainafog | 172 | 0.0273 | 0.8830 | 0.7897 |
| E4 k=1, 8 ep | test_sys | rainasnow | 166 | 0.0223 | 0.9045 | 0.8312 |
| E4 k=1, 8 ep | test_sys | snow | 172 | 0.0232 | 0.9056 | 0.8285 |
| E4 k=1, 8 ep | test_sys | snowafog | 156 | 0.0235 | 0.8925 | 0.8144 |
| E4 k=2, load-balance weight 0 | test_real | dark | 125 | 0.0221 | 0.8989 | 0.8266 |
| E4 k=2, load-balance weight 0 | test_real | fog | 126 | 0.0172 | 0.9055 | 0.8161 |
| E4 k=2, load-balance weight 0 | test_real | light | 93 | 0.0257 | 0.8875 | 0.7836 |
| E4 k=2, load-balance weight 0 | test_real | rain | 120 | 0.0199 | 0.9075 | 0.8458 |
| E4 k=2, load-balance weight 0 | test_real | snow | 90 | 0.0129 | 0.9382 | 0.8744 |
| E4 k=2, load-balance weight 0 | test_sys | clean | 167 | 0.0172 | 0.9239 | 0.8656 |
| E4 k=2, load-balance weight 0 | test_sys | dark | 156 | 0.0209 | 0.9036 | 0.8479 |
| E4 k=2, load-balance weight 0 | test_sys | fog | 166 | 0.0192 | 0.9268 | 0.8841 |
| E4 k=2, load-balance weight 0 | test_sys | light | 166 | 0.0235 | 0.9041 | 0.8620 |
| E4 k=2, load-balance weight 0 | test_sys | rain | 179 | 0.0203 | 0.9166 | 0.8670 |
| E4 k=2, load-balance weight 0 | test_sys | rainafog | 172 | 0.0255 | 0.8864 | 0.8212 |
| E4 k=2, load-balance weight 0 | test_sys | rainasnow | 166 | 0.0211 | 0.9075 | 0.8562 |
| E4 k=2, load-balance weight 0 | test_sys | snow | 172 | 0.0211 | 0.9088 | 0.8547 |
| E4 k=2, load-balance weight 0 | test_sys | snowafog | 156 | 0.0220 | 0.8964 | 0.8405 |
| E4 k=2, dense expert type | test_real | dark | 125 | 0.0206 | 0.9020 | 0.8315 |
| E4 k=2, dense expert type | test_real | fog | 126 | 0.0164 | 0.9087 | 0.8285 |
| E4 k=2, dense expert type | test_real | light | 93 | 0.0265 | 0.8867 | 0.7854 |
| E4 k=2, dense expert type | test_real | rain | 120 | 0.0192 | 0.9103 | 0.8530 |
| E4 k=2, dense expert type | test_real | snow | 90 | 0.0130 | 0.9378 | 0.8772 |
| E4 k=2, dense expert type | test_sys | clean | 167 | 0.0170 | 0.9246 | 0.8689 |
| E4 k=2, dense expert type | test_sys | dark | 156 | 0.0212 | 0.9026 | 0.8504 |
| E4 k=2, dense expert type | test_sys | fog | 166 | 0.0191 | 0.9263 | 0.8877 |
| E4 k=2, dense expert type | test_sys | light | 166 | 0.0231 | 0.9049 | 0.8664 |
| E4 k=2, dense expert type | test_sys | rain | 179 | 0.0200 | 0.9167 | 0.8700 |
| E4 k=2, dense expert type | test_sys | rainafog | 172 | 0.0252 | 0.8867 | 0.8225 |
| E4 k=2, dense expert type | test_sys | rainasnow | 166 | 0.0212 | 0.9077 | 0.8582 |
| E4 k=2, dense expert type | test_sys | snow | 172 | 0.0226 | 0.9046 | 0.8512 |
| E4 k=2, dense expert type | test_sys | snowafog | 156 | 0.0214 | 0.8974 | 0.8465 |
| E4 k=2, no MoE at all | test_real | dark | 125 | 0.0218 | 0.9004 | 0.8290 |
| E4 k=2, no MoE at all | test_real | fog | 126 | 0.0168 | 0.9071 | 0.8233 |
| E4 k=2, no MoE at all | test_real | light | 93 | 0.0267 | 0.8834 | 0.7801 |
| E4 k=2, no MoE at all | test_real | rain | 120 | 0.0196 | 0.9066 | 0.8450 |
| E4 k=2, no MoE at all | test_real | snow | 90 | 0.0125 | 0.9385 | 0.8745 |
| E4 k=2, no MoE at all | test_sys | clean | 167 | 0.0171 | 0.9231 | 0.8643 |
| E4 k=2, no MoE at all | test_sys | dark | 156 | 0.0211 | 0.9030 | 0.8493 |
| E4 k=2, no MoE at all | test_sys | fog | 166 | 0.0192 | 0.9256 | 0.8837 |
| E4 k=2, no MoE at all | test_sys | light | 166 | 0.0231 | 0.9042 | 0.8619 |
| E4 k=2, no MoE at all | test_sys | rain | 179 | 0.0203 | 0.9172 | 0.8677 |
| E4 k=2, no MoE at all | test_sys | rainafog | 172 | 0.0255 | 0.8867 | 0.8195 |
| E4 k=2, no MoE at all | test_sys | rainasnow | 166 | 0.0203 | 0.9085 | 0.8547 |
| E4 k=2, no MoE at all | test_sys | snow | 172 | 0.0213 | 0.9074 | 0.8512 |
| E4 k=2, no MoE at all | test_sys | snowafog | 156 | 0.0214 | 0.8986 | 0.8444 |
| S40: E2 k=1, dense gate, batch 40 | test_real | dark | 125 | 0.0223 | 0.8962 | 0.8366 |
| S40: E2 k=1, dense gate, batch 40 | test_real | fog | 126 | 0.0172 | 0.9029 | 0.8268 |
| S40: E2 k=1, dense gate, batch 40 | test_real | light | 93 | 0.0266 | 0.8854 | 0.7930 |
| S40: E2 k=1, dense gate, batch 40 | test_real | rain | 120 | 0.0205 | 0.9032 | 0.8532 |
| S40: E2 k=1, dense gate, batch 40 | test_real | snow | 90 | 0.0128 | 0.9328 | 0.8827 |
| S40: E2 k=1, dense gate, batch 40 | test_sys | clean | 167 | 0.0178 | 0.9193 | 0.8655 |
| S40: E2 k=1, dense gate, batch 40 | test_sys | dark | 156 | 0.0214 | 0.8998 | 0.8538 |
| S40: E2 k=1, dense gate, batch 40 | test_sys | fog | 166 | 0.0192 | 0.9250 | 0.8911 |
| S40: E2 k=1, dense gate, batch 40 | test_sys | light | 166 | 0.0235 | 0.9016 | 0.8660 |
| S40: E2 k=1, dense gate, batch 40 | test_sys | rain | 179 | 0.0198 | 0.9150 | 0.8731 |
| S40: E2 k=1, dense gate, batch 40 | test_sys | rainafog | 172 | 0.0268 | 0.8786 | 0.8191 |
| S40: E2 k=1, dense gate, batch 40 | test_sys | rainasnow | 166 | 0.0204 | 0.9032 | 0.8589 |
| S40: E2 k=1, dense gate, batch 40 | test_sys | snow | 172 | 0.0221 | 0.9027 | 0.8572 |
| S40: E2 k=1, dense gate, batch 40 | test_sys | snowafog | 156 | 0.0225 | 0.8902 | 0.8454 |
| S40: E4 k=2, dense gate, batch 40 | test_real | dark | 125 | 0.0216 | 0.8991 | 0.8392 |
| S40: E4 k=2, dense gate, batch 40 | test_real | fog | 126 | 0.0168 | 0.9051 | 0.8289 |
| S40: E4 k=2, dense gate, batch 40 | test_real | light | 93 | 0.0269 | 0.8804 | 0.7934 |
| S40: E4 k=2, dense gate, batch 40 | test_real | rain | 120 | 0.0204 | 0.9043 | 0.8565 |
| S40: E4 k=2, dense gate, batch 40 | test_real | snow | 90 | 0.0128 | 0.9349 | 0.8811 |
| S40: E4 k=2, dense gate, batch 40 | test_sys | clean | 167 | 0.0178 | 0.9206 | 0.8673 |
| S40: E4 k=2, dense gate, batch 40 | test_sys | dark | 156 | 0.0219 | 0.9002 | 0.8528 |
| S40: E4 k=2, dense gate, batch 40 | test_sys | fog | 166 | 0.0192 | 0.9235 | 0.8903 |
| S40: E4 k=2, dense gate, batch 40 | test_sys | light | 166 | 0.0228 | 0.9015 | 0.8654 |
| S40: E4 k=2, dense gate, batch 40 | test_sys | rain | 179 | 0.0206 | 0.9128 | 0.8719 |
| S40: E4 k=2, dense gate, batch 40 | test_sys | rainafog | 172 | 0.0258 | 0.8809 | 0.8241 |
| S40: E4 k=2, dense gate, batch 40 | test_sys | rainasnow | 166 | 0.0209 | 0.9037 | 0.8612 |
| S40: E4 k=2, dense gate, batch 40 | test_sys | snow | 172 | 0.0222 | 0.9011 | 0.8541 |
| S40: E4 k=2, dense gate, batch 40 | test_sys | snowafog | 156 | 0.0223 | 0.8910 | 0.8463 |

## 3. Proxy ablations

Each arm perturbs the *trained* model at inference time, so this measures how much each component contributes to the final prediction rather than what training does. `random_routing` reports the mean over four seeds.

| model | arm | MAE | S-measure | source |
|---|---|---|---|---|
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE` | full_model | 0.0548 | 0.7671 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE` | no_entropy_fusion | 0.0548 | 0.7669 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE` | forced_expert_scale4 | 0.0549 | 0.7665 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE` | random_routing (4 seeds) | 0.0504 | 0.7821 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | full_model | 0.0199 | 0.9034 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | no_entropy_fusion | 0.0199 | 0.9033 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | forced_expert_scale4 | 0.0200 | 0.9031 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | random_routing (4 seeds) | 0.0204 | 0.9006 | `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K2_S32_R1_L3_M_SPARSE` | full_model | 0.0195 | 0.9053 | `results/EXP_B4_E2_K2_S32_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K2_S32_R1_L3_M_SPARSE` | no_entropy_fusion | 0.0195 | 0.9054 | `results/EXP_B4_E2_K2_S32_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K2_S32_R1_L3_M_SPARSE` | forced_expert_scale4 | 0.0195 | 0.9054 | `results/EXP_B4_E2_K2_S32_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E2_K2_S32_R1_L3_M_SPARSE` | random_routing (4 seeds) | 0.0195 | 0.9053 | `results/EXP_B4_E2_K2_S32_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K1_S32_R1_L3_M_SPARSE` | full_model | 0.0213 | 0.8955 | `results/EXP_B4_E4_K1_S32_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K1_S32_R1_L3_M_SPARSE` | no_entropy_fusion | 0.0206 | 0.8991 | `results/EXP_B4_E4_K1_S32_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K1_S32_R1_L3_M_SPARSE` | forced_expert_scale4 | 0.0214 | 0.8949 | `results/EXP_B4_E4_K1_S32_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K1_S32_R1_L3_M_SPARSE` | random_routing (4 seeds) | 0.0194 | 0.9060 | `results/EXP_B4_E4_K1_S32_R1_L3_M_SPARSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL` | full_model | 0.0192 | 0.9056 | `results/EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL` | no_entropy_fusion | 0.0191 | 0.9063 | `results/EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL` | forced_expert_scale4 | 0.0192 | 0.9056 | `results/EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL` | random_routing (4 seeds) | 0.0192 | 0.9056 | `results/EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL` | full_model | 0.0196 | 0.9040 | `results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL` | no_entropy_fusion | 0.0199 | 0.9028 | `results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL` | forced_expert_scale4 | 0.0196 | 0.9040 | `results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL` | random_routing (4 seeds) | 0.0196 | 0.9040 | `results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB` | full_model | 0.0198 | 0.9035 | `results/EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB` | no_entropy_fusion | 0.0195 | 0.9054 | `results/EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB` | forced_expert_scale4 | 0.0198 | 0.9034 | `results/EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB` | random_routing (4 seeds) | 0.0195 | 0.9053 | `results/EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | full_model | 0.0197 | 0.9043 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | no_entropy_fusion | 0.0200 | 0.9016 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | forced_expert_scale4 | 0.0200 | 0.9036 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | random_routing (4 seeds) | 0.0204 | 0.8994 | `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO` | full_model | 0.0197 | 0.9043 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO` | no_entropy_fusion | 0.0197 | 0.9044 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO` | forced_expert_scale4 | 0.0196 | 0.9048 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO` | random_routing (4 seeds) | 0.0193 | 0.9060 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE` | full_model | 0.0202 | 0.9026 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE` | no_entropy_fusion | 0.0204 | 0.8986 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE` | forced_expert_scale4 | 0.0203 | 0.9018 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE` | random_routing (4 seeds) | 0.0205 | 0.8998 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14` | full_model | 0.0174 | 0.9139 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14` | no_entropy_fusion | 0.0175 | 0.9141 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14` | forced_expert_scale4 | 0.0175 | 0.9138 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14` | random_routing (4 seeds) | 0.0174 | 0.9142 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43` | full_model | 0.0201 | 0.9021 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43` | no_entropy_fusion | 0.0201 | 0.9019 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43` | forced_expert_scale4 | 0.0200 | 0.9028 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43` | random_routing (4 seeds) | 0.0197 | 0.9046 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44` | full_model | 0.0192 | 0.9062 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44` | no_entropy_fusion | 0.0192 | 0.9059 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44` | forced_expert_scale4 | 0.0192 | 0.9058 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44` | random_routing (4 seeds) | 0.0190 | 0.9065 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1` | full_model | 0.0195 | 0.9052 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1` | no_entropy_fusion | 0.0194 | 0.9052 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1` | forced_expert_scale4 | 0.0195 | 0.9050 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1/proxy_ablations/proxy_ablation_results.json` |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1` | random_routing (4 seeds) | 0.0195 | 0.9051 | `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1/proxy_ablations/proxy_ablation_results.json` |
| `legacy` | full_model | 0.0168 | 0.9151 | `results/legacy/legacy_8expert/evaluation_results/proxy_ablation_results.json` |
| `legacy` | no_entropy_fusion | 0.0169 | 0.9144 | `results/legacy/legacy_8expert/evaluation_results/proxy_ablation_results.json` |
| `legacy` | random_routing | 0.0168 | 0.9154 | `results/legacy/legacy_8expert/evaluation_results/proxy_ablation_results.json` |
| `legacy` | forced_expert_scale4 | 0.0169 | 0.9147 | `results/legacy/legacy_8expert/evaluation_results/proxy_ablation_results.json` |
| `legacy` | full_model | 0.0198 | 0.9042 | `results/legacy/legacy_8expert/proxy_ablations/proxy_ablation_results.json` |
| `legacy` | no_entropy_fusion | 0.0196 | 0.9047 | `results/legacy/legacy_8expert/proxy_ablations/proxy_ablation_results.json` |
| `legacy` | forced_expert_scale4 | 0.0198 | 0.9040 | `results/legacy/legacy_8expert/proxy_ablations/proxy_ablation_results.json` |
| `legacy` | random_routing (4 seeds) | 0.0195 | 0.9056 | `results/legacy/legacy_8expert/proxy_ablations/proxy_ablation_results.json` |
| `proxy_ablation_results.json` | full_model | 0.0209 | 0.8966 | `results/proxy_ablation_results.json` |
| `proxy_ablation_results.json` | no_entropy_fusion | 0.0207 | 0.8971 | `results/proxy_ablation_results.json` |
| `proxy_ablation_results.json` | random_routing | 0.0200 | 0.9015 | `results/proxy_ablation_results.json` |
| `proxy_ablation_results.json` | forced_expert_scale4 | 0.0209 | 0.8965 | `results/proxy_ablation_results.json` |

## 4. Routing behaviour

Per-token routing entropy, averaged over the test set. For reference, the maximum entropy of an E-way gate distribution is ln(E): ln(2)=0.693, ln(4)=1.386, ln(8)=2.079.

| model | source |
|---|---|
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE` | real: scale_16=0.6888, scale_4=0.6927, scale_8=0.6927 |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | real: scale_16=0.6919, scale_4=0.6931, scale_8=0.6931 |
| `EXP_B4_E2_K2_S32_R1_L3_M_SPARSE` | real: scale_16=0.6867, scale_4=0.6927, scale_8=0.6927 |
| `EXP_B4_E4_K1_S32_R1_L3_M_SPARSE` | real: scale_16=1.3764, scale_4=1.3840, scale_8=1.3539 |
| `EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL` | real: scale_16=0.0000, scale_4=0.0000, scale_8=0.0000 |
| `EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL` | real: scale_16=0.0000, scale_4=0.0000, scale_8=0.0000 |
| `EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB` | real: scale_16=1.3819, scale_4=1.3858, scale_8=1.3860 |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | real: scale_16=1.3829, scale_4=1.3863, scale_8=1.3863 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO` | real: scale_16=2.0752, scale_4=2.0787, scale_8=2.0789 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE` | real: scale_16=2.0786, scale_4=2.0794, scale_8=2.0794 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14` | real: scale_16=2.0731, scale_4=2.0786, scale_8=2.0790 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43` | real: scale_16=2.0737, scale_4=2.0787, scale_8=2.0790 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44` | real: scale_16=2.0741, scale_4=2.0783, scale_8=2.0790 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1` | real: scale_16=0.0000, scale_4=2.0786, scale_8=0.0000 |
| `legacy` | real: scale_16=0.6924, scale_4=0.6930, scale_8=0.6931 |
| `legacy` | real: scale_16=0.6924, scale_4=0.6930, scale_8=0.6931 |
| `legacy` | real: scale_16=2.0756, scale_4=2.0788, scale_8=2.0789 |

Expert usage (hard assignment fraction, %) at the final logged epoch, per scale. Values near 100/E are uniform; values near 0 mean the expert is effectively dead.

| run | epoch | scale | usage % | dead (<2%) |
|---|---|---|---|---|
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_4 | 48.3, 51.7 | 0 |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_8 | 51.6, 48.4 | 0 |
| `EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_16 | 49.2, 50.8 | 0 |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_4 | 25.0, 24.9, 24.8, 25.4 | 0 |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_8 | 24.6, 25.6, 24.8, 25.0 | 0 |
| `EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE` | 8 | moe_16 | 20.3, 26.3, 26.8, 26.6 | 0 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO` | 8 | moe_4 | 21.1, 6.3, 5.3, 5.6, 1.5, 23.8, 29.7, 6.7 | 1 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO` | 8 | moe_8 | 1.1, 6.9, 3.1, 4.2, 15.5, 0.8, 40.7, 27.7 | 2 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO` | 8 | moe_16 | 9.2, 5.1, 14.1, 13.1, 4.1, 27.1, 16.3, 11.1 | 0 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE` | 8 | moe_4 | 16.4, 13.2, 11.6, 12.4, 11.4, 12.3, 12.4, 10.3 | 0 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE` | 8 | moe_8 | 14.9, 12.2, 11.3, 12.7, 10.8, 12.7, 12.6, 12.7 | 0 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE` | 8 | moe_16 | 12.7, 13.9, 11.0, 12.0, 11.9, 13.7, 12.8, 12.0 | 0 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43` | 8 | moe_4 | 6.4, 28.3, 4.0, 1.9, 27.3, 5.4, 10.1, 16.8 | 1 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43` | 8 | moe_8 | 12.1, 27.8, 3.0, 7.4, 4.6, 28.9, 2.9, 13.2 | 0 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43` | 8 | moe_16 | 1.3, 14.3, 23.9, 16.9, 26.2, 3.5, 4.0, 9.9 | 1 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44` | 8 | moe_4 | 18.7, 16.4, 7.9, 3.1, 14.4, 1.0, 3.7, 34.8 | 1 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44` | 8 | moe_8 | 24.5, 15.5, 2.5, 2.2, 35.7, 5.3, 12.6, 1.6 | 1 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44` | 8 | moe_16 | 2.4, 5.8, 13.4, 31.9, 7.6, 19.7, 9.2, 10.1 | 0 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_full_moe` | 7 | moe_4 | 23.3, 4.5, 5.5, 6.6, 1.9, 22.2, 30.7, 5.4 | 1 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_full_moe` | 7 | moe_8 | 1.0, 6.8, 3.2, 3.1, 16.3, 0.6, 38.2, 30.8 | 2 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_full_moe` | 7 | moe_16 | 7.8, 8.1, 10.6, 10.9, 5.5, 26.3, 12.2, 18.6 | 0 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_moe16_dense` | 7 | moe_4 | 21.4, 3.6, 3.4, 7.3, 1.7, 24.9, 32.7, 5.1 | 1 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_moe16_dense` | 7 | moe_8 | 0.7, 3.6, 2.4, 9.4, 12.6, 0.9, 32.7, 37.6 | 2 |
| `EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_moe16_dense` | 7 | moe_16 | 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 | 8 |

## 5. What is missing

Stated plainly so no claim outruns the evidence:

- **The E4 k=2 reference arm was never run.** `EXP_B4_E4_K2_S32_R1_L3_M_SPARSE` has no checkpoint and no metrics. Expert count is bracketed by E2 (0.0193) and E8 (0.0195), which differ by less than the seed spread, so the E4 k=2 value lies in that band; the top-k and load-balance comparisons are stated against it.
- **Comparisons against prior methods are cited, not measured.** Every number in this file is this project's own. The published WXSOD tables give 17 methods plus WFANet on the same two splits and metrics, so a comparison table is possible as reported -- but no external method has been re-run here.
- **Single seed per configuration** except the REPRO seed repeats (42, 43, 44), which give a spread of about +/-0.0004 MAE. A difference smaller than that cannot be interpreted; the S40 arms have no repeat at all.
- **The backbone axis is not testable as configured.** `config.model.backbone` does not reach the model — `src/model.py` hard-codes `pvt_v2_b4` — so the generated ablation arms that vary it would train identical architectures. Do not report a backbone comparison until this is wired through.

## 6. Source files

- `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_real_20260923_060513.json`
- `results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_sys_20260923_060210.json`
- `results/EXP_B4_E2_K2_S32_R1_L3_M_SPARSE/eval_results/best/test_real/hflip/metrics_test_real_20260925_225059.json`
- `results/EXP_B4_E2_K2_S32_R1_L3_M_SPARSE/eval_results/best/test_sys/hflip/metrics_test_sys_20260925_225624.json`
- `results/EXP_B4_E4_K1_S32_R1_L3_M_SPARSE/eval_results/best/test_real/hflip/metrics_test_real_20260925_231219.json`
- `results/EXP_B4_E4_K1_S32_R1_L3_M_SPARSE/eval_results/best/test_sys/hflip/metrics_test_sys_20260925_231817.json`
- `results/EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL/eval_results/best/test_real/hflip/metrics_test_real_20260926_040352.json`
- `results/EXP_B4_E4_K2_S32_R1_L3_M_DENSE_E4_DENSECTRL/eval_results/best/test_sys/hflip/metrics_test_sys_20260926_040946.json`
- `results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/eval_results/best/test_real/hflip/metrics_test_real_20260926_040401.json`
- `results/EXP_B4_E4_K2_S32_R1_L3_M_NONE_E4_NONECTRL/eval_results/best/test_sys/hflip/metrics_test_sys_20260926_040952.json`
- `results/EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB/eval_results/best/test_real/hflip/metrics_test_real_20260925_231534.json`
- `results/EXP_B4_E4_K2_S32_R1_L3_M_SPARSE_NOLB/eval_results/best/test_sys/hflip/metrics_test_sys_20260925_232119.json`
- `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_real_20260923_060823.json`
- `results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results/metrics_test_sys_20260923_060508.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/eval_results/best/test_real/hflip/metrics_test_real_20260924_044224.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO/eval_results/best/test_sys/hflip/metrics_test_sys_20260924_043703.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE/eval_results/best/test_real/hflip/metrics_test_real_20260924_044906.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRODENSE/eval_results/best/test_sys/hflip/metrics_test_sys_20260924_044342.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14/eval_results/best/test_real/hflip/metrics_test_real_20260926_015530.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_E14/eval_results/best/test_sys/hflip/metrics_test_sys_20260926_020154.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43/eval_results/best/test_real/hflip/metrics_test_real_20260924_044342.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED43/eval_results/best/test_sys/hflip/metrics_test_sys_20260924_043819.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44/eval_results/best/test_real/hflip/metrics_test_real_20260924_041616.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_REPRO_SEED44/eval_results/best/test_sys/hflip/metrics_test_sys_20260924_041058.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1/eval_results/best/test_real/hflip/metrics_test_real_20260926_040405.json`
- `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE_SCALE1/eval_results/best/test_sys/hflip/metrics_test_sys_20260926_040935.json`
- `results/legacy_8expert_best/none/legacy_best/test_real/none/metrics_test_real_20260925_193945.json`
- `results/legacy_8expert_best/none/legacy_best/test_sys/none/metrics_test_sys_20260925_194231.json`
