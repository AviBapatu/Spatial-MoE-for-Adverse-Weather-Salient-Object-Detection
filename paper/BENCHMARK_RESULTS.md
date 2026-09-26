# Published WXSOD benchmark numbers, and how this project compares

Transcribed from the WXSOD paper: Chen et al., *A Benchmark for Robust Salient Object
Detection in Adverse Weather Conditions*, Pattern Recognition vol. 170 (2026),
DOI 10.1016/j.patcog.2026.114003, arXiv 2508.12250. Tables II (synthesized test set) and
III (real test set) of that paper.

**These are the published authors' own numbers, reproduced by them under their own
protocol.** They are not measurements from this repository. Any table in our paper that uses
them must label them "as reported" and must disclose the protocol difference: the WXSOD
paper measures on original images with each method run at its own default settings, whereas
this project predicts at 384x384 and then reverse-geometries predictions back to original
resolution before scoring. Both score in original image coordinates.

## Table II — synthesized test set (1,500 images)

| Method | MAE | S | F_adaptive | F_mean | F_max | E_adaptive | E_mean | E_max | Params (M) | MACs (G) |
|---|---|---|---|---|---|---|---|---|---|---|
| MINet | 0.0775 | 0.7403 | 0.6207 | 0.6219 | 0.6501 | 0.8130 | 0.7976 | 0.8263 | 0.36 | 0.27 |
| FSMINet | 0.0473 | 0.8306 | 0.7558 | 0.7635 | 0.7805 | 0.8781 | 0.8693 | 0.8808 | 3.56 | 11.82 |
| ADMNet | 0.0770 | 0.7497 | 0.6349 | 0.6349 | 0.6594 | 0.8213 | 0.8061 | 0.8307 | 0.94 | 0.83 |
| TRACER | 0.0280 | 0.8898 | 0.8285 | 0.8473 | 0.8671 | 0.9310 | 0.9319 | 0.9414 | 3.89 | 3.12 |
| SEANet | 0.0332 | 0.8694 | 0.8129 | 0.8210 | 0.8391 | 0.9201 | 0.9136 | 0.9240 | 2.74 | 3.21 |
| CorrNet | 0.0613 | 0.7652 | 0.7378 | 0.6911 | 0.7270 | 0.8403 | 0.7729 | 0.8281 | 4.06 | 47.95 |
| A3Net | 0.0327 | 0.8784 | 0.8158 | 0.8286 | 0.8472 | 0.9176 | 0.9131 | 0.9253 | 22.70 | 27.94 |
| HDNet | 0.0397 | 0.8537 | 0.7875 | 0.7964 | 0.8162 | 0.8990 | 0.8901 | 0.9053 | 24.67 | 26.19 |
| AESINet | 0.0337 | 0.8618 | 0.8075 | 0.8141 | 0.8247 | 0.9096 | 0.9039 | 0.9085 | 41.04 | 155.12 |
| ICONet | 0.0433 | 0.8411 | 0.7496 | 0.7704 | 0.7957 | 0.8924 | 0.8925 | 0.9078 | 33.03 | 24.95 |
| DCNet | 0.0320 | 0.8792 | 0.8336 | 0.8372 | 0.8498 | 0.9179 | 0.9086 | 0.9221 | 88.95 | 120.29 |
| MEANet | 0.0332 | 0.8772 | 0.8174 | 0.8294 | 0.8502 | 0.9225 | 0.9185 | 0.9288 | 3.27 | 13.48 |
| SAFINet | 0.0384 | 0.8643 | 0.8049 | 0.8105 | 0.8303 | 0.9073 | 0.8982 | 0.9120 | 3.12 | 13.67 |
| MSRMNet | 0.0275 | 0.8895 | 0.8361 | 0.8493 | 0.8657 | 0.9267 | 0.9265 | 0.9346 | 89.66 | 54.30 |
| TCGNet | 0.0317 | 0.8832 | 0.8278 | 0.8388 | 0.8576 | 0.9228 | 0.9180 | 0.9288 | 70.26 | 60.14 |
| GeleNet | 0.0239 | 0.9038 | 0.8538 | 0.8675 | 0.8868 | 0.9418 | 0.9410 | 0.9499 | 25.45 | 13.91 |
| GPONet | 0.0266 | 0.8962 | 0.8445 | 0.8568 | 0.8774 | 0.9352 | 0.9318 | 0.9425 | 24.86 | 11.52 |
| **WFANet (their baseline)** | **0.0229** | **0.9051** | 0.8601 | 0.8713 | 0.8888 | **0.9464** | **0.9443** | **0.9523** | 50.87 | 112.63 |

## Table III — real test set (554 images)

| Method | MAE | S | F_adaptive | F_mean | F_max | E_adaptive | E_mean | E_max | Params (M) | MACs (G) |
|---|---|---|---|---|---|---|---|---|---|---|
| MINet | 0.0549 | 0.7710 | 0.6265 | 0.6425 | 0.6748 | 0.8278 | 0.8205 | 0.8442 | 0.36 | 0.27 |
| FSMINet | 0.0264 | 0.8869 | 0.8123 | 0.8284 | 0.8541 | 0.9239 | 0.9231 | 0.9353 | 3.56 | 11.82 |
| ADMNet | 0.0544 | 0.7839 | 0.6676 | 0.6683 | 0.6921 | 0.8503 | 0.8262 | 0.8554 | 0.94 | 0.83 |
| TRACER | 0.0227 | 0.8961 | 0.8225 | 0.8390 | 0.8586 | 0.9230 | 0.9224 | 0.9322 | 3.89 | 3.12 |
| SEANet | 0.0239 | 0.8978 | 0.8192 | 0.8390 | 0.8706 | 0.9261 | 0.9290 | 0.9408 | 2.74 | 3.21 |
| CorrNet | 0.0534 | 0.7674 | 0.7288 | 0.6629 | 0.7223 | 0.8363 | 0.7488 | 0.8268 | 4.06 | 47.95 |
| A3Net | 0.0263 | 0.8909 | 0.8172 | 0.8350 | 0.8586 | 0.9207 | 0.9199 | 0.9329 | 22.70 | 27.94 |
| HDNet | 0.0266 | 0.8938 | 0.8179 | 0.8447 | 0.8734 | 0.9163 | 0.9219 | 0.9361 | 24.67 | 26.19 |
| AESINet | 0.0251 | 0.8845 | 0.8202 | 0.8359 | 0.8602 | 0.9267 | 0.9304 | 0.9367 | 41.04 | 155.12 |
| ICONet | 0.0327 | 0.8619 | 0.7595 | 0.7885 | 0.8209 | 0.9007 | 0.9117 | 0.9276 | 33.03 | 24.95 |
| DCNet | 0.0215 | 0.9058 | 0.8564 | 0.8672 | 0.8847 | 0.9425 | 0.9357 | 0.9456 | 88.95 | 120.29 |
| MEANet | 0.0236 | 0.9063 | 0.8284 | 0.8472 | 0.8803 | 0.9293 | 0.9286 | 0.8418 | 3.27 | 13.48 |
| SAFINet | 0.0217 | 0.9050 | 0.8274 | 0.8438 | 0.8723 | 0.9265 | 0.9242 | 0.9407 | 3.12 | 13.67 |
| MSRMNet | 0.0198 | 0.9091 | 0.8595 | 0.8716 | 0.8881 | 0.9453 | 0.9432 | 0.9516 | 89.66 | 54.30 |
| TCGNet | 0.0185 | 0.9185 | 0.8529 | 0.8739 | 0.8992 | 0.9429 | 0.9476 | 0.9594 | 70.26 | 60.14 |
| GeleNet | 0.0192 | 0.9186 | 0.8442 | 0.8698 | 0.8932 | 0.9316 | 0.9425 | 0.9528 | 25.45 | 13.91 |
| GPONet | 0.0188 | 0.9098 | 0.8434 | 0.8604 | 0.8830 | 0.9362 | 0.9364 | 0.9468 | 24.86 | 11.52 |
| **WFANet (their baseline)** | **0.0159** | **0.9248** | 0.8617 | 0.8828 | 0.9080 | **0.9494** | **0.9529** | **0.9617** | 50.87 | 112.63 |

WFANet uses a **weather prediction branch trained with weather class labels** (a ResNet18
classifier over 9 weather categories, cross-entropy loss). It is therefore a
weather-supervised method. This project's model uses **no weather labels at training or
inference** — that difference is the paper's central claim, and it is a genuine like-for-like
comparison in every other respect (same dataset, same splits, same metrics, same 384x384
input convention on the benchmark's side).

## This project's numbers, for the same two splits

Certified on 2026-09-25 against the recovered checkpoint, under current code, both TTA
settings. Source: `results/legacy_8expert_best/{none,hflip}/legacy_best/`.

| model | TTA | test_sys MAE | test_sys S | test_real MAE | test_real S |
|---|---|---|---|---|---|
| E8 k=2, 14 epochs (reported) | none | **0.0192** | 0.9140 | **0.0168** | 0.9151 |
| E8 k=2, 14 epochs (reported) | hflip | 0.0191 | 0.9166 | 0.0167 | 0.9185 |
| E8 k=2, 14 epochs (this project's own 14-ep run) | hflip | 0.0202 | 0.9141 | 0.0174 | 0.9169 |

Cost of this project's model, measured: **69.21 M parameters, 277.9 G MACs** at 384x384
(the earlier 66.27 M figure is wrong for every architecture variant; see
`RESEARCH_TRUTH.md`).

## What this means

- **Synthesized split:** 0.0192 beats all 17 comparison methods **and** WFANet (0.0229).
- **Real split:** 0.0168 is second of nineteen, behind only WFANet (0.0159) and ahead of
  TCGNet (0.0185), GPONet (0.0188) and GeleNet (0.0192).
- The model is larger than WFANet (69.21 M vs 50.87 M) and heavier (277.9 G vs 112.63 G).
  State that plainly; it is the honest trade.
- **The claim this supports is "a label-free model matches or beats a weather-supervised
  baseline on this benchmark", not "our method is state of the art."** WFANet still wins the
  real split.
