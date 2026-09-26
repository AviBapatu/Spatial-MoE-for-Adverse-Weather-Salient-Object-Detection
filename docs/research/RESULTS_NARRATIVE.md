# RESULTS NARRATIVE — Scientific Interpretation of Completed Experiments

**Audit date:** 2026-09-02
**Purpose:** Concise scientific interpretation of what the experiments actually show.

---

## 1. What the Baseline Demonstrates

The SpatialMoESODNet — a PVTv2-B4 backbone with three independent spatial MoE layers (8 experts each, top-k=2) and a cross-attention decoder with entropy fusion — was trained on WXSOD synthetic training data and evaluated on two held-out test sets.

**Core results (single model, single run):**

| Test Set | N | MAE | S_measure | F_max |
|----------|---|-----|-----------|-------|
| Synthetic (test_sys) | 1,500 | 0.0192 | 0.9139 | 0.9015 |
| Real-world (test_real) | 554 | 0.0168 | 0.9151 | 0.8936 |

The model processes images at 384×384 resolution and produces pixel-wise saliency maps plus boundary maps. It has **69.21M parameters** by direct enumeration (69,213,120 for the reported recipe, which has deep supervision on; 69,212,349 with it off; the legacy `compute_cost.json` reports 68.9M, which no configuration reproduces) and **277.9G MACs** at 384×384 (the current measurement; the legacy file says 278.2G).

**What this establishes:** The architecture is functional and produces reasonable SOD predictions across diverse weather conditions. The evaluation pipeline is deterministic (identical metrics across 6 independent evaluation runs of the same checkpoint).

**What this does NOT establish:** Whether any specific design choice (MoE, routing, entropy fusion, multi-scale independent routing) contributes to this performance. No ablation against a simpler baseline exists.

---

## 2. What the Weather-Wise Analysis Shows

### 2.1 Synthetic Test (9 categories)

The model handles single-weather conditions well (clean MAE=0.0142, fog MAE=0.0166, snow MAE=0.0189) but degrades on compound weather (rainafog MAE=0.0237). The worst synthetic condition (rainafog) is 1.67× harder than the best (clean).

### 2.2 Real-World Test (5 categories)

Snow is easiest (MAE=0.0094, S=0.9478). Low-light is hardest (MAE=0.0243, S=0.8897). The real-world performance spread (2.6× between best and worst) is larger than the synthetic spread (1.7×), suggesting real-world conditions present more varied difficulty.

### 2.3 Interpretation

The weather-wise breakdown reveals that the model's performance is **not uniform** across conditions. This is expected — different weather types create different visual degradations. The key question is whether the MoE architecture specifically addresses this non-uniformity, and the current experiments cannot answer that question.

---

## 3. What the Forced-Expert Ablation Shows

Forcing all tokens at scale 4 to expert 0 (bypassing the router) barely moves the global
metrics. In the tracked proxy file
(`results/legacy/legacy_8expert/eval_results/proxy_ablation_results.json`) MAE goes from
0.016841 to 0.016923 (+0.0001, +0.5%) and S_measure from 0.915091 to 0.914655
(-0.0004, -0.05%). The per-condition breakdown of the forced run is not present in the
tracked result files, so no weather-wise claim about it is made here. (The manuscript quotes
a slightly larger delta, +0.0002 / +0.9%; that pair is not reproducible from any file under
`results/` and should be reconciled before submission.)

**This is a surprisingly small degradation.** Two possible interpretations:

1. **Optimistic:** The router is not critical because the expert pool is over-parameterized — even a single expert can handle most tokens adequately.
2. **Pessimistic:** The router is not learning meaningful specialization — experts may be learning similar functions, making routing redundant.

The forced-expert test only affects one of three scales. The full effect of disabling all routing is unknown.

**Cannot determine** which interpretation is correct without per-expert output analysis.

---

## 4. What the Entropy Data Shows

Mean routing entropy across all tokens is 0.68-0.69 nats in the legacy measurement
(`results/legacy/legacy_8expert/eval_results/entropy_comparison.json`), which is the ceiling
`ln 2 = 0.6931` of a two-way gate distribution. That file predates the fix to the entropy
computation: `SpatialMoELayer` now takes the entropy of the full softmax over all E router
logits, range `[0, ln E]`, and under that code the same checkpoint measures 2.076-2.079 nats
against `ln 8 = 2.0794`
(`results/legacy/legacy_8expert/routing_entropy/entropy_comparison.json`), i.e. 99.8-100% of
the ceiling. Both readings say the same thing. This indicates:
- Tokens are NOT being routed to a single expert (entropy would be 0)
- Tokens are routed close to uniformly (entropy sits at its maximum)
- The router is making almost no distinction between experts

Entropy is very similar between synthetic and real data (difference < 0.003 at all scales), suggesting the router's behavior generalizes across domains.

Scale 1/16 has marginally lower entropy (2.0756 nats) than scales 1/4 and 1/8 (2.0788 and
2.0789 nats) in the current-code measurement, i.e. the coarsest scale's routing is very
slightly less uniform. The difference is small and no significance test has been run.

**Cannot claim** that entropy correlates with difficulty or that the entropy fusion in the decoder provides benefit.

---

## 5. What Is Surprising

1. **Real-world MAE is lower than synthetic MAE.** This is counterintuitive — real-world weather should be harder than controlled synthetic weather. Possible explanations: (a) the synthetic test set includes harder compound conditions, (b) real-world images have characteristics that make SOD easier despite weather, (c) the synthetic augmentation process creates harder examples than natural weather.

2. **Forced-expert degradation is minimal.** If routing were critical, forcing all tokens to one expert should cause substantial degradation. The small effect suggests either over-parameterization or lack of specialization.

3. **Snow is the easiest condition on real data.** Snow images may have high contrast between the white snow and darker salient objects, making segmentation easier.

4. **Boundary F1 is low across the board.** 0.36 on real, 0.55 on synthetic. The boundary head appears to be the weakest component.

---

## 6. Design Choices That Are Empirically Supported

**None.** No design choice has been empirically validated through ablation. All architectural decisions (MoE, routing, entropy fusion, multi-scale independent routing) remain unvalidated design choices.

---

## 7. Design Choices That Are NOT Supported

Cannot be determined without ablation studies. The current experiments provide zero evidence for or against any specific design choice.

---

## 8. Limitations

1. **Single model, single run.** No variance estimation. No multiple seeds. Results could be specific to this particular training run.

2. **No ablation studies.** The most critical gap. Without ablations, no causal claims can be made about any design choice.

3. **No SOTA comparison.** Cannot claim competitive or superior performance.

4. **No interpretability analysis.** Cannot claim experts specialize or that routing reflects meaningful decisions.

5. **Boundary performance is weak.** Boundary F1 of 0.36 (real) is low and may need to be disclosed as a limitation.

6. **Low-light is a weakness.** MAE of 0.0243 on real-world low-light images is 44% worse than the overall average.

7. **No training logs.** Cannot assess convergence, overfitting, or training stability.

8. **No non-MoE baseline.** Cannot attribute any performance to the MoE architecture.

---

## 9. Honest Summary for the Paper

The completed experiments demonstrate:

1. A functional architecture that produces reasonable SOD predictions across diverse weather conditions.
2. Quantitative performance on two test sets with weather-wise breakdowns.
3. That the model handles synthetic-to-real transfer without severe degradation (observational).
4. Basic computational cost metrics (parameters, MACs, FPS).

The completed experiments do NOT demonstrate:

1. That any specific design choice causes improvement.
2. That MoE is better than non-MoE for this task.
3. That routing is meaningful or that experts specialize.
4. That the method outperforms existing approaches.
5. That the model is computationally efficient.
6. That entropy fusion provides benefit.

**The paper should be framed as an architectural proposal with comprehensive empirical characterization, not as a validated design with proven contributions.**


## Provenance note

`best.pth` and `best_new_1.pth` are the same checkpoint (identical file hash), and
MAE / S-measure / E-* / F-* are byte-identical across every evaluation run of that model,
so the evaluation pipeline is deterministic. The differing boundary numbers under
`legacy_8expert/evaluation_results/best/` come from an earlier scoring pass; the
timestamped re-run is canonical. Source: `results/legacy/legacy_8expert/`.
