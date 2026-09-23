# PAPER_CORRECTIONS.md — Audit of paper/main.tex Against Code and Results

> **Note.** This file audits a specific revision of `paper/main.tex`. Citations to
> source files refer to the layout at the time of the audit; module paths have since
> changed (the decoder is now the `src/decoder/` package, training code lives under
> `src/training/`). Treat the findings as issues to check, not as current fact.

**Audit date:** 2026-09-03
**Paper version:** paper/main.tex (v1)
**Authority:** Source code, evaluation results, configuration files, docs/research/

---

## Issue 1

PAPER CURRENTLY SAYS:
Eq. 3: $W_1 \in \mathbb{R}^{128 \times 2C}$, $W_2 \in \mathbb{R}^{E \times 128}$, with "The hidden dimension 128 provides a bottleneck between the $2C{=}512$ input and $E{=}8$ output."

PROBLEM:
The router MLP hidden dimension (128) is hardcoded in the `SpatialMoELayer` constructor default argument (`router_hidden=128`) and is never overridden by the baseline config (`experiments/baseline_v1.json`). The paper states it as a factual architectural specification but does not mention it is a fixed default, nor does the config expose it as a tunable hyperparameter. This is not wrong, but is misleadingly precise for a value that is buried as a constructor default rather than a documented design choice.

ACTUAL FACT:
`src/moe_layer.py:36` — `def __init__(self, dim=256, num_experts=8, k=2, router_hidden=128, ...)` — the value 128 is a hardcoded default. `src/model.py:14` instantiates `SpatialMoELayer(dim=dim, num_experts=num_experts, k=k)` without passing `router_hidden`, so the default of 128 is always used in the baseline. The config file does not contain a `router_hidden` field.

WHAT PAPER SHOULD SAY:
The equation and description are numerically correct. No change needed to the equation itself. However, for full transparency, the paper should note that the router hidden dimension is a fixed architectural constant (128), not a tuned hyperparameter, or should add it to the config and document it explicitly.

SOURCE:
`src/moe_layer.py:36`, `experiments/baseline_v1.json`

SEVERITY:
MINOR

---

## Issue 2

PAPER CURRENTLY SAYS:
Section III-F, Eq. 8 description: "Routing entropy has been explored for uncertainty estimation in MoE classifiers, but not as an explicit decoder feature for dense prediction."

PROBLEM:
This novelty claim is stated as fact but has not been empirically validated. The paper correctly qualifies this in the Discussion and Limitations sections, but the Related Work section presents it as an established differentiator without noting it is unvalidated. The entropy fusion mechanism exists in the code and is active during training, but no ablation (entropy ON vs. OFF) has been run to demonstrate that it actually improves performance. The forced-expert ablation, which sets entropy to 0 for the forced scale, shows only marginal degradation — but this test does not isolate the entropy fusion effect because it also changes the routing pattern.

ACTUAL FACT:
`src/decoder.py:14-27` — `EntropyFusionBlock` is implemented and active. `experiments/baseline_v1.json:33` — `deep_supervision` is false but the entropy fusion is always on (no config flag to disable it). `results/legacy/legacy_8expert/evaluation_results/force_expert_ablation.json` — forcing expert 0 at scale 4 sets entropy to 0 at that scale, but the overall MAE change is only +0.0002, which cannot distinguish entropy-fusion effect from routing-pattern effect. No ON/OFF ablation of entropy fusion alone exists.

WHAT PAPER SHOULD SAY:
"Routing entropy has been explored for uncertainty estimation in MoE classifiers. We inject it as an explicit decoder feature for dense prediction, though we have not yet isolated its contribution via ablation."

SOURCE:
`src/decoder.py:14-27`, `results/legacy/legacy_8expert/evaluation_results/force_expert_ablation.json`

SEVERITY:
MAJOR — Overclaims a contribution that has not been empirically demonstrated.

---

## Issue 3

PAPER CURRENTLY SAYS:
Contributions list, item 3: "Routing entropy, normalized by $\log K$, injected as an explicit decoder feature via learnable additive fusion."

PROBLEM:
The paper states $\log K$ but the code normalizes by $\log 2$ (a constant), not $\log K$ where $K$ is the actual top-k value. Since $K{=}2$ in the baseline, $\log K = \log 2$, so the result is the same. However, the formulation implies the normalization adapts to different $K$ values, which it does not — the code hardcodes `math.log(2.0)`.

ACTUAL FACT:
`src/decoder.py:23` — `entropy_norm = entropy / math.log(2.0 + 1e-8)`. This divides by $\log 2$ regardless of the actual $K$ value. If $K$ were changed to 3, the normalization would still use $\log 2$, producing values in $[0, \log 3 / \log 2] \approx [0, 1.58]$ rather than $[0, 1]$.

WHAT PAPER SHOULD SAY:
"Routing entropy, normalized by $\log 2$ (the maximum entropy for $K{=}2$), injected as an explicit decoder feature via learnable additive fusion." Or, if generalization to other $K$ is intended, the code should be changed to divide by $\log K$ and the paper should note this.

SOURCE:
`src/decoder.py:23`

SEVERITY:
MINOR — Correct for the reported configuration ($K{=}2$), but misleading if interpreted as a general formula.

---

## Issue 4

PAPER CURRENTLY SAYS:
Section III-G (Decoder): "Global context is then injected into the fine scale: $\bF_4^{\text{ctx}} = \bF_4 + \text{Proj}_g(\text{AvgPool}(\bF_8'))$." The text implies this happens before the 1/8→1/4 cross-attention.

PROBLEM:
The paper's prose describes the global context injection as happening between the 1/16→1/8 fusion and the 1/8→1/4 fusion, which is correct. However, the description could be clearer about the exact ordering. In the code, the 1/8→1/4 cross-attention (Eq. 11) uses `F4_ctx` as KV, and `F4_ctx` is computed as `F4_local + g_ctx` where `g_ctx` is derived from `F8` (the output of the 1/16→1/8 cross-attention). The paper's Eq. 10 and Eq. 11 are in the correct order. The prose description is accurate.

ACTUAL FACT:
`src/decoder.py:252-259`:
1. `F8 = cross_attn_16_to_8(q=F16_up, kv=F8_local)` (Eq. 10)
2. `g_ctx = global_context_proj(global_context_pool(F8))`
3. `F4_ctx = F4_local + g_ctx`
4. `F4 = cross_attn_8_to_4(q=F8_up, kv=F4_ctx)` (Eq. 11)

The paper's equation order is correct. The prose description at line 403 is also correct.

WHAT PAPER SHOULD SAY:
No change needed. The equations and prose are accurate.

SOURCE:
`src/decoder.py:252-259`

SEVERITY:
MINOR (no actual error — included for completeness since the audit flagged the ordering for verification)

---

## Issue 5

PAPER CURRENTLY SAYS:
Section IV-A (Setup): "Training uses DDP on 2 GPUs, AMP (FP16), AdamW optimizer (lr $10^{-4}$, weight decay $10^{-4}$), WarmupCosine schedule (1% warmup, 50 epochs), gradient clipping at 1.0, effective batch size 32."

PROBLEM:
The paper does not mention that the backbone is frozen during epoch 0 and unfrozen from epoch 1 onward. This is a non-trivial training detail that affects reproducibility and training dynamics.

ACTUAL FACT:
`src/train_ddp.py:382-388` — backbone is frozen during epoch 0 (`freeze_backbone(model.module)`) and unfrozen at epoch 1 (`unfreeze_backbone(model.module)`). `experiments/baseline_v1.json:50` — `"freeze_backbone_epochs": 1`.

WHAT PAPER SHOULD SAY:
"Training uses DDP on 2 GPUs, AMP (FP16), AdamW optimizer (lr $10^{-4}$, weight decay $10^{-4}$), WarmupCosine schedule (1% warmup, 50 epochs), gradient clipping at 1.0, effective batch size 32. The backbone is frozen during the first epoch for warmup, then unfrozen for the remainder of training."

SOURCE:
`src/train_ddp.py:382-388`, `experiments/baseline_v1.json:50`

SEVERITY:
MAJOR — Omitted a non-trivial training detail that affects reproducibility.

---

## Issue 6

PAPER CURRENTLY SAYS:
Section IV-B (Main Results), paragraph 2: "Real-world MAE is comparable to (and slightly better than) synthetic MAE, which suggests effective feature transfer across domains."

PROBLEM:
The paper correctly qualifies this as "observational" later in the same paragraph, but the initial sentence "suggests effective feature transfer across domains" is a causal interpretation that is not supported by the evidence. The observation that real-world MAE (0.0168) is lower than synthetic MAE (0.0192) could equally reflect that the synthetic test set contains harder compound weather conditions (rainafog, rainasnow, snowafog) not present in the real test set, rather than any "feature transfer." The paper's own caveat partially addresses this, but the initial framing leads the reader toward an unsupported interpretation.

ACTUAL FACT:
`results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/metrics.json` — test_sys has 9 weather categories including compound conditions (rainafog MAE=0.0237, rainasnow MAE=0.0194, snowafog MAE=0.0189). `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json` — test_real has only 5 single-degradation categories. The synthetic test set is compositionally harder, which alone explains the MAE difference without invoking "feature transfer."

WHAT PAPER SHOULD SAY:
"Real-world MAE (0.0168) is slightly lower than synthetic MAE (0.0192). However, this comparison is confounded by test set composition: the synthetic test set includes compound weather conditions (rain+fog, rain+snow, snow+fog) absent from the real test set, which may inflate the synthetic MAE independently of any domain gap."

SOURCE:
`results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/metrics.json`, `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json`

SEVERITY:
MAJOR — Misleading causal interpretation of an observational comparison.

---

## Issue 7

PAPER CURRENTLY SAYS:
Section IV-C (Weather-Wise Analysis): "Performance varies by $2.6{\times}$ across weather types: snow is easiest (MAE~0.0094, $N{=}90$), likely due to high foreground-background contrast, while low-light is hardest (MAE~0.0243, $N{=}93$), where reduced signal-to-noise ratio degrades feature extraction."

PROBLEM:
The paper attributes snow's low MAE to "high foreground-background contrast" and low-light's high MAE to "reduced signal-to-noise ratio." These are plausible hypotheses but are presented as explanations without supporting evidence. No analysis of foreground-background contrast or signal-to-noise ratio across weather categories has been performed.

ACTUAL FACT:
The weather-wise metrics show the MAE variation (snow=0.0094, light=0.0243), but no analysis of image statistics (contrast, SNR, etc.) per weather category exists in the repository. These are post-hoc hypotheses, not verified explanations.

WHAT PAPER SHOULD SAY:
"Performance varies by $2.6{\times}$ across weather types: snow yields the lowest MAE (0.0094, $N{=}90$) while low-light yields the highest (0.0243, $N{=}93$). Fog (0.0148, $N{=}126$), rain (0.0165, $N{=}120$), and dark (0.0191, $N{=}125$) fall between these extremes." Remove the speculative contrast/SNR explanations, or mark them explicitly as hypotheses.

SOURCE:
`results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json`

SEVERITY:
MINOR — Speculative explanations presented as plausible interpretations. Acceptable if marked as hypotheses.

---

## Issue 8

PAPER CURRENTLY SAYS:
Section IV-D (Routing Entropy): "Values of $0.68$--$0.69$ indicate moderate uncertainty: the routers neither collapse to a single expert (entropy~$\to 0$) nor distribute uniformly (entropy~$\to 1$). Scale $1/16$ is slightly more confident (lower entropy), possibly because coarser spatial resolution reduces the need for specialized routing."

PROBLEM:
The paper states scale 1/16 is "more confident (lower entropy)" and attributes this to "coarser spatial resolution reduc[ing] the need for specialized routing." This is a speculative interpretation presented as a possible explanation. Additionally, the actual entropy values across scales are very close (0.6800–0.6931), and the difference at scale 1/16 (~0.01 below the others) is small enough that it may not be meaningful. The paper does not quantify whether this difference is statistically significant.

ACTUAL FACT:
`results/legacy/legacy_8expert/evaluation_results/entropy_comparison.json` — Scale 1/16 synthetic: 0.679954, real: 0.682216. Scale 1/4: ~0.691, scale 1/8: ~0.693. The difference between scale 1/16 and the others is ~0.01, which is small. No statistical test has been performed.

WHAT PAPER SHOULD SAY:
"Mean normalized entropy values range from 0.68 to 0.69 across scales, indicating moderate routing uncertainty. Scale $1/16$ has marginally lower entropy (0.680–0.682) than scales $1/4$ (0.691) and $1/8$ (0.693), though the difference is small."

SOURCE:
`results/legacy/legacy_8expert/evaluation_results/entropy_comparison.json`

SEVERITY:
MINOR — Speculative interpretation of a small difference.

---

## Issue 9

PAPER CURRENTLY SAYS:
Abstract: "On WXSOD, the model achieves MAE $0.0192$ on 1,500 synthetic and $0.0168$ on 554 real-world test images across five weather categories."

PROBLEM:
The abstract says "across five weather categories" which is only accurate for the real-world test set (which has 5 categories: dark, fog, light, rain, snow). The synthetic test set has 9 weather categories (clean, dark, fog, light, rain, rainafog, rainasnow, snow, snowafog). The phrasing conflates the two test sets.

ACTUAL FACT:
`results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/metrics.json` — 9 weather categories. `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json` — 5 weather categories.

WHAT PAPER SHOULD SAY:
"On WXSOD, the model achieves MAE $0.0192$ on 1,500 synthetic test images (9 weather categories) and $0.0168$ on 554 real-world test images (5 weather categories)."

SOURCE:
`results/legacy/legacy_8expert/evaluation/best_new_1/test_sys/none/metrics.json`, `results/legacy/legacy_8expert/evaluation/best_new_1/test_real/none/metrics.json`

SEVERITY:
MINOR — Ambiguous phrasing that could mislead about the synthetic test coverage.

---

## Issue 10

PAPER CURRENTLY SAYS:
Section III-H (Training Objective): "$\mathcal{L}_{\text{IMP}}$ is the importance loss: $\frac{1}{|\mathcal{S}|}\sum_{s} (\text{std}(\bP^{(s)}) / \text{mean}(\bP^{(s)}))^2$."

PROBLEM:
The paper describes the importance loss as computing the coefficient of variation of $\bP^{(s)}$ (the routing probability vector over experts). This implies computing std/mean across the $E{=}8$ expert dimension. However, the code computes std/mean across all $E \times C$ elements of `P_j_sum` (the unnormalized gate sums), not across the $E$ expert-level mean probabilities. The `P_j_sum` tensor has shape `[E]` (one value per expert), but `std_imp` and `mean_imp` are computed on this `[E]` tensor — so the computation is actually across experts, matching the paper's description. The issue is that the paper uses the symbol $\bP^{(s)}$ which could be misinterpreted as the per-token routing probability matrix rather than the per-expert mean gate mass vector.

ACTUAL FACT:
`src/loss.py:141-143`:
```python
mean_imp = P_j_sum.mean()          # mean across E experts
std_imp = P_j_sum.std(unbiased=False)  # std across E experts
l_imp = (std_imp / (mean_imp + 1e-6))**2
```
`P_j_sum` has shape `[E]` — it is the sum of gate weights assigned to each expert. So the computation IS across the $E$ expert dimension, which matches the paper's description. The notation $\bP^{(s)}$ in the paper is ambiguous but the formula is correct.

WHAT PAPER SHOULD SAY:
The formula is correct. For clarity, the notation should specify that $\bP^{(s)} \in \mathbb{R}^E$ is the per-expert mean gate mass vector (i.e., $P_j^{(s)} = \frac{1}{N}\sum_i g_{i,j}$ for expert $j$), not the per-token routing probability matrix.

SOURCE:
`src/loss.py:141-143`

SEVERITY:
MINOR — Notation ambiguity, but the formula and implementation match.

---

## Issue 11

PAPER CURRENTLY SAYS:
Section IV-A (Setup): "The model has 66.27M parameters and 278.2G MACs."

PROBLEM:
The paper reports these numbers without specifying the hardware, batch size, or measurement conditions for the MAC count. The MAC count from `compute_cost.json` (278.2G) was computed for a single 384×384 image, but this is not stated. The FPS value (3.83) is also reported in the file but omitted from the paper, which is appropriate given the unspecified hardware.

ACTUAL FACT:
`results/legacy/legacy_8expert/evaluation_results/compute_cost.json` — `{"params_M": 66.27, "macs_G": 278.2, "fps": 3.83}`. No hardware specification is recorded.

WHAT PAPER SHOULD SAY:
"The model has 66.27M parameters and 278.2G MACs per $384{\times}384$ image."

SOURCE:
`results/legacy/legacy_8expert/evaluation_results/compute_cost.json`

SEVERITY:
MINOR — Missing the "per image" qualifier.

---

## Issue 12

PAPER CURRENTLY SAYS:
Section II-B (MoE for Weather Restoration): "WM-MoE~\cite{wmmoe2023} performs token-level routing via a weather-aware router; it does not require weather labels at test time but uses a dedicated weather feature branch during training."

PROBLEM:
The claim that WM-MoE "does not require weather labels at test time" is stated without a citation to the WM-MoE paper itself. The literature notes (`LITERATURE_NOTES.md`) record this claim, but it comes from the blueprint, not from direct verification of the WM-MoE paper. If the WM-MoE paper actually does require weather labels at test time, this claim would be wrong and would undermine a key differentiator of the current work.

ACTUAL FACT:
`docs/research/LITERATURE_NOTES.md:25-26` — "Routes on feature maps with weather-conditioned gating" and "NOT per-token, weather-agnostic routing." The notes say WM-MoE uses "weather-conditioned gating" during training, but it is unclear whether this conditioning is needed at test time. The `spatial-moe-adverse-weather-sod-blueprint.md` would need to be checked against the actual WM-MoE paper.

WHAT PAPER SHOULD SAY:
Verify the WM-MoE paper's test-time requirements before making this claim. If WM-MoE does require weather labels at test time, the claim should be "WM-MoE~\cite{wmmoe2023} performs token-level routing via a weather-aware router that conditions on weather labels at both train and test time." If it truly does not require them at test time, the current wording is fine but should cite the specific section of the WM-MoE paper.

SOURCE:
`docs/research/LITERATURE_NOTES.md`, `spatial-moe-adverse-weather-sod-blueprint.md`

SEVERITY:
MAJOR — A factual claim about a cited paper that has not been verified against the primary source.

---

## Issue 13

PAPER CURRENTLY SAYS:
Contribution 1: "A token-level spatial router assigning each token to top-$K{=}2$ experts from 8 per scale, using depthwise convolutional context and Shazeer-style noisy gating with true sparse dispatch."

PROBLEM:
The paper describes the router as using "Shazeer-style noisy gating." The Shazeer (2017) noisy gating adds Gaussian noise scaled by $\text{softplus}(\text{noise\_scale})$ where the noise scale is a learned per-expert parameter. The implementation in this codebase uses a noise scale that is a learned linear projection of the input tokens (`self.noise_linear = nn.Linear(dim, num_experts)`), making it input-dependent and per-token — which is actually closer to the original Shazeer formulation than a simple per-expert parameter. The description is accurate.

ACTUAL FACT:
`src/moe_layer.py:60` — `self.noise_linear = nn.Linear(dim, num_experts)` (input-dependent, per-token noise scale). `src/moe_layer.py:105-106` — `noise_std = self.router_noise_scale * F.softplus(self.noise_linear(x_tokens))`. This matches the Shazeer (2017) formulation where noise is input-dependent.

WHAT PAPER SHOULD SAY:
No change needed. The description is accurate.

SOURCE:
`src/moe_layer.py:60,105-106`

SEVERITY:
N/A (no error)

---

## Issue 14

PAPER CURRENTLY SAYS:
Section IV-E (Forced-Expert Analysis): "MAE increased by $+0.0002$ ($+1.2\%$), $S_\phi$ decreased by $-0.0012$ ($-0.1\%$)."

PROBLEM:
The percentage changes are inaccurate. The MAE increase from 0.016841 to 0.016991 is +0.000150, which is +0.89% (not +1.2%). The S_measure decrease from 0.915091 to 0.913917 is -0.001174, which is -0.13% (not -0.1%). The absolute deltas (+0.0002 and -0.0012) are correctly rounded to 4 decimal places, but the percentages are wrong.

ACTUAL FACT:
`results/legacy/legacy_8expert/evaluation_results/force_expert_ablation.json` — MAE: 0.016991033084321077 (forced) vs 0.016840667104452152 (normal). Delta = +0.000150365979868925 = +0.89%. S_measure: 0.9139173865148662 (forced) vs 0.9150906427309119 (normal). Delta = -0.0011732562160457 = -0.13%.

WHAT PAPER SHOULD SAY:
"MAE increased by $+0.0002$ ($+0.9\%$), $S_\phi$ decreased by $-0.0012$ ($-0.1\%$)."

SOURCE:
`results/legacy/legacy_8expert/evaluation_results/force_expert_ablation.json`

SEVERITY:
MINOR — Percentage values are slightly inaccurate. The absolute deltas are correct.

---

## Issue 15

PAPER CURRENTLY SAYS:
Section III-H (Training Objective): "$\mathcal{L}_{\text{LB}}$ is the load-balancing loss: $\frac{1}{|\mathcal{S}|}\sum_{s} E \sum_j f_j^{(s)} P_j^{(s)}$, where $f_j$ is the detached hard assignment fraction and $P_j$ the differentiable mean gate mass."

PROBLEM:
The description of $f_j$ as "the detached hard assignment fraction" is correct but underspecified. In the code, $f_j$ is computed as `f_j_counts / total_assignments` where `total_assignments = B * N_tokens * K`. Since each token selects $K{=}2$ experts, the total assignment count is $2{\times}$ the token count, meaning $f_j$ represents the fraction of expert-slot assignments (not tokens) assigned to expert $j$. This is a subtle but important distinction: $\sum_j f_j = 1$ because each of the $K$ slots per token contributes to the count. The paper's notation is consistent with this, but the prose could be clearer.

ACTUAL FACT:
`src/loss.py:123-129` — `total_assignments = B * N_tokens * K`, `f_j = f_j_counts.float() / total_assignments`. Since each token has $K$ slots, $\sum_j f_j = K/N_{\text{tokens}} \times N_{\text{tokens}} = K$ ... actually, let me recheck. `flat_indices = out.topk_indices.view(-1)` has `B * N_tokens * K` elements. `f_j_counts = torch.bincount(flat_indices, minlength=E)` sums to `B * N_tokens * K`. So `f_j = f_j_counts / (B * N_tokens * K)` and $\sum_j f_j = 1$. This is correct. The paper's description is adequate.

WHAT PAPER SHOULD SAY:
No change strictly needed. For additional clarity: "$f_j$ is the fraction of all routing slots (over all tokens, all $K$ selections) assigned to expert $j$, computed from hard assignments and detached."

SOURCE:
`src/loss.py:123-129`

SEVERITY:
MINOR (no actual error — notation could be slightly more precise)

---

## Issue 16

PAPER CURRENTLY SAYS:
Section I (Introduction): "Existing MoE-SOD methods route at the modality, scale, or task level for multi-modal inputs---not at the spatial token level for single-RGB SOD."

PROBLEM:
This claim is stated as fact but has not been exhaustively verified. The paper cites V-MoE (classification, not SOD) and mentions MoE-SOD methods implicitly, but does not cite a specific survey or systematic review that confirms no MoE-SOD method performs per-token routing for single-RGB inputs. The claim may be true, but the evidence base is the literature notes in this repository, not a comprehensive survey.

ACTUAL FACT:
`docs/research/LITERATURE_NOTES.md` — documents V-MoE (classification), Soft MoE (classification), M³ViT (multi-task), and Mod-Squad (multi-task). None are specifically MoE-SOD for single-RGB. The claim is plausible but not rigorously established.

WHAT PAPER SHOULD SAY:
"To our knowledge, no existing MoE-SOD method performs per-token routing for single-RGB inputs." Adding "to our knowledge" properly scopes the claim.

SOURCE:
`docs/research/LITERATURE_NOTES.md`

SEVERITY:
MINOR — Standard academic hedging is missing.

---

## Issue 17

PAPER CURRENTLY SAYS:
Table 1 caption: "Global performance on WXSOD test splits."

PROBLEM:
The table does not report $F_\beta^{\text{mean}}$ or $E_\phi^{\text{mean}}$ or $E_\phi^{\max}$, only $F_\beta^{\max}$ and $E_\phi^{\text{adp}}$. While this is a valid editorial choice, the paper's metric list in Section IV-A says "Metrics: MAE, $S_\phi$, $E_\phi^{\text{adp}}$, $F_\beta^{\max}$" which is consistent. No error here.

ACTUAL FACT:
Table 1 columns: Set, MAE, S_phi, F_max, E_adp. Consistent with the metric description.

WHAT PAPER SHOULD SAY:
No change needed.

SOURCE:
`paper/main.tex:446-453`

SEVERITY:
N/A (no error)

---

## Issue 18

PAPER CURRENTLY SAYS:
Section IV-A (Setup): "Metrics: MAE, $S_\phi$~\cite{struct-measure}, $E_\phi^{\text{adp}}$~\cite{enhanced-measure}, $F_\beta^{\max}$~\cite{f-measure}."

PROBLEM:
The paper lists only 4 metrics but the evaluation actually computes 8 core metrics (MAE, S_measure, E_adaptive, E_mean, E_max, F_adaptive, F_mean, F_max) plus 2 boundary metrics. The paper chose to report only 4 in the main tables, which is fine, but the metric list in the setup section implies these are the only metrics computed. This is not an error but is incomplete.

ACTUAL FACT:
`src/metrics.py:44-59` — computes MAE, S_measure, E_adaptive, E_mean, E_max, F_adaptive, F_mean, F_max. `src/metrics.py:61-114` — computes boundary_MAE and boundary_F1.

WHAT PAPER SHOULD SAY:
"Metrics: MAE, $S_\phi$~\cite{struct-measure}, $E_\phi^{\text{adp}}$~\cite{enhanced-measure}, $F_\beta^{\max}$~\cite{f-measure}. Additional metrics ($E_\phi^{\text{mean}}$, $E_\phi^{\max}$, $F_\phi^{\text{adp}}$, $F_\phi^{\text{mean}}$, boundary MAE, boundary F1) are computed but reported in the supplementary material." (Or similar, if supplementary exists.)

SOURCE:
`src/metrics.py`

SEVERITY:
MINOR — Incomplete metric list, acceptable if supplementary exists.

---

## Issue 19

PAPER CURRENTLY SAYS:
Title: "Spatially Dynamic Mixture-of-Experts for Adverse-Weather Salient Object Detection"

PROBLEM:
The title implies the model has been demonstrated to work for adverse-weather SOD, which is supported by the experimental results. However, the word "Dynamic" in "Spatially Dynamic" could be misinterpreted as implying the routing changes over time (temporal dynamics) rather than varying across spatial locations. In context, "Spatially Dynamic" means "spatially varying" (different tokens routed differently), which is the intended meaning.

ACTUAL FACT:
The model routes different spatial tokens to different experts — this is "spatially varying" or "spatially dynamic" routing. The title is accurate but could be clearer.

WHAT PAPER SHOULD SAY:
The title is acceptable. "Spatially Dynamic" is a reasonable description of per-token spatial routing. No change needed.

SOURCE:
N/A

SEVERITY:
MINOR (stylistic, not a factual error)

---

## Issue 20

PAPER CURRENTLY SAYS:
Section III-B (Router), after Eq. 2: "where $|$ denotes channel-wise concatenation, yielding a $2C$-dimensional vector that encodes both the token's content and its spatial neighborhood."

PROBLEM:
The router input is described as encoding "the token's content and its spatial neighborhood." This is accurate — the concatenation of the original token $\bx_i$ and its DWConv3x3-smoothed version captures both the token itself and its local context. However, the DWConv is applied to the full feature map (not per-token), so each "local context" vector is the result of a depthwise convolution centered on that token's spatial location. The description is correct.

ACTUAL FACT:
`src/moe_layer.py:97-99` — `local_feat = self.router_dwconv(x)` applies DWConv3x3 to the full `[B, C, H, W]` feature map, then flattens to tokens. This produces a local context vector for each spatial position. The concatenation with the original token is correct.

WHAT PAPER SHOULD SAY:
No change needed.

SOURCE:
`src/moe_layer.py:97-99`

SEVERITY:
N/A (no error)

---

## Issue 21

PAPER CURRENTLY SAYS:
Section III-E (Sparse Expert Aggregation): "Each expert is a token-wise residual MLP: $\E_k(\bx) = \bx + W_k^{(2)} \cdot \text{GELU}(W_k^{(1)} \cdot \text{LN}(\bx))$, with $W_k^{(1)} \in \mathbb{R}^{4C \times C}$, $W_k^{(2)} \in \mathbb{R}^{C \times 4C}$ ($4{\times}$ expansion, $C{=}256$ hidden)."

PROBLEM:
The paper says "$C{=}256$ hidden" which could be read as "the hidden dimension is 256." In context, $C{=}256$ refers to the input/output dimension of the expert, not the hidden dimension. The hidden dimension is $4C = 1024$. The parenthetical is slightly ambiguous.

ACTUAL FACT:
`src/moe_layer.py:20-25` — `nn.Linear(dim, dim * expansion)` where `dim=256` and `expansion=4`, producing a hidden dimension of 1024. The input and output dimensions are 256.

WHAT PAPER SHOULD SAY:
"$W_k^{(1)} \in \mathbb{R}^{4C \times C}$, $W_k^{(2)} \in \mathbb{R}^{C \times 4C}$ ($4{\times}$ expansion from $C{=}256$ to $4C{=}1024$)."

SOURCE:
`src/moe_layer.py:20-25`

SEVERITY:
MINOR — Ambiguous phrasing.

---

## Issue 22

PAPER CURRENTLY SAYS:
Section III-C (Noisy Top-K Routing), Eq. 4: $\sigma_j(\bx_i) = \text{softplus}(w_j^\top \bx_i + b_j)$ is described as "a learned per-expert, per-token noise scale."

PROBLEM:
The description "per-expert, per-token" is correct — `self.noise_linear = nn.Linear(dim, num_experts)` produces one noise scale per expert per token. However, the paper's Eq. 4 notation $\sigma_j(\bx_i)$ uses $w_j$ (a vector) suggesting a per-expert weight vector applied to the token, which matches the linear layer. The description is accurate.

ACTUAL FACT:
`src/moe_layer.py:60` — `self.noise_linear = nn.Linear(dim, num_experts)` — for each token $\bx_i \in \mathbb{R}^C$, produces a vector $\in \mathbb{R}^E$ of noise scales, one per expert. `src/moe_layer.py:105` — `noise_std = self.router_noise_scale * F.softplus(self.noise_linear(x_tokens))`.

WHAT PAPER SHOULD SAY:
No change needed. The equation and description are accurate.

SOURCE:
`src/moe_layer.py:60,105`

SEVERITY:
N/A (no error)

---

## Issue 23

PAPER CURRENTLY SAYS:
Section IV-B (Main Results): The paper presents only one model's results with no comparison to other methods or baselines.

PROBLEM:
This is correctly noted in the Discussion and Limitations sections. The paper does not claim SOTA or competitive performance. However, the presentation of a single model's results in a standalone table could give the impression that these are strong results without context. The paper is honest about this limitation.

ACTUAL FACT:
No SOTA comparison exists. `RESEARCH_TRUTH.md` explicitly lists "Our method outperforms SOTA" as a forbidden claim.

WHAT PAPER SHOULD SAY:
The paper is appropriately humble. No change needed.

SOURCE:
`RESEARCH_TRUTH.md`

SEVERITY:
N/A (correctly handled)

---

## Issue 24

PAPER CURRENTLY SAYS:
Section IV-F (Discussion): "the forced-expert analysis suggests that expert specialization may be limited, and the benefit of multi-expert routing over a single larger expert remains unclear."

PROBLEM:
The paper correctly identifies this limitation. However, it could be more precise about what the forced-expert experiment actually tested. The experiment forced only scale 1/4 to expert 0 while leaving scales 1/8 and 1/16 routed normally. The marginal degradation (+0.0002 MAE) could mean either (a) expert specialization is limited, or (b) the remaining routing at scales 1/8 and 1/16 compensated for the forced scale. The paper acknowledges this ambiguity ("Since only 1 of 3 scales was affected, the full effect of disabling all routing remains unknown") which is appropriate.

ACTUAL FACT:
`results/legacy/legacy_8expert/evaluation_results/force_expert_ablation.json` — Only scale 1/4 was forced. Scales 1/8 and 1/16 remained normally routed. The paper's caveat is accurate.

WHAT PAPER SHOULD SAY:
The paper's discussion is appropriately cautious. No change needed.

SOURCE:
`results/legacy/legacy_8expert/evaluation_results/force_expert_ablation.json`

SEVERITY:
N/A (correctly handled)

---

## Issue 25

PAPER CURRENTLY SAYS:
The paper includes the routing entropy table (Table 3) showing entropy values for each scale on synthetic and real test sets.

PROBLEM:
The paper reports the entropy values to 4 decimal places, which implies a level of precision that may not be meaningful given these are means over thousands of tokens and hundreds of images. Additionally, the paper does not report standard deviations or confidence intervals for these values.

ACTUAL FACT:
`results/legacy/legacy_8expert/evaluation_results/entropy_comparison.json` — values are reported to 16 decimal places. The paper rounds to 4 decimal places. No standard deviations are computed or reported.

WHAT PAPER SHOULD SAY:
Consider reporting entropy values as 0.68–0.69 (range) rather than precise 4-decimal values, or compute and report standard deviations. This is a presentation choice, not a factual error.

SOURCE:
`results/legacy/legacy_8expert/evaluation_results/entropy_comparison.json`

SEVERITY:
MINOR — Precision implies more certainty than warranted.

---

## Issue 26

PAPER CURRENTLY SAYS:
The WXSOD citation (references.bib): `year={2026}` for the WXSOD paper, and `journal={Pattern Recognition}, volume={170}, pages={114003}`.

PROBLEM:
The WXSOD paper is cited as published in Pattern Recognition, volume 170, 2026. This should be verified against the actual publication. The current year is 2026, so a 2026 publication is plausible. However, the DOI `10.1016/j.patcog.2026.114003` should be checked.

ACTUAL FACT:
`paper/references.bib:3-11` — WXSOD citation with year=2026, Pattern Recognition. Cannot verify against the actual journal without web access.

WHAT PAPER SHOULD SAY:
Verify the WXSOD citation against the actual publication record before submission.

SOURCE:
`paper/references.bib`

SEVERITY:
MINOR — Citation should be verified but is plausible.

---

## Issue 27

PAPER CURRENTLY SAYS:
The NIFM citation (references.bib): `year={2025}` and `journal={arXiv preprint arXiv:2512.10592}`.

PROBLEM:
The NIFM paper is cited as a 2025 arXiv preprint. The arXiv ID `2512.10592` implies submission in December 2025. This should be verified.

ACTUAL FACT:
`paper/references.bib:13-18` — NIFM citation. Cannot verify without web access.

WHAT PAPER SHOULD SAY:
Verify the NIFM citation. If it has been published in a venue by submission time, update the citation.

SOURCE:
`paper/references.bib`

SEVERITY:
MINOR — Citation should be verified.

---

## Issue 28

PAPER CURRENTLY SAYS:
Section II-A (Adverse-Weather SOD): "WXSOD~\cite{wxsod2025} provides 14,945 images with per-image weather labels and synthetic/real-world test splits, establishing the first benchmark for weather-robust SOD."

PROBLEM:
The claim that WXSOD is "the first benchmark for weather-robust SOD" is stated without qualification. Other benchmarks may exist (e.g., for weather-robust segmentation more broadly). The WXSOD paper itself may make this claim, but this paper should verify or soften it.

ACTUAL FACT:
`docs/research/DATASETS.md` — WXSOD is described as providing weather labels and synthetic/real test splits. The "first benchmark" claim comes from the blueprint, not from a comprehensive survey of weather-robust SOD benchmarks.

WHAT PAPER SHOULD SAY:
"WXSOD~\cite{wxsod2025} provides 14,945 images with per-image weather labels and synthetic/real-world test splits for weather-robust SOD." Remove "establishing the first benchmark" unless verified against the WXSOD paper's own claims.

SOURCE:
`docs/research/DATASETS.md`, `spatial-moe-adverse-weather-sod-blueprint.md`

SEVERITY:
MINOR — "First benchmark" claim should be verified or softened.

---

# Most Important Corrections

1. **Issue 6 (MAJOR):** The paper suggests "effective feature transfer across domains" from the observation that real-world MAE is lower than synthetic MAE. This is a confounded comparison (synthetic test set has harder compound weather). The paper should remove the causal interpretation or reframe it as a composition effect.

2. **Issue 2 (MAJOR):** The paper claims routing entropy injection is a contribution but has no ablation demonstrating it helps. The Related Work and Contributions sections should be tempered to say this is a design choice whose contribution has not yet been isolated.

3. **Issue 5 (MAJOR):** Backbone freezing during epoch 0 is omitted from the training setup description. This is a significant reproducibility detail.

4. **Issue 12 (MAJOR):** The claim about WM-MoE not requiring weather labels at test time has not been verified against the primary source. This should be verified or qualified.

5. **Issue 14 (MINOR):** The forced-expert MAE percentage is +0.89%, not +1.2%. Small but should be corrected for accuracy.

6. **Issue 3 (MINOR):** The entropy normalization formula says $\log K$ but the code hardcodes $\log 2$. Should say $\log 2$ or the code should be made general.

7. **Issue 9 (MINOR):** "Five weather categories" in the abstract conflates the two test sets. Synthetic has 9 categories.

8. **Issue 16 (MINOR):** The novelty claim "Existing MoE-SOD methods route at the modality, scale, or task level" should be hedged with "to our knowledge."

9. **Issue 25 (MINOR):** Entropy values reported to 4 decimal places without standard deviations implies false precision.

10. **Issue 11 (MINOR):** The MAC count should specify "per 384×384 image."

---

# Final Accurate Paper Position

Based on the implementation and experimental evidence that actually exists, this paper can honestly claim the following: We present SpatialMoE-SOD, an architecture that routes individual spatial tokens to separate expert networks at three independent pyramid scales (1/4, 1/8, 1/16) via depthwise-convolutional routers with Shazeer-style noisy top-2 gating and true sparse dispatch, without any weather-specific supervision at train or test time. Per-token routing entropy is injected as a feature channel into the cross-attention decoder via learnable additive fusion. On the WXSOD benchmark, a single trained model (66.27M parameters, 278.2G MACs) achieves MAE 0.0192 on 1,500 synthetic test images and 0.0168 on 554 real-world test images, with weather-wise MAE ranging from 0.0094 (snow) to 0.0243 (low-light) on real-world data. The model was trained once without replicate runs, and no controlled comparison against existing methods (e.g., NIFM) has been performed. A forced-expert ablation at scale 1/4 shows only marginal degradation (+0.0002 MAE), suggesting that the 8-expert pool may be over-parameterized or that experts have learned similar representations — but this test covers only one of three scales and cannot establish or refute the utility of routing. No ablation has been run to isolate the contribution of any individual design choice (entropy fusion, multi-scale routing, MoE vs. shared MLP, number of experts, top-k value). The architectural contribution — token-level spatial MoE routing for single-RGB SOD without weather labels, with independent routers at three pyramid scales and entropy-informed decoding — is novel in its combination, though its practical benefit over simpler alternatives has not been demonstrated.
