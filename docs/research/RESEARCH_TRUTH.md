# RESEARCH_TRUTH.md — authoritative scientific claims

The single source of truth for what may be claimed in the paper. Before drafting or
revising any section, read this. Never add a claim that is not supported by the
implementation, by a result file, or by a verified literature source.

Cite implementations by module and function, not by line number: line cites rot within
days, and half the stale statements in this file's predecessor were stale line cites.
Experimental numbers live in `docs/research/RESULTS.md`, which is generated from the
result files and names its source for every value.

---

## 1. Implementation-verified claims

### 1.1 Architecture (`src/model.py`, `src/backbone.py`, `src/moe_layer.py`, `src/decoder/`)

| Claim | Evidence |
|---|---|
| Backbone is PVTv2-B4 via timm, `features_only=True`, `out_indices=(0,1,2)` | `MultiScaleBackbone.__init__` |
| Each scale is 1x1-convolved to a common width (256) | `MultiScaleBackbone.proj_4/8/16` |
| Three independent MoE layers, one per scale | `SpatialMoESODNet.__init__` (`self.moe_4/8/16`) |
| Router input is the token's own features concatenated with a local depthwise-conv context (`cat([x_tokens, dwconv(x)])`, `2C` wide) | `SpatialMoELayer.forward` |
| Routing is noisy top-k with a learned softplus noise scale | `RouterNoise`, `add_noise` |
| Dispatch is truly sparse: a loop over experts with mask selection | `SpatialMoELayer.forward` |
| Every expert is called every forward pass, even on an empty slice | `SpatialMoELayer.forward` (required for DDP gradient sync) |
| Expert is a token-wise MLP: LN -> 4C -> C with a residual | `TokenWiseMLP` |
| Decoder is a package: `src/decoder/decoder.py`, `src/decoder/blocks.py` | `SpatialMoEDecoder`, `EntropyFusionBlock` |
| Decoder fuses routing entropy as a per-scale channel | `EntropyFusionBlock` |
| Routing entropy is computed over the full E-way gate distribution (`-sum(g log g)` over all experts), so its range is `[0, ln E]` | `SpatialMoELayer.forward` |
| Heads: saliency, boundary, and three deep-supervision aux heads | `SpatialMoEDecoder` |
| Parameter count, E8 k=2 | **69,213,120** for the reported recipe (window 7, deep supervision **on** — every `experiments/v_e8_*.json` sets `deep_supervision: true`, and the run artifacts agree); **69,212,349** with deep supervision off; **69,212,797** / **69,213,568** at window 8 (off / on). Enumerated directly over `parameters()` |
| Active loss terms, reported (legacy) recipe | BCE 1.0, IoU 1.0, SSIM 1.0, boundary 1.0, per-scale load-balance 0.01, importance 0.01, aux-boundary 0.5, deep supervision 0.4; Z-loss and router-confidence 0.0. The 2026-09 arm configs differ on the router terms: importance 0.05, Z-loss 0.001, per-scale load-balance 0.08/0.15/0.10 | `experiments/registry.csv` (loss string `BCE=1.0,IoU=1.0,SSIM=1.0,BND=1.0`), `results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_full_moe/final_config.json`, `experiments/v_e8_repro_best.json` |

**`gate_mode`** selects how top-k gates are formed and is genuinely consumed:
`"renormalized"` (softmax over the selected top-k values) or `"dense"` (softmax over all
experts, gathered at the top-k indices — which gives every expert gradient).

**The preset is not the recipe.** `experiments/baseline_v1.json` (`ssim_weight: 0`,
`boundary_weight: 0`, `deep_supervision: false`, `epochs: 50`, `warmup_ratio: 0.01`) is a
template. The reported metrics (0.0192 / 0.0168) come from the **legacy-era run** of the E8
k=2 S32 L3 recipe (old code), whose config is `results/legacy/preflight/final_config.json`:
`ssim_weight: 1.0`, `boundary_weight: 1.0`, `aux_boundary_weight: 0.5`,
`deep_supervision_weight: 0.4`, `model.deep_supervision: true`, `load_balance_weight: 0.01`,
`importance_weight: 0.01`, `z_loss_weight: 0.0`, `warmup_ratio: 0.01`, `epochs: 50`,
`batch_per_gpu: 4` × `grad_accum_steps: 4` × 2 GPUs. The 2026-09 arm configs
(`experiments/v_e8_repro_best.json` and the tracked
`results/EXP_B4_E8_K2_S32_R1_L3_M_SPARSE__ablation_full_moe/final_config.json`) reproduce the
same recipe under the current code and differ on the router/optimisation terms:
`importance_weight: 0.05`, `z_loss_weight: 0.001`,
`load_balance_weights: [0.08, 0.15, 0.10]`, `warmup_ratio: 0.04`, `epochs: 8` (14 for the
flagship run). A loss table, warmup or epoch count quoted from the preset will not describe a
trained run — this is the single most common way these docs have gone stale.

**`moe_type`** is consumed and selects the ablation arm: `"sparse"` (routed experts,
the model under study), `"dense"` (one shared expert on every token), `"none"` (no MoE,
expert parameters removed), `"sparse_fine"` (routed experts at the 1/4 scale only, the
coarser two pass through — a routing-scope ablation that is not capacity-matched).

### 1.2 Training (`src/training/`, `src/optimization.py`, `src/config.py`)

| Claim | Evidence |
|---|---|
| `ExperimentConfig` (`src/config.py`) is the single source of truth for hyperparameters | config dataclasses |
| The built model is asserted to match its config (experts, top-k, gate mode, MoE type) | `assert_model_matches_config`, called from `build_model` |
| The experiment ID is a pure function of the config and is derived on every rank | `generate_experiment_id`, `resolve_workspace` |
| `find_unused_parameters=True` is deliberate: which modules participate depends on config | `build_model` |
| AMP fp16 with a `GradScaler`; the scaler's scale is logged on the progress line | `OptimizationEngine`, `_optimizer_health` |
| Scheduler is linear warmup then cosine decay to 1% of peak | `WarmupCosineScheduler` |
| Optimizer is AdamW with decay / no-decay parameter groups | `get_parameter_groups` |
| Gradient clipping is applied before the optimizer step | `OptimizationEngine.step` |
| Checkpoint selection is by validation MAE | training loop |
| Checkpoints are written every `train.checkpoint_every_n_steps` (200) | `save_latest` |
| `--preflight` redirects outputs to a separate root, disables HF pushes, and allows one rank | `resolve_workspace` |
| Freezing the backbone is done by nulling its gradients after backward | training loop; `freeze_backbone` is intentionally a no-op |

### 1.3 Dataset (`src/dataset.py`)

| Claim | Evidence |
|---|---|
| Splits: `train_sys`, `test_sys`, `test_real` | `WXSODDataset` |
| Scene-aware `GroupShuffleSplit` for train/val, with a disjointness assertion | `get_dataloaders` |
| Weather label parsed from the filename | `get_weather_type` |
| Train augmentation is `HorizontalFlip(0.5)` + `ColorJitter(0.5)` + normalize; other splits normalize only | `build_transforms` |
| Geometry is aspect-preserving resize then **centre** reflect-pad to 384x384 | `_aspect_preserving_resize_pad` |
| Augmentation is deterministically seeded per sample and epoch | `__getitem__` |
| Boundary targets are morphological dilate-minus-erode with a 5x5 ellipse | `src/boundary.py` |
| Split sizes: train_sys 12,891 / test_sys 1,500 / test_real 554 | file listing; confirmed in training logs |

### 1.4 Evaluation (`src/evaluate.py`, `src/metrics.py`)

| Claim | Evidence |
|---|---|
| Metrics: MAE, S-measure, E-adaptive/mean/max, F-adaptive/mean/max | `SODMetrics` (py_sod_metrics) |
| Boundary metrics: boundary MAE and boundary F1 | `BoundaryMetrics` |
| Prediction maps are reverse-geometried to original resolution before scoring | `reverse_geometry` |
| Weather-wise breakdown comes from filename parsing | `evaluate` |
| Flip-average TTA exists (`--tta hflip`) and is the default for new runs | `predict_probs` |
| Different model architectures must be rebuilt exactly from the checkpoint config | `evaluate` |

---

## 2. Experimentally-verified claims

Full tables, with per-value source files, are in **`docs/research/RESULTS.md`**. Summary:

| Model | test_real MAE / S | test_sys MAE / S |
|---|---|---|
| E8 legacy (old code, renormalized, eff batch 32) | 0.0168 / 0.9151 | 0.0192 / 0.9139 |
| E4 K2 (dense gate, eff batch 40) | 0.0197 / 0.9043 | 0.0215 / 0.9040 |
| E2 K1 (dense gate, eff batch 40) | 0.0199 / 0.9034 | 0.0215 / 0.9041 |

**These rows are not directly comparable.** The E8 row differs from the other two in five
respects at once (gate mode, router noise, load-balance weights, importance weight, and
effective batch size), so a gap between rows cannot be attributed to expert count. All
values are single-pass; none is TTA-boosted.

### 2.1 Routing behaviour — measured

| Observation | Value | Source |
|---|---|---|
| Per-token routing entropy sits at its ceiling | normalized entropy 0.997-1.000 of `ln E` at every scale (E8: 2.074-2.079 nats of `ln 8 = 2.0794`) | diagnostics routing stats |
| Router logits are nearly tied | mean logit std 0.001-0.10; top1-top2 margin 0.002-0.07 across runs and scales | same |
| Dead or near-dead experts persist under the renormalized gate | `DEAD_EXPERT` warnings appear in the E8 arms (e.g. `[2]`, `[5]`, `[2, 5]`); in the legacy `moe_8` stats 2 of 8 experts sit below 2% usage | same |
| The pattern reproduces across training runs | same dominant-expert shape as the legacy checkpoint | same, plus legacy stats |
| Experts are not weather-specialised | per-expert weather divergence from the dataset prior is tiny (max KL 0.04 at scale 8, JS <= 0.05) | same, `weather_enrichment` |
| The largest weather skew sits in the least-used experts | e.g. a 2.3%-usage expert is snow-enriched (26% vs 12% prior) | same |
| Learned routing is neutral to slightly positive versus random routing | E8: 0.0168 vs 0.0168 MAE; E2/E4: 0.0204 vs 0.0199 / 0.0197 | `proxy_ablation_results.json` |

### 2.2 Runs that exist

`results/` contains exactly: the legacy 8-expert checkpoint evaluations, the two
GATEDENSE runs (E2 K1 and E4 K2, effective batch 40), the two E8 ablation runs
(`ablation_full_moe`, `ablation_moe16_dense`), and the 2026-09 experiment set (E8 k=2
renormalized and dense, two seed repeats, and E4 at the E8 recipe).

---

## 3. Forbidden claims

Not supported by any evidence in this repository. Do not write them.

- **Any *measured* comparison to prior methods.** No external method has been re-implemented
  or re-run. A comparison against the WXSOD benchmark's published tables is permitted, but
  only labelled "as reported" with the protocol difference disclosed.
- **"The MoE helps."** The mixture-vs-shared-MLP control (`..._M_NONE_E4_NONECTRL`) and the
  dense-gate control (`..._M_DENSE_E4_DENSECTRL`) were training at the time of writing;
  results pending. Do not assert either outcome until their metrics land.
- **"Experts specialise by weather."** Measured divergence from the weather prior is
  negligible; the residual skew is concentrated in the least-used experts.
- **"Routing causes the improvement"** or any causal claim about a design choice — no
  matched control exists for any of them.
- **Any backbone comparison.** `config.model.backbone` does not reach the model;
  `src/model.py` hard-codes PVTv2-B4, so configs that vary it train identical
  architectures. The ablation arms that "vary the backbone" are currently no-ops.
- **Any efficiency or FLOPs claim against other methods.** MACs were measured once for
  the legacy model only.
- **Anything about datasets other than WXSOD**, or about clean SOD benchmarks.
- **Statistical claims from a single seed.** Except the seed repeats now running, every
  number is one run, and there is no historical variance estimate.
- **Any claim resting on a config field that is not consumed by the code**
  (`router_variant` currently affects only the experiment ID).

---

## 4. Known implementation gaps

| Gap | Consequence |
|---|---|
| `config.model.backbone` is never passed to the model | backbone ablations are meaningless as configured |
| `router_variant` is consumed only by the experiment-ID builder | the three router variants train the same router |
| The mixture-vs-dense control arm has never been run | the central claim has no support |
| Single seed per configuration (except the repeats) | no variance estimate |
| `src/ablations.py` `# A: PVT-B2` arm would train B4 | a fabricated backbone comparison if ever run |

---

### RESOLVED: parameter count

The two legacy cost files — `results/legacy/legacy_8expert/eval_results/compute_cost.json` and
its ignored copy under `evaluation_results/` — both report **68.9M** (`params_M`), with
278.2G MACs. Direct enumeration over `parameters()` gives **69.21M** for every E8 k=2
variant, and **69,213,120** for the reported recipe, which has deep supervision on: every
`experiments/v_e8_*.json` sets `deep_supervision: true`, and the training log's 69,213,120
matches it. The other enumerated values are 69,212,349 (window 7, deep supervision off),
69,212,797 (window 8, off) and 69,213,568 (window 8, on).

The **66.27M** figure that circulated in earlier manuscript revisions and in
`PAPER_CORRECTIONS.md` is not present in any file under `results/` and must not be used.

**The parameter count to report is 69.21M**, with **277.9G MACs** at 384x384 (measured with
`thop` on the current model; the legacy file's 278.2G is a separate, older measurement and
the two must not be quoted as a matched pair). The training log's 69,213,120 was correct
all along. Component split: backbone 51.18M, decoder 5.19M, each MoE layer 4.28M.

### RESOLVED: routing entropy

`SpatialMoELayer.forward` computes per-token entropy over the **full softmax over all E
router logits** (`full_gates = softmax(noisy_logits)`; `-sum(full_gates * log(full_gates))`),
so its range is `[0, ln E]`, not `[0, ln K]`. `routing_summary` normalises by `ln E` and
`EntropyFusionBlock` divides by `math.log(8.0)`, which is exactly the ceiling of the
quantity the 8-expert model produces — the earlier note calling that a mismatch was written
against a top-k-entropy implementation that no longer exists.

| Quantity | E8 measurement | Meaning |
|---|---|---|
| `mean_entropy` | 2.074-2.079 nats | mean entropy of the full 8-way gate distribution |
| `max_entropy` | 2.0794 nats | `ln 8`, the ceiling |
| `mean_normalized_entropy` | 0.997-1.000 | that value divided by `ln 8` |

The mean therefore sits at **~99.7-100% of the ceiling** at every scale on both test splits:
the router's full distribution is nearly uniform, so the per-token selection carries almost
no information. Because the normalised value is ~1.0, the decoder's entropy channel is
nearly constant, which is an adequate explanation for why entropy fusion is inert
independent of the earlier `no_entropy_fusion` comparison. Sources:
`results/**/routing_entropy/entropy_comparison.json` and
`results/**/diagnostics/*/routing_stats_contentonly.json`.

**Stale measurement warning.** `results/legacy/legacy_8expert/eval_results/entropy_comparison.json`
(and the ignored copy under `evaluation_results/`) reports 0.692-0.693 nats for the
eight-expert legacy checkpoint. That file predates the fix to the entropy computation and
holds the *top-2 gate* entropy, capped at `ln 2 = 0.6931`; it must not be compared against
`ln 8`, and its values must not be mixed with the current-code numbers in the table above.
The current-code measurement for the same checkpoint is
`results/legacy/legacy_8expert/routing_entropy/entropy_comparison.json` (2.075-2.079 nats).

## 5. Maintenance

Update this file whenever a claim becomes supported or stops being supported, and
regenerate `docs/research/RESULTS.md` after new evaluations. Provenance for every number
must remain a file path that a reader can open.
