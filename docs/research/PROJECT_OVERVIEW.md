# Project Overview

## Title
Spatially Dynamic Mixture-of-Experts for Adverse-Weather Salient Object Detection

## Problem statement
Salient object detection degrades sharply under adverse weather (fog, rain, snow,
low light). Existing responses either condition on a weather label or embed a
weather-specific branch at the image level. This project routes individual **spatial
tokens** to specialised experts with **no weather label at training or test time**.

## Claimed novelty
1. **Label-free spatial routing** — the router sees no weather information.
2. **Independent routers at three pyramid scales** (1/4, 1/8, 1/16).
3. **Routing applied to a segmentation-topology task** rather than pixel-restoration.

The measured behaviour of the router is documented in `RESEARCH_TRUTH.md` §2.1: it is
close to uniform and shows no weather specialisation. The architecture claim above is
about design intent; the paper must not claim learned specialisation the data does not
show.

## Study status

The ablation study is complete. The mixture-of-experts, its routing and its entropy channel
are each removable, randomisable or short-circuitable with no measurable effect: the no-MoE,
one-scale, dense-expert and no-load-balance variants all sit within the seed-noise band of the
reference, and all but the dense-expert variant pass a TOST equivalence test at a 0.001 MAE
margin. The performance comes from the backbone, decoder and training budget, not routing.
*Why* the router is inert is a ranked set of hypotheses, not a settled finding: **H1** selection
is noise-dominated at initialisation; **H2** the auxiliary and Z-loss terms do not reward
decisiveness; **H3** the experts are near-identical at initialisation, leaving the gate nothing
to separate. The measured gate is a **static selector**: flat full-softmax
entropy (0.997-1.000 of the ceiling) coexists with a strongly skewed ranking (at 1/8, three
experts take ~84% of hard assignments and two are dead). The decisive measurements — the
initial-vs-trained top-2 assignment agreement, the input-dependence of the assignment, and
oracle routing by weather class as the positive control — have not been run. See
`RESEARCH_TRUTH.md` §2.3 and `RESULTS.md`.

## System summary

| Component | Choice | Source |
|---|---|---|
| Backbone | PVTv2-B4 (timm, `features_only`, `out_indices=(0,1,2)`) | `src/backbone.py` |
| Working width | 256, one 1x1 projection per scale | `src/backbone.py` |
| Experts per scale | 8 (configurable) | `experiments/v_e8_repro_best.json` |
| Top-k | 2 (configurable) | same |
| Expert | token-wise MLP, LN -> 4C -> C with residual | `src/moe_layer.py` |
| Router | DWConv3x3 + MLP over the token and its local context, noisy top-k | `src/moe_layer.py` |
| Gate mode | `renormalized` or `dense` | `src/moe_layer.py` |
| Decoder | cross-attention fusion + refinement, entropy fused per scale | `src/decoder/` |
| Loss | BCE + IoU + SSIM + boundary + load-balance + importance + z + routing-confidence + deep supervision | `src/loss.py` |
| Model size | 69,213,120 parameters (E8 k=2, window 7, deep supervision on = the reported recipe); 69,212,349 with supervision off | enumerated over `parameters()` |
| Training | DDP 2 GPU, AMP fp16, AdamW, warmup+cosine | `src/training/` |
| Dataset | WXSOD — 12,891 train / 1,500 synthetic test / 554 real test | `src/dataset.py` |

## Repository map

| Path | Purpose |
|---|---|
| `src/` | Model, training, evaluation, diagnostics |
| `src/training/` | Training loop, setup, checkpointing, DDP helpers |
| `src/decoder/` | Decoder package (blocks + fusion) |
| `src/diagnostics/` | Routing statistics, reporting, aggregation |
| `src/hf_sync/` | Code and checkpoint sync with the Hugging Face Hub |
| `tests/` | Unit and integration tests |
| `experiments/` | Experiment configs; the config is the source of truth for a run |
| `results/` | Evaluations, routing statistics, legacy artifacts |
| `docs/research/` | This knowledge base; `RESULTS.md` holds every metric |
| `paper/` | Manuscript planning documents |

Evaluation results previously lived in a top-level `evaluation/` directory; they were
archived under `results/legacy/` and that directory no longer exists.
