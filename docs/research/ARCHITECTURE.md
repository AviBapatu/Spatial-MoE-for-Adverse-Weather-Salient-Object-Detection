# Architecture

> **Note on citations.** Line-number citations below predate the current source
> layout (the decoder is now the `src/decoder/` package, the training loop lives under
> `src/training/`, the evaluation artifacts are under `results/`). Treat module and
> function names as authoritative and `RESEARCH_TRUTH.md` as the verified reference;
> numbers live in `RESULTS.md`, which is generated from the result files.

Verified against the source in `src/`. Where a claim comes from reading code it says which
file; where it is inference, it says so. Anything the code contradicts is called out in
**Findings** at the end — those are the parts a paper must not repeat wrongly.

## End-to-end pipeline

```
Input [B, 3, 384, 384]
  -> MultiScaleBackbone (PVTv2-B4 via timm)          src/backbone.py
       crops out_indices=(0,1,2) -> 1/4, 1/8, 1/16
       three 1x1 convs project each scale to d=256
  -> one SpatialMoELayer per scale                   src/moe_layer.py
       router: depthwise 3x3 conv + MLP, noisy top-k (k=2)
       experts: E x TokenWiseMLP (LayerNorm -> 4C -> C, residual)
       true sparse dispatch: one call per expert over its selected tokens
  -> SpatialMoEDecoder                               src/decoder/decoder.py
       per scale: EntropyFusionBlock(features, entropy channel)
       top-down: global cross-attention 1/16 -> 1/8
                 windowed cross-attention 1/8 -> 1/4
       refinement blocks, bilinear upsample to 384x384
  -> saliency head (1x1 conv) -> [B, 1, 384, 384]
  -> boundary head (1x1 conv) -> [B, 1, 384, 384]
```

## Backbone — `src/backbone.py`

`MultiScaleBackbone` wraps a timm model with `features_only=True, out_indices=(0, 1, 2)` and
projects each scale to a common width with a 1x1 convolution. The projection widths are read
from the backbone itself (`feature_info.channels()`), so a different backbone's channel counts
adapt automatically — only the pyramid strides have to match.

| Scale | Shape | Tokens |
|---|---|---|
| 1/4 | `[B, 256, 96, 96]` | 9,216 |
| 1/8 | `[B, 256, 48, 48]` | 2,304 |
| 1/16 | `[B, 256, 24, 24]` | 576 |

**Finding (Backbone name is not wired).** `MultiScaleBackbone` accepts `model_name`, but
`SpatialMoESODNet` constructs it as `MultiScaleBackbone(d=dim, pretrained=pretrained_backbone)`
— the name is never passed, so **the model is always `pvt_v2_b4`**. `config.model.backbone` is
read only by `src/experiment.py` for the experiment ID and the run record. Changing it in a
config changes the run's *name* and nothing else. The generated ablation matrix in
`src/ablations.py` varies exactly this field, so those arms would train identical
architectures. Do not report a backbone comparison until this is wired through.

## Spatial MoE layer — `src/moe_layer.py`

### Router

Spatial context is gathered with a depthwise 3x3 convolution; the token embedding and the
local context are concatenated and passed through a small MLP to produce one logit per expert.
During training, Gaussian noise is added before the top-k selection, with a learned or
clamped standard deviation (`router_noise_enabled`, `router_noise_scale`,
`router_noise_min_std`).

### Gate mode — what the top-k weights are

`gate_mode` selects how the selected experts' weights are formed. Both modes are implemented
and both have been trained.

- `"renormalized"` (default): `softmax` over the **top-k logits only**, so the selected gates
  sum to 1. Gradient flows only to the experts that were selected.
- `"dense"`: `softmax` over **all E logits**, with the weights at the top-k indices taken from
  that distribution. The selected gates no longer sum to 1, and crucially every expert receives
  gradient, which is what allows an unselected expert to recover.

This distinction is the whole point of the dense arm: under a renormalized gate an expert that
stops being selected stops receiving gradient, so selection is rich-get-richer and experts die.

### Mixture type — `moe_type`

`SpatialMoESODNet` builds the per-scale module according to `moe_type`:

| `moe_type` | Module per scale | Meaning |
|---|---|---|
| `"sparse"` | `SpatialMoELayer` | routed top-k experts — the model under study |
| `"dense"` | `DenseMoE16Adapter` | one shared expert on every token: routing and sparsity removed, capacity-matched control |
| `"none"` | `PassthroughMoELayer` | features pass through; no expert parameters at all |

`moe_16_mode` additionally lets the 1/16 scale alone use the dense adapter, independently of
the other two scales.

### Experts

`TokenWiseMLP`: `LayerNorm(C) -> Linear(C, 4C) -> GELU -> Linear(4C, C)`, plus a residual.
Expert count per scale is `config.model.num_experts`; the 1/16 scale uses the same count
unless `moe_16_mode == "dense"`.

### Sparse dispatch

The forward pass loops over experts and processes only the tokens routed to each:

```
for i, expert in enumerate(self.experts):
    active = (flat_topk_indices == i).any(dim=-1)
    tokens = flat_x[active]
    out[active] += expert(tokens) * gates[active].unsqueeze(-1)
```

Every expert is called on every step, even with zero routed tokens — that is deliberate and
load-bearing for DDP (see `TRAINING.md`), not an inefficiency to remove.

### Counterfactual ablation

When a forced expert is requested, the router is bypassed: all tokens go to that expert, its
routing probability is 1.0 and entropy is 0.0. Used for the expert-knockout studies in
`RESULTS.md`.

## Decoder — `src/decoder/decoder.py`, `src/decoder/blocks.py`

### Entropy fusion

Each scale's routed features are fused with that scale's routing entropy. The entropy is
normalized before projection, and the normalization constant is hard-coded:

```python
entropy_norm = entropy / math.log(2.0 + 1e-8)   # normalize assuming K=2
y = self.proj_y(Y)
e = self.proj_entropy(entropy_norm)
return y + self.scale * e                        # learnable scale, init 0.1
```

### Top-down fusion

- **1/16 -> 1/8**: global cross-attention, with the 1/16 features upsampled to the 1/8 grid.
- **1/8 -> 1/4**: windowed cross-attention (window size `config.model.window_size`, default 7),
  with a relative position bias table and attention masking for non-divisible dimensions.
- **Global context** is pooled from the 1/8 features and added at the 1/4 scale.

### Refinement and heads

Three refinement blocks (two 3x3 convs with GroupNorm and a residual) take the 1/4 features to
full resolution, feeding the saliency and boundary heads. With
`config.model.deep_supervision` enabled, auxiliary heads predict at 1/16, 1/8 and 1/4.

## Outputs

| Output | Shape | Notes |
|---|---|---|
| Saliency logits | `[B, 1, 384, 384]` | sigmoid at inference |
| Boundary logits | `[B, 1, 384, 384]` | sigmoid at inference |
| Aux logits (16 / 8 / 4) | `[B, 1, 24, 24]` / `[B, 1, 48, 48]` / `[B, 1, 96, 96]` | deep supervision |
| Routing probs, per scale | `[B, E, H_s, W_s]` | full routing distribution |
| Entropy, per scale | `[B, 1, H_s, W_s]` | per-token routing entropy |

## Parameter count

For the 8-expert, k=2 configuration used by the reproduction runs, the trainer reports
**69,213,120 parameters, all trainable** (read from a training log, not computed here).

## Findings — read before writing architecture claims

1. **`config.model.backbone` does not reach the model.** `src/model.py` hard-codes
   `pvt_v2_b4`. Every number in `RESULTS.md` is therefore PVTv2-B4, and any config that names a
   different backbone mislabels its run ID. The fix is small: pass the name through
   `build_model` into `SpatialMoESODNet` into `MultiScaleBackbone`, store it, and extend
   `assert_model_matches_config` to compare it.

2. **The entropy channel's scale depends on a stale assumption.** The decoder normalizes
   entropy by `ln(2)`, which was correct when entropy was computed over the renormalized
   top-k gates (k=2, so entropy <= ln 2). Entropy is now computed over the full softmax, so the
   channel reaches `ln(E)/ln(2)` times the designed scale — **3x at E=8, 2x at E=4**. Measured
   per-token entropies are 1.386 (= ln 4) for the E4 runs and 2.075 (= ln 8) for an 8-expert
   model. The learnable per-scale coefficient can partly absorb this, but the reproduction run
   does not see the channel at the scale its recipe was tuned for. Any claim that the
   reproduction is exact must account for this.

3. **A model that does not match its config is now impossible to build silently.**
   `assert_model_matches_config` (`src/model.py`) compares the constructed MoE layers against
   the config's expert count, top-k, gate mode and mixture type, and `build_model` calls it.
   This exists because a `top_k: 1` run once trained a k=2 model. Evaluation and diagnostics
   have an equivalent guard: a shape mismatch fails the state-dict load.
