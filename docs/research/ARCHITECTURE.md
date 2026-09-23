# Architecture

Verified against `src/`. Module and function names are authoritative; no line numbers.

## Pipeline

```
input  [B, 3, 384, 384]
  MultiScaleBackbone (PVTv2-B4, timm, features_only, out_indices=(0,1,2))
      1x1 conv per scale -> working width 256
      res_4  [B, 256, 96, 96]     1/4 scale, 9216 tokens
      res_8  [B, 256, 48, 48]     1/8 scale, 2304 tokens
      res_16 [B, 256, 24, 24]     1/16 scale, 576 tokens
  SpatialMoELayer  (one independent instance per scale)
      Router -> noisy top-k (k=2) -> sparse dispatch to N experts
  SpatialMoEDecoder
      EntropyFusionBlock per scale (features + normalized routing entropy)
      GlobalCrossAttentionBlock  1/16 -> 1/8, plus a pooled global context into 1/4
      WindowedCrossAttentionBlock 1/8 -> 1/4
      RefinementBlock x3, bilinear upsampling back to 384x384
      saliency_head, edge_head, and three aux heads (deep supervision)
```

## Backbone — `src/backbone.py`

`MultiScaleBackbone` wraps a timm model with `features_only=True` and
`out_indices=(0, 1, 2)`, then applies a 1x1 convolution per scale to a common width.
Channel counts are read from `feature_info.channels()` rather than assumed, so any timm
pyramid backbone with 1/4, 1/8 and 1/16 levels would fit.

**Caveat.** The model name is hard-coded in `SpatialMoESODNet.__init__`:
`MultiScaleBackbone(d=dim, pretrained=...)`. `config.model.backbone` never reaches it, and
is read only by the experiment-ID builder. Changing that field today produces a run whose
*name* claims a different backbone and whose *weights* are still PVTv2-B4. Do not report a
backbone comparison until the field is wired through.

## Mixture-of-Experts — `src/moe_layer.py`

Each scale has its own `SpatialMoELayer` with its own router and expert pool.

| Element | Implementation |
|---|---|
| Router input | local depthwise-conv context concatenated with a pooled global context |
| Gating | noisy top-k, Shazeer-style, with a learned softplus noise scale (training only) |
| `gate_mode="renormalized"` | softmax over the selected top-k values |
| `gate_mode="dense"` | softmax over all experts, gathered at the top-k indices — every expert receives gradient |
| Dispatch | sparse: a Python loop over experts with mask selection and scatter-add |
| Every expert is called each forward pass | required so DDP's gradient reduction does not desync across ranks with different expert usage |
| Expert | token-wise MLP: LayerNorm -> 4C -> C, with a residual |
| Entropy | computed over the full expert distribution, per token, per scale |

Three ablation arms are selectable through `moe_type` and are asserted to match the config
at construction (`assert_model_matches_config` in `src/model.py`):

| `moe_type` | Behaviour |
|---|---|
| `sparse` | routed top-k experts — the model under study |
| `dense` | one shared expert applied to every token: the capacity-matched control |
| `none` | no MoE; features pass through |

**None of these arms has been run.** `results/` contains zero `M_DENSE` or `M_NONE` runs.
See `RESULTS.md` §5 and `RESEARCH_TRUTH.md` §3.

`moe_16_mode="dense"` replaces only the 1/16-scale layer with a single-expert adapter.

## Decoder — `src/decoder/`

`SpatialMoEDecoder` (`decoder.py`) assembles blocks from `blocks.py`:

1. `EntropyFusionBlock` on each scale: `y_proj` plus a learnable-scaled projection of the
   normalized entropy channel. The scale is a learned parameter initialized to 0.1, and
   `disable_entropy=True` skips it for the ablation.
   **Note:** the normalization constant is hard-coded as `1 / ln(8)`, matching 8 experts.
   For a four-expert configuration the entropy channel is therefore scaled differently —
   worth knowing before comparing entropy-fused arms across expert counts.
2. `GlobalCrossAttentionBlock`: LayerNorm + 8-head `nn.MultiheadAttention` over flattened
   tokens, plus a residual FFN. Fuses 1/16 into 1/8.
3. A pooled global context is added to the 1/4 features (`global_context_pool` +
   `global_context_proj`).
4. `WindowedCrossAttentionBlock`: window-partitioned attention with a relative position
   bias table and a padding mask, so non-divisible spatial sizes still round-trip exactly.
   Fuses 1/8 into 1/4.
5. `RefinementBlock` x3 (Conv3x3 -> GroupNorm(32) -> ReLU, twice, residual) with bilinear
   x2 upsampling between them.
6. Heads: 1x1 `saliency_head` and `edge_head` at full resolution; with
   `use_deep_supervision`, 1x1 aux heads at 1/16, 1/8 and 1/4.

## Output shapes (asserted in the code)

| Tensor | Shape |
|---|---|
| `saliency_logits` | `[B, 1, 384, 384]` |
| `boundary_logits` | `[B, 1, 384, 384]` |
| `aux_logits_16` / `_8` / `_4` | `[B, 1, 24, 24]` / `[B, 1, 48, 48]` / `[B, 1, 96, 96]` |
| `routing_probs` per scale | `[B, E, H_s, W_s]` |
| `entropy` per scale | `[B, 1, H_s, W_s]` |

## Inference-time ablation hooks — `SpatialMoESODNet.forward`

An optional `ablation_cfg` supports `scale` (4/8/16) with `expert_id` to force every token
to one expert, `random_routing` to replace learned gates with random logits, and
`disable_entropy` to drop the entropy channel. These are what the proxy-ablation table in
`RESULTS.md` measures; they perturb a trained model rather than retraining it.

## Measured behaviour

Routing is close to uniform: per-token entropy sits at the maximum of its range
(normalized 0.9995–0.9998 at every scale), the router's logit spread is ~0.03–0.07, and
per-expert weather distributions barely deviate from the dataset prior. Dead experts
persist under the renormalized gate. Numbers and sources: `RESULTS.md` §4 and
`RESEARCH_TRUTH.md` §2.1.

## Size

69,213,120 parameters for the E8 k=2 configuration, as reported by `build_model` and in
training logs. A legacy `compute_cost.json` reports 66.27M; the two have not been
reconciled — see `RESEARCH_TRUTH.md`.
