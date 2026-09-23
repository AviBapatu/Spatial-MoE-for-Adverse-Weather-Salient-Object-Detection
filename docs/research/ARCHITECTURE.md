# Architecture Details

> **Note on citations.** Line-number citations below predate the current source
> layout (the decoder is now the `src/decoder/` package, the training loop lives under
> `src/training/`, the evaluation artifacts are under `results/`). Treat module and
> function names as authoritative and `RESEARCH_TRUTH.md` as the verified reference;
> numbers live in `RESULTS.md`, which is generated from the result files.

## End-to-End Pipeline

```
Input [B,3,384,384]
  → Backbone (PVTv2-B4) → {res_4, res_8, res_16}
  → 1x1 projection → [B,256,H_s,W_s] per scale
  → Independent SpatialMoELayer per scale
  → EntropyFusionBlock (features + entropy channel)
  → Top-down cross-attention fusion
  → Refinement blocks → bilinear upsample to full res
  → Saliency head (1x1 conv) → [B,1,384,384]
  → Boundary head (1x1 conv) → [B,1,384,384]
```

## Backbone (`src/backbone.py`)

**Model:** `timm.create_model('pvt_v2_b4', pretrained=True, features_only=True, out_indices=(0,1,2))`

**Raw channel dims from PVTv2-B4:** `[64, 128, 320]`

**Projection:** Three separate 1x1 convolutions map each scale to unified dimension `d=256`:
- `self.proj_4 = nn.Conv2d(feature_channels[0], d, 1)`
- `self.proj_8 = nn.Conv2d(feature_channels[1], d, 1)`
- `self.proj_16 = nn.Conv2d(feature_channels[2], d, 1)`

**Output shapes (verified at `backbone.py:62-64`):**
| Scale | Shape | Token Count |
|-------|-------|-------------|
| 1/4 | `[B, 256, 96, 96]` | 9,216 |
| 1/8 | `[B, 256, 48, 48]` | 2,304 |
| 1/16 | `[B, 256, 24, 24]` | 576 |

## Spatial MoE Layer (`src/moe_layer.py:35-165`)

### Router

**Spatial context extraction:**
```python
self.router_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
```

**MLP for routing logits:**
```python
self.router_mlp = nn.Sequential(
    nn.Linear(2 * dim, router_hidden),  # 2*dim because of concat
    nn.GELU(),
    nn.Linear(router_hidden, num_experts)
)
```

**Routing computation (`moe_layer.py:96-112`):**
1. `local_feat = self.router_dwconv(x)` — depthwise 3x3 conv for local context
2. `router_input = cat([x_tokens, local_tokens], dim=-1)` — `[B, H*W, 2C]`
3. `clean_logits = self.router_mlp(router_input)` — `[B, H*W, E]`
4. Add Gaussian noise during training: `noise = randn * softplus(noise_linear(x))`
5. Top-k selection: `topk(noisy_logits, k=K)`
6. Gate softmax over selected experts only

**Noise parameters:**
- `self.router_noise_enabled = True` (default)
- `self.router_noise_scale = 1.0` (default)

### Expert Pool

**Architecture:** `TokenWiseMLPExpert` (`moe_layer.py:14-33`)
```
LayerNorm(C) → Linear(C, 4C) → GELU → Linear(4C, C) → + residual
```
- Expansion factor: 4x
- Residual connection: `return x + z`

**Default count:** 8 experts per scale (3 scales × 8 = 24 total experts)

### Sparse Dispatch (`moe_layer.py:114-141`)

True sparse execution via Python loop:
```python
for i, expert in enumerate(self.experts):
    active_mask = (flat_topk_indices == i).any(dim=-1)
    token_active = active_mask.any(dim=-1)
    if not token_active.any(): continue
    selected_tokens = flat_x[token_active]     # [N_active, C]
    expert_out = expert(selected_tokens)         # [N_active, C]
    expert_gates = flat_topk_gates[token_active][active_mask[token_active]]
    output_tokens[token_active] += expert_out * expert_gates.unsqueeze(-1)
```

**Critical:** Experts receive `[N_active, 256]` (variable-size 2D), never `[B, C, H, W]`.

### Counterfactual Ablation (`moe_layer.py:68-94`)

When `force_expert_id` is set, the router is bypassed:
- All tokens routed to specified expert
- Routing probs set to 1.0 for that expert
- Entropy set to 0.0
- Used for single-expert knockout studies

## Decoder (`src/decoder.py`)

### Entropy Fusion (`decoder.py:14-27`)

Each scale's MoE output is fused with its routing entropy:
```python
entropy_norm = entropy / math.log(2.0 + 1e-8)  # Normalize assuming K=2
y_proj = self.proj_y(Y)          # Conv2d(in_dim, out_dim, 1)
e_proj = self.proj_entropy(entropy_norm)  # Conv2d(1, out_dim, 1)
return y_proj + self.scale * e_proj  # Learnable scale (init=0.1)
```

### Top-Down Fusion (`decoder.py:247-259`)

**1/16 → 1/8 (Global Cross-Attention):**
```python
F16_up = F.interpolate(F16, size=F8_local.shape[-2:], mode='bilinear')
F8 = GlobalCrossAttentionBlock(q_x=F16_up, kv_x=F8_local)
```

**1/8 → 1/4 (Windowed Cross-Attention):**
```python
F8_up = F.interpolate(F8, size=F4_ctx.shape[-2:], mode='bilinear')
F4 = WindowedCrossAttentionBlock(q_x=F8_up, kv_x=F4_ctx)
```

**Global context injection at 1/4:**
```python
g_ctx = self.global_context_proj(self.global_context_pool(F8))  # AdaptiveAvgPool2d(1)
F4_ctx = F4_local + g_ctx
```

### Windowed Cross-Attention (`decoder.py:74-203`)

- Window size: configurable (default 7 in baseline)
- Relative position bias table: `(2*w-1)^2 × num_heads` parameters
- Handles padding with attention mask for non-divisible dimensions
- 8 attention heads

### Refinement + Prediction (`decoder.py:205-280`)

Three `RefinementBlock`s (Conv3x3→GroupNorm32→ReLU→Conv3x3→GroupNorm32→ReLU + residual):
```
F4 → refine_4 → upsample 2x → refine_2 → upsample 2x → refine_1
→ saliency_head (1x1 conv) → [B, 1, 384, 384]
→ edge_head (1x1 conv) → [B, 1, 384, 384]
```

**Deep supervision heads** (optional, controlled by `config.model.deep_supervision`):
- `aux_head_16`: `[B, 1, 24, 24]`
- `aux_head_8`: `[B, 1, 48, 48]`
- `aux_head_4`: `[B, 1, 96, 96]`

## Output Tensor Summary

| Output | Shape | Notes |
|--------|-------|-------|
| Saliency logits | `[B, 1, 384, 384]` | Sigmoid at inference |
| Boundary logits | `[B, 1, 384, 384]` | Sigmoid at inference |
| Aux logits 16 | `[B, 1, 24, 24]` | Deep supervision |
| Aux logits 8 | `[B, 1, 48, 48]` | Deep supervision |
| Aux logits 4 | `[B, 1, 96, 96]` | Deep supervision |
| Routing probs (per scale) | `[B, E, H_s, W_s]` | Full routing probability map |
| Entropy (per scale) | `[B, 1, H_s, W_s]` | Per-token routing entropy |

## Parameter Counts

Total parameters are reported during training at `train_ddp.py:285-287`:
```python
total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
Exact counts depend on configuration; run `python -c "from src.model import SpatialMoESODNet; ..."` to verify.
