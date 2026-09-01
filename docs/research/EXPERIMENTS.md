# Experiments

## Configuration System

### Config Files
- **Canonical:** `experiments/baseline_v1.json`
- **Dataclass definitions:** `src/config.py`

### Config Structure (`src/config.py`)

Nested dataclasses:
```python
ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    opt: OptimizationConfig
    train: TrainingConfig
    eval: EvaluationConfig
    diag: DiagnosticsConfig
```

### Baseline Configuration (`experiments/baseline_v1.json`)

```json
{
    "data": {
        "dataset_root": "/kaggle/input/wxsod-dataset",
        "max_samples": null
    },
    "model": {
        "backbone": "pvt_v2_b4",
        "working_dim": 256,
        "num_experts": 8,
        "top_k": 2,
        "router_variant": "token_only",
        "moe_type": "sparse",
        "window_size": 7,
        "deep_supervision": false
    },
    "loss": {
        "bce_weight": 1.0,
        "iou_weight": 1.0,
        "ssim_weight": 0.0,
        "boundary_weight": 0.0,
        "load_balance_weight": 0.01,
        "importance_weight": 0.01,
        "deep_supervision_weight": 0.4
    },
    "opt": {
        "optimizer": "AdamW",
        "backbone_lr": 1e-4,
        "new_module_lr": 1e-4,
        "weight_decay": 1e-4,
        "warmup_ratio": 0.01,
        "grad_accum_steps": 16,
        "effective_global_batch": 32,
        "batch_per_gpu": 1
    },
    "train": {
        "epochs": 50,
        "freeze_backbone_epochs": 1,
        "seed": 42
    }
}
```

### Canonical Hash

**Purpose:** Identical configs produce identical hashes; different configs produce different hashes.

**Computation** (`config.py:109-135`):
1. Convert config to dict
2. Remove volatile keys: `experiment_id`, `run_id`, `batch_equivalence`
3. Remove runtime paths: `dataset_root`, `split_manifest_path`
4. Remove runtime params: `num_workers`, `checkpoint_every_n_steps`
5. Sort all dicts recursively
6. SHA256 of compact JSON

**Used for:** Resume validation (`train_ddp.py:348-349`)

## Experiment ID System (`src/experiment.py`)

### ID Format
```
EXP_{BB}_E{e}_K{k}_S{s}_{router}_{loss}_{moe_type}
```

**Example:** `EXP_B4_E8_K2_S32_R1_L1_M_SPARSE`

**Components:**
- `BB`: Backbone abbreviation (PVT_V2_B4 → B4)
- `E{e}`: Number of experts
- `K{k}`: Top-k value
- `S{s}`: Effective global batch size
- `R1/R2/R3`: Router variant (token_only/token_local/global)
- `L1/L2/L3`: Loss variant (BCE+IoU / +SSIM / +Boundary)
- `M_NONE/M_DENSE/M_SPARSE`: MoE type

### Run ID
```
{experiment_id}__{timestamp}_{uuid[:8]}
```

### Experiment Registry (`experiments/registry.csv`)

CSV tracking all runs with fields:
```
run_id, experiment_id, config_hash, git_commit, seed,
backbone, experts, top_k, router_variant, loss_variant,
moe_type, batch_equivalence, validation_primary_metric,
validation_primary_value, best_epoch, best_global_step,
status, timestamp
```

**Status values:** CREATED → RUNNING → COMPLETED/FAILED

## Existing Results

### Checkpoint: `checkpoints/best_new_1.pth`

**Evaluation output:** `evaluation/best_new_1/`

#### Synthetic Test (test_sys) — 1,500 images

| Weather | Count | MAE | S_measure | F_max | boundary_F1 |
|---------|-------|-----|-----------|-------|-------------|
| Global | 1500 | 0.0192 | 0.9139 | 0.9015 | 0.5472 |
| clean | 167 | 0.0142 | 0.9329 | 0.9228 | 0.6011 |
| fog | 166 | 0.0166 | 0.9308 | 0.9255 | 0.5953 |
| rain | 179 | 0.0200 | 0.9173 | 0.9030 | 0.5538 |
| snow | 172 | 0.0189 | 0.9192 | 0.9068 | 0.5443 |
| dark | 156 | 0.0199 | 0.9049 | 0.8836 | 0.5276 |
| light | 166 | 0.0215 | 0.9091 | 0.9046 | 0.5404 |
| rainafog | 172 | 0.0237 | 0.8902 | 0.8691 | 0.4962 |
| rainasnow | 166 | 0.0194 | 0.9152 | 0.9032 | 0.5459 |
| snowafog | 156 | 0.0189 | 0.9051 | 0.8950 | 0.5168 |

**Best performance:** clean (MAE=0.0142, S=0.9329)
**Worst performance:** rainafog (MAE=0.0237, S=0.8902)

#### Real-World Test (test_real) — 554 images

| Weather | Count | MAE | S_measure | F_max | boundary_F1 |
|---------|-------|-----|-----------|-------|-------------|
| Global | 554 | 0.0168 | 0.9151 | 0.8936 | 0.3601 |
| fog | 126 | 0.0148 | 0.9179 | 0.8914 | 0.4004 |
| rain | 120 | 0.0165 | 0.9157 | 0.8935 | 0.3408 |
| snow | 90 | 0.0094 | 0.9478 | 0.9421 | 0.4525 |
| dark | 125 | 0.0191 | 0.9072 | 0.8885 | 0.4151 |
| light | 93 | 0.0243 | 0.8897 | 0.8629 | 0.2302 |

**Best performance:** snow (MAE=0.0094, S=0.9478)
**Worst performance:** light (MAE=0.0243, S=0.8897)

### Key Observations

1. **Real-world performance close to synthetic** — MAE 0.0168 vs 0.0192 (real is slightly better)
2. **Snow is easiest** — lowest MAE on real-world (0.0094)
3. **Low-light is hardest** — highest MAE on real-world (0.0243)
4. **Compound weather degrades most** — rainafog worst on synthetic (0.0237)
5. **Boundary F1 is low overall** — 0.36-0.55 range, indicating boundary detection is challenging

## Evaluation Metrics (`src/metrics.py`)

### Standard SOD Metrics (via py_sod_metrics)

- **MAE:** Mean Absolute Error
- **S_measure:** Structure measure (affinity-based)
- **E_measure:** Enhanced alignment measure (adaptive/mean/max)
- **F_measure:** F-measure (adaptive/mean/max)

### Custom Boundary Metrics (`metrics.py:61-114`)

- **Boundary MAE:** MAE computed only on GT boundary pixels
- **Boundary F1:** F1 score of binarized boundary prediction vs GT boundary

### Evaluation Pipeline (`src/evaluate.py`)

```bash
uv run python -m src.evaluate --checkpoint checkpoints/best.pth --dataset both --tta none
```

**Geometry reversal** (`evaluate.py:23-45`):
1. Unpad: crop to `(resized_h, resized_w)` from `(pad_top, pad_left)`
2. Resize: `(orig_w, orig_h)` using bilinear interpolation

**Output:** `evaluation/{checkpoint_name}/{dataset}/{tta_mode}/` containing:
- `metrics.json` — full metrics with weather breakdown
- `summary.txt` — human-readable summary
- `evaluation_manifest.json` — metadata
- `*.png` — per-image prediction maps
