# AGENTS.md — Spatial-MoE for Adverse-Weather Salient Object Detection

## 0. Default Mode vs. Engineering Mode

This repository is used two ways, and they have different rules:

- **Default mode (research/paper-writing).** Most sessions here are about understanding, documenting, and writing about a codebase that already exists. Default mode is **read-only for source code** (see Rule 8 below). This is what the rest of this file assumes unless a session says otherwise.
- **Engineering mode (explicit code-change sessions).** Some sessions are explicitly scoped to rewrite or fix code — e.g. a DDP bug fix, a refactor, a module split. These sessions override Rule 8's "don't modify code" default *for the files they're explicitly scoped to touch, and no others*. Engineering-mode rules live in **Section 8**. If a prompt doesn't explicitly say "modify these files," treat the session as default mode.

## 1. What This Project Is

A research codebase for **Spatially Dynamic Mixture-of-Experts for Adverse-Weather Salient Object Detection (SOD)**. The system routes individual spatial tokens to specialized expert networks without any weather label at train or test time, enabling robust SOD under fog, rain, snow, and low-light conditions.

The target output is a scientific paper. This file and `docs/research/` are the persistent research context for paper-writing sessions.

## 2. Major Repository Directories

| Directory | Purpose |
|-----------|---------|
| `src/` | All source code: model, training, evaluation, config, dataset, diagnostics |
| `tests/` | Unit + integration tests (sparse dispatch, DDP, resume correctness) |
| `experiments/` | Canonical experiment config JSONs (`baseline_v1.json`) |
| `evaluation/` | Saved evaluation results per checkpoint (`metrics.json`, `summary.txt`) |
| `checkpoints/` | Model weight files (`.pth`) |
| `data/WXSDO_data/` | WXSOD dataset (train_sys, test_sys, test_real splits) |
| `docs/research/` | Persistent research knowledge base for paper writing |
| `frontend/` | Web inference UI (FastAPI + static files) |

## 3. How the Model Is Organized

**Entry point:** `src/model.py` → `SpatialMoESODNet`

```
Input [B, 3, 384, 384]
  → MultiScaleBackbone (PVTv2-B4 via timm)
      → res_4: [B, 256, 96, 96]    (1/4 scale, 9216 tokens)
      → res_8: [B, 256, 48, 48]    (1/8 scale, 2304 tokens)
      → res_16: [B, 256, 24, 24]   (1/16 scale, 576 tokens)
  → Three independent SpatialMoELayer (one per scale)
      → Router: DWConv3x3 + MLP, noisy top-k (k=2)
      → Experts: 8 × TokenWiseMLP (LN→4C→C + residual)
      → True sparse dispatch (loop over experts, mask selection)
  → SpatialMoEDecoder
      → EntropyFusionBlock per scale (features + scaled entropy channel)
      → Top-down cross-attention: Global (1/16→1/8), Windowed (1/8→1/4)
      → RefinementBlock → bilinear upsample to 384×384
      → Saliency head + Boundary head (1×1 conv each)
```

**Key source files:**
- `src/backbone.py` — PVTv2-B4 feature extraction
- `src/moe_layer.py` — Router + expert pool + sparse dispatch
- `src/decoder.py` — Cross-attention fusion + prediction heads
- `src/loss.py` — BCE + IoU + load-balance + importance + deep supervision

## 4. How Experiments Are Run

```bash
# Training (2 GPUs, DDP)
torchrun --nproc_per_node=2 -m src.train_ddp --config experiments/baseline_v1.json

# Resume from checkpoint
torchrun --nproc_per_node=2 -m src.train_ddp --config experiments/baseline_v1.json --resume latest

# Evaluation
uv run python -m src.evaluate --checkpoint checkpoints/best.pth --dataset both

# Smoke tests (pre-flight)
uv run python -m src.smoke_test --mode smoke1gpu --data_root data/WXSDO_data --result_file /tmp/r.json
torchrun --nproc_per_node=2 -m src.smoke_test --mode ddp --data_root data/WXSDO_data --result_file /tmp/r.json

# Run tests
uv run pytest tests/ -v
```

**Config system:** `experiments/baseline_v1.json` → dataclasses in `src/config.py` → SHA256 canonical hash for resume validation.

**Ablation configs:** Generated programmatically by `src/ablations.py` from baseline, tracked in `experiments/registry.csv`.

## 5. Where Experimental Results Are Stored

| Location | Contents |
|----------|----------|
| `evaluation/best_new_1/` | Evaluation results for `best_new_1.pth` checkpoint |
| `evaluation/best_new_1/test_sys/none/` | Synthetic test metrics + predictions |
| `evaluation/best_new_1/test_real/none/` | Real-world test metrics + predictions |
| `checkpoints/best.pth` | Best model weights (by val MAE) |
| `checkpoints/best_new_1.pth` | Another best model (source of current evaluation) |
| `experiments/registry.csv` | Run registry (not yet created — no formal ablation runs recorded) |

**Result file format:**
- `metrics.json` — Full metrics with weather-wise breakdown
- `summary.txt` — Human-readable summary
- `*.png` — Per-image prediction maps

## 6. Where Research Documentation Is Stored

| File | Purpose |
|------|---------|
| `docs/research/PROJECT_OVERVIEW.md` | Title, novelty claims, system summary |
| `docs/research/ARCHITECTURE.md` | Model architecture with verified tensor dims |
| `docs/research/TRAINING.md` | Training pipeline, loss, optimization, checkpointing |
| `docs/research/DATASETS.md` | WXSOD dataset structure, splitting, augmentation |
| `docs/research/EXPERIMENTS.md` | Config system, baseline, existing results |
| `docs/research/ABLATIONS.md` | Ablation matrices, counterfactual studies |
| `docs/research/PAPER_PLAN.md` | Suggested paper structure, figures, open questions |
| `docs/research/LITERATURE_NOTES.md` | Key papers, differentiators, benchmarks |
| `spatial-moe-adverse-weather-sod-blueprint.md` | Research blueprint with literature review |

## 7. Where the Paper Manuscript Is Stored

No paper manuscript file exists yet in the repository. The `docs/research/` directory serves as the persistent knowledge base from which the paper will be written. `PAPER_PLAN.md` contains the suggested paper structure.

## Rules for This Project (Default Mode)

1. **Never invent implementation details.** If unsure about how something is implemented, read the source file. Every architecture claim should reference a file and line number.

2. **Never invent experimental results.** Use only values from `evaluation/` or `metrics.json`. If results don't exist, say so.

3. **Never invent citations.** Use only papers listed in `docs/research/LITERATURE_NOTES.md` or `spatial-moe-adverse-weather-sod-blueprint.md`. If a citation is needed and not listed, flag it as unknown.

4. **Distinguish verified facts from inference.** When stating something, note whether it comes from reading the code (verified) or from interpretation (inferred). The research docs separate these explicitly where possible.

5. **Use the repository as the source of truth for implementation.** The code defines what the system does, not documentation or blueprints. If blueprint and code conflict, trust the code.

6. **Use `docs/research/` as the persistent research knowledge base.** Read the relevant file before answering architecture/experiment/paper questions. Update these files when new information is discovered.

7. **Verify mathematical notation against implementation.** Before writing equations in the paper, check the actual code in `src/loss.py`, `src/moe_layer.py`, and `src/decoder.py`. The blueprint formulas are aspirational; the code is definitive.

8. **Do not modify source code unless explicitly asked.** This is a research codebase. Code changes should only happen with explicit user instruction. Analysis and documentation are the default mode. "Explicitly asked" means the session prompt names specific files/modules to change — see Section 8 for what governs those sessions once that bar is met.

9. **If a code-change session touches something a research doc describes, flag it.** You don't have to update `docs/research/` yourself unless asked, but if Section 8 work changes a tensor shape, a loss term, a config field, or anything else `ARCHITECTURE.md`/`TRAINING.md`/`EXPERIMENTS.md` claims, say so explicitly in your summary — those docs become stale silently otherwise, and Rule 5/Evidence Hierarchy depend on them not drifting from the code.

## 8. Engineering Mode — Rules That Apply Only to Explicit Code-Change Sessions

These rules govern any session explicitly scoped to modify `src/`, `tests/`, or `train.py` — i.e. sessions that override Rule 8's default. They do not apply to research/paper-writing sessions.

### 8.1 Non-negotiable invariants

- `ExperimentConfig` (`src/config.py`) is the single source of truth for all hyperparameters. Never hardcode a hyperparameter that already has a config field.
- Every `DataLoader` reachable from the training or evaluation entrypoint must be rank-aware (`DistributedSampler`) when running distributed — not just the train loader. This codebase has previously shipped with `val_loader` silently missing a `DistributedSampler` while `train_loader` had one; treat this class of bug as something to actively check for, not assume is fixed.
- Any code path gated by `if rank == 0:` must not contain collective/expensive work (validation forward passes, diagnostics forward passes) that leaves other ranks idle for more than a few seconds. If a block only rank 0 should do (logging, checkpoint writing, HF/artifact upload), it must be provably short, or must be followed by a `dist.monitored_barrier(timeout=...)` so a stall is caught immediately with a clear error instead of a downstream NCCL timeout.
- `dist.init_process_group` must always be called with an explicit `timeout=timedelta(...)`. Never rely on the 600s NCCL default as the actual safety margin.
- Checkpoint I/O (`torch.save`/`torch.load`) must never block the training hot path synchronously for verification; integrity checks belong off the critical path (background thread, separate step).
- Every expert module in `SpatialMoELayer` must be touched by the forward pass on every rank every step (even with zero routed tokens routed to it), or DDP's gradient sync will desync across ranks with different expert-usage patterns. The existing sparse-dispatch pattern in `moe_layer.py` already does this correctly — preserve it exactly; do not "optimize" it into conditional expert calls.
- No dead code: if a file is renamed or replaced, delete the old file and its `__pycache__` entry. Do not leave orphaned modules that duplicate functionality.

### 8.2 Style

- Type hints on all new/edited function signatures.
- Docstrings on every public function/class (purpose, args, returns — a few lines, not essays).
- No function longer than ~60 lines; extract helpers instead of nesting.
- No file longer than ~400 lines; split into a package if it grows past that.
- Prefer explicit imports over wildcard imports.
- Every module needs at least a smoke-level test in `tests/`.

### 8.3 Scope discipline

- Only edit the files a code-change session explicitly names. If you notice something else that looks wrong outside that scope, note it in your summary instead of fixing it — a different session owns that file.
- Before changing a shared interface (config field names, `get_dataloaders()`'s signature, the model's `forward()` signature, the checkpoint dict format, `evaluate()`'s signature), check whether other modules depend on it exactly as-is, and update every call site you find — list every file you touched in your summary.

### 8.4 Skill

If `skills/pytorch-ddp-safety/SKILL.md` exists in this repo, read it before finishing any edit that touches `train_ddp.py`, dataloader construction, or checkpoint save/load code — it's a checklist for exactly the rank-imbalance/NCCL-timeout class of bug this codebase has hit before.

## Evidence Hierarchy

When conflicts arise between sources, use this priority order:

1. **Source code** is authoritative for implementation.
2. **Configuration files** (`experiments/*.json`) are authoritative for experiment settings.
3. **Actual logs/results** (`evaluation/`, `metrics.json`) are authoritative for experimental results.
4. **`docs/research/*.md`** contains verified summaries derived from those sources.
5. **AI-generated explanations** are never authoritative.

When there is a conflict, inspect the lower-level source and correct the higher-level summary.

## Research Truth

**`RESEARCH_TRUTH.md`** is the authoritative summary of scientific claims.
Before drafting or revising any paper section, inspect it.
Never silently add a claim that is not supported by:
- the implementation
- experimental evidence
- or a verified literature source.