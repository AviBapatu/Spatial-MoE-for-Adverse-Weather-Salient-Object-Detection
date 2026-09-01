# Project Overview

## Title
Spatially Dynamic Mixture-of-Experts for Adverse-Weather Salient Object Detection

## Problem Statement
Salient Object Detection (SOD) degrades severely under adverse weather conditions (fog, rain, snow, low-light). Existing approaches either use weather-specific dual-branch networks (image-level routing) or general-purpose backbones that fail to specialize. This project proposes a **token-level spatial Mixture-of-Experts** that routes individual spatial tokens to specialized experts without any weather label at train or test time.

## Core Novelty
Three-axis contribution (from `spatial-moe-adverse-weather-sod-blueprint.md`):

1. **Fully label-free spatial/token routing** — no weather-conditioned gating
2. **Independent routers at multiple pyramid scales** (1/4, 1/8, 1/16)
3. **Applied to segmentation-topology-sensitive SOD task** rather than pixel-regression restoration

## System Summary

| Component | Choice | Source |
|-----------|--------|--------|
| Backbone | PVTv2-B4 (timm) | `src/backbone.py:6` |
| Working dim | 256 | `experiments/baseline_v1.json` |
| Experts per scale | 8 | `experiments/baseline_v1.json` |
| Top-k | 2 | `experiments/baseline_v1.json` |
| Expert type | TokenWiseMLP (LN→4C→C+residual) | `src/moe_layer.py:14-33` |
| Router | DWConv3x3 + MLP (concat local+global) | `src/moe_layer.py:49-57` |
| Decoder | Cross-attention fusion + refinement blocks | `src/decoder.py:219-280` |
| Loss | BCE + IoU + load-balance + importance + deep-sup | `src/loss.py:62-219` |
| Training | DDP 2-GPU, AMP FP16, WarmupCosine | `src/train_ddp.py` |
| Dataset | WXSOD (14,945 images, 9 weather categories) | `src/dataset.py:19` |

## Repository Roles

| Directory | Purpose |
|-----------|---------|
| `src/` | All source code (model, training, evaluation, configs) |
| `tests/` | Unit + integration tests (sparse dispatch, DDP, resume) |
| `experiments/` | Canonical experiment configs (baseline_v1.json) |
| `evaluation/` | Saved evaluation results per checkpoint |
| `checkpoints/` | Model weight files (.pth) |
| `data/WXSDO_data/` | WXSOD dataset |
| `spatial-moe-adverse-weather-sod-blueprint.md` | Research blueprint with literature review |
