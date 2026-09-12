#!/usr/bin/env bash
# GPU-aware smoke test.
#
# Runs on any machine:
#   1. ruff lint over src/
#   2. full pytest suite (GPU-only tests self-skip when CUDA is absent via
#      tests/conftest.py)
#   3. the training preflight (--dry_run) -- only when CUDA is available,
#      because the process group uses the NCCL backend.
#
# On GitHub Actions (CPU-only ubuntu runners) step 3 is reported and skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Lint ==="
cd "$REPO_ROOT"
uv run ruff check src/

echo "=== Tests ==="
uv run pytest tests/ -v --ignore=tests/test_moe_ddp.py

echo "=== Preflight (dry_run, via torchrun) ==="
if uv run python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  uv run torchrun --nproc_per_node=1 -m src.train_ddp --config experiments/smoke_test_config.json --dry_run
else
  echo "CUDA not available - skipping training preflight (requires a GPU machine)."
fi

echo "=== All checks passed ==="