---
name: pytorch-ddp-safety
description: Use whenever editing distributed (DDP/torchrun) training code, dataloaders shared across ranks, or checkpoint I/O in a multi-GPU training loop. Covers rank-imbalance, NCCL timeout, and sampler bugs.
---

# PyTorch DDP safety checklist

Before finishing any edit that touches `train_ddp.py`, `dataset.py`'s loader construction,
or checkpoint save/load code, verify:

1. **Every DataLoader used inside the training/eval loop has a DistributedSampler**
   when running distributed, not just the train loader. Grep for every `DataLoader(`
   call reachable from the training entrypoint and confirm each one takes `distributed`
   into account.
2. **No `if rank == 0:` block contains a full forward pass over a shared loader.**
   If validation/diagnostics must run, either shard the loader across ranks and
   `dist.all_gather`/`all_reduce` the results, or explicitly justify in a comment why
   it's safe for this block to be rank-0-only (e.g. it's O(ms), not O(minutes)).
3. **`dist.init_process_group` always has an explicit `timeout=timedelta(...)`.**
   Never rely on the NCCL default (600s) as the actual safety margin — set a real one
   AND fix the underlying imbalance; the timeout is a backstop, not the fix.
4. **Checkpoint writes never synchronously reload+verify on the hot path.** Save with
   `torch.save`, and if you want integrity checking, hash/reload it off-thread or in a
   background step, not inline before the next epoch starts.
5. **Any rank-0-only block that other ranks must eventually wait on is followed by a
   `dist.monitored_barrier(timeout=...)`**, so a stall surfaces as "rank 0 didn't
   arrive" instead of a random gradient all-reduce timeout three-quarters through
   the next epoch.
6. **Unused-parameter safety for MoE**: every expert module must be touched by the
   forward pass on every rank every step (even with zero routed tokens), or DDP's
   gradient sync will desync across ranks with different expert-usage patterns.
