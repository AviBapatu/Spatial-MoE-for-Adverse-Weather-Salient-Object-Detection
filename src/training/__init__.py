"""Training package for Spatial-MoE SOD.

Submodules:
    distributed: process group init/teardown, rank helpers, monitored_barrier.
    checkpoint: save/load/resume logic, background HF-sync.
    setup: config resolution, model/optimizer/dataloader construction, resume.
    epoch: per-epoch training loop (batches + validation + diagnostics).
    loop: distributed validation, diagnostics, periodic checkpointing.
    cli: argument parsing and main() entrypoint.
"""
