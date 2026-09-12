#!/usr/bin/env python
"""Thin CLI shim — delegates to src.training.cli.main().

Preserves the exact same CLI interface so existing launch scripts and
``torchrun -m src.train_ddp`` invocations keep working unchanged.
"""
from src.training.cli import main

if __name__ == "__main__":
    main()
