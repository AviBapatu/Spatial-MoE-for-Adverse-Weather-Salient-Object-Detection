"""CLI entry point: ``python -m src.hf_sync push-code`` etc.

Keeps the historical monolith's ``python -m`` invocation working after the
split into a package. See :mod:`src.hf_sync` for the subcommands.
"""

from src.hf_sync import _cli

if __name__ == "__main__":
    _cli()
