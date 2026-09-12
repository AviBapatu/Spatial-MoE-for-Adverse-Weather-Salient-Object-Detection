"""Shared pytest fixtures for the test suite.

Skips tests marked ``gpu`` when CUDA is unavailable, so the suite can run on
CPU-only runners (e.g. GitHub Actions) without failing on GPU-dependent tests.
On GPU-equipped machines all tests run unconditionally.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _skip_gpu_tests_without_cuda(request: pytest.FixtureRequest) -> None:
    """Skip tests marked ``gpu`` when CUDA is not available on this machine."""
    if torch.cuda.is_available():
        return
    if request.node.get_closest_marker("gpu"):
        pytest.skip("requires CUDA (not available in this environment)")