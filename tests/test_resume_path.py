"""``--resume latest`` must resolve inside the run's own directory.

It used to resolve against the preflight *root* rather than the run directory
under it, so a preflight resume could not find the checkpoint the run had just
written -- which is how the pre-run check failed on its first real execution.
"""
from types import SimpleNamespace

from src.training.setup import _resolve_resume_path


def test_latest_resolves_inside_the_run_directory():
    args = SimpleNamespace(resume="latest")
    assert _resolve_resume_path(args, "/kaggle/working/WXSOD_Preflight/EXP_X") == \
        "/kaggle/working/WXSOD_Preflight/EXP_X/latest.pth"


def test_latest_also_resolves_for_a_production_run_directory():
    args = SimpleNamespace(resume="latest")
    assert _resolve_resume_path(args, "/kaggle/working/WXSOD_Checkpoints/EXP_X") == \
        "/kaggle/working/WXSOD_Checkpoints/EXP_X/latest.pth"


def test_an_explicit_path_is_passed_through():
    args = SimpleNamespace(resume="/tmp/some/ckpt.pth")
    assert _resolve_resume_path(args, "/whatever") == "/tmp/some/ckpt.pth"
