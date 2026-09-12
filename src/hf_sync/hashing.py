"""Hashing + shared HF-hub helpers used by both code and checkpoint sync.

This module owns the load-bearing integrity primitives from the historical
``src/hf_sync.py``:

* :func:`sha256_file` / :func:`compute_dir_hash` — the exact functions that
  compute ``archive_sha256`` / ``source_content_sha256`` on the push side
  (VS Code) and the verify side (Kaggle notebook). Both sides MUST keep using
  this one module or the manifest check silently no-ops, so these bodies are
  preserved verbatim from the original file.

It also holds the shared Hub plumbing (``HF_TOKEN`` / ``HF_REPO_ID`` resolution,
``get_api``, ``ensure_repo``) used identically by ``code_sync`` and the
completed ``checkpoint_sync``.
"""

import hashlib
import os
from typing import List, Optional

# Load a local .env file if python-dotenv is installed and one exists,
# so HF_TOKEN / HF_REPO_ID don't need to be exported by hand every session.
# Real env vars (and Kaggle Secrets, set explicitly in the notebook) always
# take priority over .env — this only fills in what's NOT already set.
try:
    from dotenv import load_dotenv

    _project_root_for_env = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(_project_root_for_env, ".env"), override=False)
except ImportError:
    pass

DEFAULT_CODE_INCLUDE = (
    "src",           # all model / training / eval source
    "experiments",   # experiment config JSONs (baseline_v1.json etc.)
    "requirements.txt",
    "pyproject.toml",
    "train.py",
)


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """SHA256 of a file's bytes, read in chunks so large .pth files don't
    need to be loaded into memory at once."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dir_hash(root_dir: str, include=DEFAULT_CODE_INCLUDE) -> str:
    """Deterministic SHA256 over (relative_path, file_bytes) for the tracked
    project files. This is the ONE function used by both:
      - the VS Code push (hashes what's about to be zipped)
      - the Kaggle notebook verify step (hashes what actually landed on disk)
    so a mismatch here means the deployed code is genuinely not what you
    think it is, not a naming mismatch between two different hash functions.
    """
    hasher = hashlib.sha256()
    file_list: List[tuple] = []
    for item in include:
        full = os.path.join(root_dir, item)
        if os.path.isdir(full):
            for base, _, files in os.walk(full):
                for fn in files:
                    fp = os.path.join(base, fn)
                    rel = os.path.relpath(fp, root_dir)
                    file_list.append((rel, fp))
        elif os.path.isfile(full):
            file_list.append((item, full))
    file_list.sort(key=lambda x: x[0])
    for rel, fp in file_list:
        hasher.update(rel.replace(os.sep, "/").encode("utf-8"))
        with open(fp, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def _get_token(explicit: Optional[str] = None) -> str:
    token = explicit or os.environ.get("HF_TOKEN")
    if not token:
        # Kaggle Secrets fallback, so notebooks never need a plaintext token.
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    if not token:
        raise RuntimeError(
            "No HF token found. Set the HF_TOKEN environment variable "
            "(locally) or add an HF_TOKEN secret via Kaggle's Add-ons > "
            "Secrets menu (on Kaggle). Never hardcode a token in a notebook."
        )
    return token


def _get_repo_id(explicit: Optional[str] = None) -> str:
    repo_id = explicit or os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise RuntimeError("Set HF_REPO_ID, e.g. 'Avi2006/spatial-moe-project'.")
    return repo_id


def get_api(token: Optional[str] = None):
    """Return a configured ``huggingface_hub.HfApi`` using ``token``."""
    from huggingface_hub import HfApi

    return HfApi(token=_get_token(token))


def ensure_repo(repo_id: Optional[str] = None, token: Optional[str] = None,
                repo_type: str = "dataset", private: bool = True) -> str:
    """Return the repo id, creating the repo (private dataset) if missing."""
    from huggingface_hub.utils import RepositoryNotFoundError

    repo_id = _get_repo_id(repo_id)
    api = get_api(token)
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
    return repo_id
