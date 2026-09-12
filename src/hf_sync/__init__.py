"""Hugging Face Hub sync layer — code + checkpoint sync against one HF dataset repo.

Backward-compatible re-export of the historical monolith ``src/hf_sync.py``:
all public functions/classes/constants are importable from ``src.hf_sync`` as
before (``from src.hf_sync import push_code``, ``hf_sync.AsyncCheckpointPusher``,
``python -m src.hf_sync push-code`` all keep working). Implementation lives in:

* :mod:`src.hf_sync.hashing` — sha256 + shared Hub plumbing;
* :mod:`src.hf_sync.code_sync` — push/pull/deploy code zip;
* :mod:`src.hf_sync.checkpoint_sync` — push/pull/verify checkpoints + pusher.

The manifest-hash mechanism (``archive_sha256`` / ``source_content_sha256``)
is load-bearing for reproducibility and unchanged.
"""

import argparse
import json

from src.hf_sync.checkpoint_sync import (
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_REMOTE_DIR,
    AsyncCheckpointPusher,
    _fetch_remote_checkpoint_manifest,
    pull_checkpoint,
    push_checkpoint,
    push_json,
    verify_checkpoint_integrity,
)
from src.hf_sync.code_sync import (
    CODE_MANIFEST_NAME,
    CODE_REMOTE_DIR,
    CODE_ZIP_NAME,
    _git_commit,
    deploy_code,
    pull_code,
    push_code,
)
from src.hf_sync.hashing import (
    DEFAULT_CODE_INCLUDE,
    _get_repo_id,
    _get_token,
    compute_dir_hash,
    ensure_repo,
    get_api,
    sha256_file,
)

__all__ = [
    "sha256_file",
    "compute_dir_hash",
    "get_api",
    "ensure_repo",
    "DEFAULT_CODE_INCLUDE",
    "push_code",
    "pull_code",
    "deploy_code",
    "push_checkpoint",
    "pull_checkpoint",
    "push_json",
    "verify_checkpoint_integrity",
    "AsyncCheckpointPusher",
    "CODE_REMOTE_DIR",
    "CODE_ZIP_NAME",
    "CODE_MANIFEST_NAME",
    "CHECKPOINT_REMOTE_DIR",
    "CHECKPOINT_MANIFEST_NAME",
    "_git_commit",
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cli() -> None:
    parser = argparse.ArgumentParser(description="HF Hub sync for spatial-moe-sod")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push_code = sub.add_parser("push-code", help="Zip + hash + upload code from this machine")
    p_push_code.add_argument("--root", default=".")
    p_push_code.add_argument("--version", default=None)

    p_pull_code = sub.add_parser("pull-code", help="Download the current code archive")
    p_pull_code.add_argument("--dest", default="./hf_cache")

    p_push_ckpt = sub.add_parser("push-checkpoint", help="Upload a checkpoint file")
    p_push_ckpt.add_argument("--file", required=True)
    p_push_ckpt.add_argument("--name", default=None)

    p_pull_ckpt = sub.add_parser("pull-checkpoint", help="Download a checkpoint file")
    p_pull_ckpt.add_argument("--name", required=True)
    p_pull_ckpt.add_argument("--dest", required=True)
    p_pull_ckpt.add_argument("--force", action="store_true")

    sub.add_parser("status", help="Show what's currently on the Hub")

    args = parser.parse_args()

    if args.cmd == "push-code":
        push_code(project_root=args.root, source_version=args.version)
    elif args.cmd == "pull-code":
        zip_path, manifest = pull_code(args.dest)
        print(json.dumps(manifest, indent=2))
        print(f"zip: {zip_path}")
    elif args.cmd == "push-checkpoint":
        sha = push_checkpoint(args.file, name=args.name)
        print(f"Pushed. sha256={sha}")
    elif args.cmd == "pull-checkpoint":
        pull_checkpoint(args.name, args.dest, force=args.force)
    elif args.cmd == "status":
        repo_id = _get_repo_id()
        token = _get_token()
        manifest = _fetch_remote_checkpoint_manifest(repo_id, token)
        print(f"Repo: {repo_id}")
        print(json.dumps(manifest, indent=2) if manifest else "(no checkpoint manifest yet)")


if __name__ == "__main__":
    _cli()
