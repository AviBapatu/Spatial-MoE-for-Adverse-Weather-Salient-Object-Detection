"""Checkpoint push / pull / integrity against the ``checkpoints/`` folder.

Owns the off-hot-path checkpoint verification that Session 1's
``src/training/checkpoint.py`` flow should call instead of the synchronous
``torch.load`` right after ``torch.save``: :func:`verify_checkpoint_integrity`
is read-only and safe to run from the background HF-upload thread that already
reads/hashes the checkpoint while the next epoch trains.

Every checkpoint the tutorial pushes is snapshotted to a temp file first
(see :func:`push_checkpoint`), so the sha256 stored in the manifest always
matches the bytes actually uploaded — even if training atomically replaces the
live ``.pth`` path (tmp → ``os.replace``) between our hash and our upload.
"""

import json
import os
import queue
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, Optional

import torch

from src.hf_sync.hashing import _get_repo_id, _get_token, ensure_repo, get_api, sha256_file
from src.log import get_logger

log = get_logger(__name__)

CHECKPOINT_REMOTE_DIR = "checkpoints"
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"


# --------------------------------------------------------------------------- #
# Integrity verification (background-thread safe)
# --------------------------------------------------------------------------- #

def verify_checkpoint_integrity(path: str, expected_sha256: Optional[str] = None) -> bool:
    """Verify that ``path`` is a complete, loadable checkpoint file.

    Checks, in order:
      1. the file exists;
      2. its sha256 matches ``expected_sha256`` when given; and
      3. ``torch.load`` can deserialize it (``weights_only=False``, matching
         the rest of the codebase).

    Purely read-only and never raises — returns ``True``/``False`` and prints
    a warning on failure. Safe to call from the background HF-upload thread
    while training continues, because ``save_checkpoint`` writes via tmp →
    ``os.replace`` and the final path therefore never exposes a stale or
    partially-written file (a reader sees the old or the new complete file).

    Args:
        path: checkpoint path to inspect.
        expected_sha256: optional manifest sha the file must match.

    Returns:
        ``True`` when the file is intact (and matches ``expected_sha256``).
    """
    if not os.path.exists(path):
        log.warning(f"[hf_sync] verify_checkpoint_integrity: missing file {path}")
        return False

    try:
        if expected_sha256 is not None:
            actual = sha256_file(path)
            if actual != expected_sha256:
                log.warning(
                    f"[hf_sync] verify_checkpoint_integrity: sha256 mismatch for {path} "
                    f"expected={expected_sha256} actual={actual}"
                )
                return False
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception as exc:
        log.warning(f"[hf_sync] verify_checkpoint_integrity: {path} failed to load: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Checkpoint push / pull
# --------------------------------------------------------------------------- #

def _snapshot_file(path: str) -> str:
    """Copy ``path`` to a temp sibling; returns the temp path.

    The upload pipeline hashes and uploads this snapshot, never the live path,
    so the manifest sha can't drift from the uploaded bytes if training
    replaces ``path`` in the middle of a push. The caller must remove the
    returned temp file.
    """
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(path)) or ".",
        prefix=f"{os.path.basename(path)}.",
        suffix=".snapshot.pth",
    )
    os.close(fd)
    shutil.copy2(path, tmp_path)
    return tmp_path


def _fetch_remote_checkpoint_manifest(repo_id: str, token: str) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=f"{CHECKPOINT_REMOTE_DIR}/{CHECKPOINT_MANIFEST_NAME}",
            token=token,
        )
        with open(path) as f:
            return json.load(f)
    except EntryNotFoundError:
        return {}
    except Exception:
        return {}


def push_checkpoint(
    local_path: str,
    name: Optional[str] = None,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    verify: bool = False,
) -> str:
    """Upload one checkpoint file (best.pth / latest.pth / whatever) and
    update the shared checkpoint manifest. Manifest goes up last.

    Args:
        local_path: checkpoint file to push.
        name: remote filename; defaults to ``os.path.basename(local_path)``.
        repo_id: HF dataset repo id (or ``HF_REPO_ID`` env).
        token: HF token (or ``HF_TOKEN`` env).
        extra_meta: extra fields merged into the manifest entry for this file.
        verify: when True, run :func:`verify_checkpoint_integrity` on the
            exact bytes about to be uploaded and refuse to push on failure
            (intended to be used from the background pusher, never the hot path).

    Returns:
        The sha256 of the uploaded bytes.
    """
    repo_id = ensure_repo(repo_id, token)
    token = _get_token(token)
    name = name or os.path.basename(local_path)

    # Hash + upload the same snapshot, never the live file: the checkpoint at
    # local_path may be atomically replaced by the next epoch's save while this
    # push is in flight, which would otherwise leave the manifest sha pointing
    # at different bytes than what actually landed on the Hub.
    snapshot = _snapshot_file(local_path)
    try:
        sha = sha256_file(snapshot)
        size = os.path.getsize(snapshot)

        if verify and not verify_checkpoint_integrity(snapshot, expected_sha256=sha):
            raise RuntimeError(
                f"Refusing to push {name}: integrity check failed on the bytes "
                "about to be uploaded (see warnings above)."
            )

        api = get_api(token)
        api.upload_file(
            path_or_fileobj=snapshot,
            path_in_repo=f"{CHECKPOINT_REMOTE_DIR}/{name}",
            repo_id=repo_id, repo_type="dataset",
            commit_message=f"push {name} ({sha[:12]})",
        )
    finally:
        if os.path.exists(snapshot):
            os.remove(snapshot)

    manifest = _fetch_remote_checkpoint_manifest(repo_id, token)
    manifest.setdefault("files", {})
    manifest["files"][name] = {
        "sha256": sha,
        "size_bytes": size,
        "updated_at": time.time(),
        **(extra_meta or {}),
    }
    manifest["updated_at"] = time.time()

    tmp_manifest = local_path + f".{name}.manifest.tmp.json"
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    api.upload_file(
        path_or_fileobj=tmp_manifest,
        path_in_repo=f"{CHECKPOINT_REMOTE_DIR}/{CHECKPOINT_MANIFEST_NAME}",
        repo_id=repo_id, repo_type="dataset",
        commit_message=f"update checkpoint manifest ({name})",
    )
    os.remove(tmp_manifest)
    return sha


def pull_checkpoint(name: str, local_path: str, repo_id: Optional[str] = None,
                    token: Optional[str] = None, force: bool = False) -> bool:
    """Download a checkpoint only if the local copy is missing or stale
    (sha256 mismatch against the remote manifest). Returns True if a file
    was (re)downloaded, False if the local copy already matched."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    repo_id = _get_repo_id(repo_id)
    token = _get_token(token)

    manifest = _fetch_remote_checkpoint_manifest(repo_id, token)
    remote_meta = manifest.get("files", {}).get(name)

    if not force and remote_meta and os.path.exists(local_path):
        if sha256_file(local_path) == remote_meta.get("sha256"):
            log.info(f"{name}: local copy already matches remote (sha256 match), skipping download.")
            return False

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=f"{CHECKPOINT_REMOTE_DIR}/{name}",
            token=token,
        )
    except EntryNotFoundError:
        log.info(f"{name}: not found in {repo_id} — nothing to pull yet.")
        return False

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    shutil.copy2(downloaded, local_path)

    if remote_meta and remote_meta.get("sha256"):
        actual = sha256_file(local_path)
        if actual != remote_meta["sha256"]:
            raise RuntimeError(
                f"{name}: downloaded sha256 {actual} != manifest sha256 {remote_meta['sha256']}"
            )
    log.info(f"{name}: pulled to {local_path}.")
    return True


def push_json(local_path: str, name: Optional[str] = None,
              repo_id: Optional[str] = None, token: Optional[str] = None) -> None:
    """Push a small marker file (training_complete.json etc.) into
    checkpoints/. Not sha-tracked in the manifest since these are status
    files, not integrity-critical binaries."""
    repo_id = ensure_repo(repo_id, token)
    token = _get_token(token)
    name = name or os.path.basename(local_path)
    api = get_api(token)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{CHECKPOINT_REMOTE_DIR}/{name}",
        repo_id=repo_id, repo_type="dataset",
        commit_message=f"push {name}",
    )


# --------------------------------------------------------------------------- #
# Async background pusher — used inside the training loop so a checkpoint
# upload never blocks the GPUs waiting on network I/O.
# --------------------------------------------------------------------------- #

class AsyncCheckpointPusher:
    """One background thread, a FIFO queue of push jobs. Call .enqueue(...)
    from the training loop (non-blocking); call .flush(timeout) once at the
    very end (or in a `finally`) to make sure nothing is lost if the Kaggle
    session ends right after the script exits."""

    def __init__(self, repo_id: Optional[str] = None, token: Optional[str] = None):
        self.repo_id = repo_id
        self.token = token
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            job = self._q.get()
            if job is None:
                self._q.task_done()
                break
            kind, args, kwargs = job
            try:
                if kind == "checkpoint":
                    push_checkpoint(*args, repo_id=self.repo_id, token=self.token, **kwargs)
                elif kind == "json":
                    push_json(*args, repo_id=self.repo_id, token=self.token, **kwargs)
                log.info(f"[hf_sync] background push OK: {kind} {args[0] if args else ''}")
            except Exception as e:
                # Never crash training because the upload failed — the local
                # checkpoint on disk is still safe. Log and move on; the
                # next checkpoint push will naturally re-sync a newer file.
                log.warning(f"[hf_sync] WARNING: background push failed ({kind} {args[0] if args else ''}): {e}")
            finally:
                self._q.task_done()

    def enqueue_checkpoint(self, local_path: str, name: Optional[str] = None,
                           extra_meta: Optional[Dict[str, Any]] = None,
                           verify: bool = False) -> None:
        """Queue a checkpoint push (non-blocking).

        Args:
            local_path: checkpoint file to upload.
            name: remote filename; defaults to the basename.
            extra_meta: extra manifest fields for this file.
            verify: when True, the background worker verifies the exact bytes
                it is about to upload (:func:`verify_checkpoint_integrity`)
                and skips the push on failure. This is the off-hot-path home
                for the post-``torch.save`` integrity check.
        """
        self._q.put(("checkpoint", (local_path,), {"name": name, "extra_meta": extra_meta, "verify": verify}))

    def enqueue_json(self, local_path: str, name: Optional[str] = None) -> None:
        self._q.put(("json", (local_path,), {"name": name}))

    def flush(self, timeout: int = 600) -> None:
        """Block until all queued pushes complete, or timeout seconds pass.
        Always call this before the process exits."""
        start = time.time()
        while not self._q.empty() and (time.time() - start) < timeout:
            time.sleep(1)
        if not self._q.empty():
            log.warning(f"[hf_sync] WARNING: flush timed out after {timeout}s with pushes still pending.")

    def close(self, timeout: int = 600) -> None:
        self.flush(timeout)
        self._q.put(None)
        self._thread.join(timeout=30)
