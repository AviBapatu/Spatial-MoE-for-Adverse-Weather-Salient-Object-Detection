"""
Hugging Face Hub sync layer — replaces the Kaggle-Dataset zip/manifest upload
flow and the Google-Drive checkpoint flow with a single HF dataset repo.

Repo layout (one repo, e.g. "Avi2006/spatial-moe-project", repo_type="dataset"):

    code/spatial_moe_sod_code.zip
    code/project_manifest.json
    checkpoints/latest.pth
    checkpoints/best.pth
    checkpoints/checkpoint_manifest.json
    checkpoints/training_complete.json

Design rules that matter for correctness:

1. The manifest is always uploaded LAST, after the file(s) it describes.
   If a push dies partway through, the manifest on the Hub never points at
   a file that isn't actually there yet.
2. `source_content_sha256` in the code manifest is computed with the exact
   same function (`compute_dir_hash`) on both the push side (VS Code) and
   the verify side (Kaggle notebook). Previously these used different key
   names and the check silently no-op'd — that bug is what this file fixes.
3. Checkpoint pulls compare sha256 against the remote manifest before
   downloading, so resuming doesn't re-fetch a file you already have.
4. Never hardcode a token. Read it from HF_TOKEN (env var / Kaggle Secret /
   local .env file — see below).

One-time setup (VS Code / local machine):

    uv add huggingface_hub python-dotenv
    cp .env.example .env
    # edit .env, fill in HF_TOKEN and HF_REPO_ID

From then on, no export/activate needed for every session — this module
loads .env automatically, and `uv run` manages the venv for you:

    uv run python -m src.hf_sync push-code
    uv run python -m src.hf_sync pull-code --dest /tmp/hf_cache
    uv run python -m src.hf_sync push-checkpoint --file checkpoints/best.pth --name best.pth
    uv run python -m src.hf_sync pull-checkpoint --name latest.pth --dest checkpoints/latest.pth
    uv run python -m src.hf_sync status

(Env vars / Kaggle Secrets still override .env if both are present, so the
Kaggle notebook flow from Cell 1 is unaffected by any of this.)
"""

import os
import io
import json
import time
import queue
import shutil
import hashlib
import zipfile
import argparse
import threading
import subprocess
from pathlib import Path

# Load a local .env file if python-dotenv is installed and one exists,
# so HF_TOKEN / HF_REPO_ID don't need to be exported by hand every session.
# Real env vars (and Kaggle Secrets, set explicitly in the notebook) always
# take priority over .env — this only fills in what's NOT already set.
try:
    from dotenv import load_dotenv
    _project_root_for_env = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_project_root_for_env, ".env"), override=False)
except ImportError:
    pass

DEFAULT_CODE_INCLUDE = ("src", "tests", "experiments", "requirements.txt", "pyproject.toml", "train.py")
CODE_REMOTE_DIR = "code"
CHECKPOINT_REMOTE_DIR = "checkpoints"
CODE_ZIP_NAME = "spatial_moe_sod_code.zip"
CODE_MANIFEST_NAME = "project_manifest.json"
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def sha256_file(path, chunk_size=1 << 20):
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


def compute_dir_hash(root_dir, include=DEFAULT_CODE_INCLUDE):
    """Deterministic SHA256 over (relative_path, file_bytes) for the tracked
    project files. This is the ONE function used by both:
      - the VS Code push (hashes what's about to be zipped)
      - the Kaggle notebook verify step (hashes what actually landed on disk)
    so a mismatch here means the deployed code is genuinely not what you
    think it is, not a naming mismatch between two different hash functions.
    """
    hasher = hashlib.sha256()
    file_list = []
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


def _git_commit(root_dir):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root_dir,
            capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _get_token(explicit=None):
    token = explicit or os.environ.get("HF_TOKEN")
    if not token:
        # Kaggle Secrets fallback, so notebooks never need a plaintext token.
        try:
            from kaggle_secrets import UserSecretsClient
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


def _get_repo_id(explicit=None):
    repo_id = explicit or os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise RuntimeError("Set HF_REPO_ID, e.g. 'Avi2006/spatial-moe-project'.")
    return repo_id


def get_api(token=None):
    from huggingface_hub import HfApi
    return HfApi(token=_get_token(token))


def ensure_repo(repo_id=None, token=None, repo_type="dataset", private=True):
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    repo_id = _get_repo_id(repo_id)
    api = get_api(token)
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
    return repo_id


# --------------------------------------------------------------------------- #
# Code push / pull  (replaces the manual Kaggle-Dataset zip+manifest upload)
# --------------------------------------------------------------------------- #

def push_code(project_root=".", repo_id=None, token=None, include=DEFAULT_CODE_INCLUDE,
              source_version=None, work_dir=None):
    """Zip the tracked project files, hash them, and upload zip + manifest
    to the code/ folder of the HF repo. Run this from VS Code whenever you
    want the Kaggle notebook to pick up new code — no more re-uploading a
    Kaggle Dataset by hand."""
    repo_id = ensure_repo(repo_id, token)
    project_root = os.path.abspath(project_root)
    work_dir = work_dir or os.path.join(project_root, ".hf_sync_tmp")
    os.makedirs(work_dir, exist_ok=True)
    zip_path = os.path.join(work_dir, CODE_ZIP_NAME)
    manifest_path = os.path.join(work_dir, CODE_MANIFEST_NAME)

    # Deterministic content hash FIRST, over the same include list we're about
    # to zip. This is what the notebook verifies against after unzipping.
    source_content_sha256 = compute_dir_hash(project_root, include)

    # Build the zip deterministically (sorted paths, fixed order) so re-runs
    # with unchanged content produce byte-identical archives.
    file_list = []
    for item in include:
        full = os.path.join(project_root, item)
        if os.path.isdir(full):
            for base, _, files in os.walk(full):
                for fn in files:
                    fp = os.path.join(base, fn)
                    file_list.append(os.path.relpath(fp, project_root))
        elif os.path.isfile(full):
            file_list.append(item)
    file_list.sort()

    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in file_list:
            zf.write(os.path.join(project_root, rel), rel)

    archive_sha256 = sha256_file(zip_path)

    manifest = {
        "source_version": source_version or "unversioned",
        "git_commit": _git_commit(project_root),
        "creation_time": time.time(),
        "archive_name": CODE_ZIP_NAME,
        "archive_sha256": archive_sha256,
        "source_content_sha256": source_content_sha256,
        "included_paths": list(include),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    api = get_api(token)
    print(f"Uploading {zip_path} ({os.path.getsize(zip_path):,} bytes)...")
    api.upload_file(
        path_or_fileobj=zip_path,
        path_in_repo=f"{CODE_REMOTE_DIR}/{CODE_ZIP_NAME}",
        repo_id=repo_id, repo_type="dataset",
        commit_message=f"push code ({source_content_sha256[:12]})",
    )
    # Manifest goes up last, on purpose.
    api.upload_file(
        path_or_fileobj=manifest_path,
        path_in_repo=f"{CODE_REMOTE_DIR}/{CODE_MANIFEST_NAME}",
        repo_id=repo_id, repo_type="dataset",
        commit_message=f"push manifest ({source_content_sha256[:12]})",
    )
    print(f"Code pushed. source_content_sha256={source_content_sha256}")
    return manifest


def pull_code(dest_dir, repo_id=None, token=None):
    """Download code/spatial_moe_sod_code.zip + manifest from the HF repo
    into dest_dir. Returns (zip_path, manifest_dict)."""
    from huggingface_hub import hf_hub_download
    repo_id = _get_repo_id(repo_id)
    token = _get_token(token)
    os.makedirs(dest_dir, exist_ok=True)

    manifest_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset",
        filename=f"{CODE_REMOTE_DIR}/{CODE_MANIFEST_NAME}",
        local_dir=dest_dir, token=token,
    )
    with open(manifest_path) as f:
        manifest = json.load(f)

    zip_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset",
        filename=f"{CODE_REMOTE_DIR}/{CODE_ZIP_NAME}",
        local_dir=dest_dir, token=token,
    )
    actual = sha256_file(zip_path)
    if manifest.get("archive_sha256") and actual != manifest["archive_sha256"]:
        raise RuntimeError(
            f"Downloaded zip sha256 mismatch! expected={manifest['archive_sha256']} actual={actual}. "
            "Download was corrupted or truncated — re-run pull_code."
        )
    return zip_path, manifest


def deploy_code(project_root, dest_dir=None, repo_id=None, token=None,
                 include=DEFAULT_CODE_INCLUDE):
    """Pull + unzip + verify in one call. This is what the Kaggle notebook's
    '05_project_deploy' cell should call instead of reading from
    /kaggle/input/<dataset>."""
    dest_dir = dest_dir or os.path.join(os.path.dirname(project_root) or ".", "hf_cache")
    zip_path, manifest = pull_code(dest_dir, repo_id, token)

    if os.path.exists(project_root):
        shutil.rmtree(project_root)
    os.makedirs(project_root, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(project_root)

    expected = manifest.get("source_content_sha256")
    if not expected:
        raise RuntimeError(
            "project_manifest.json has no source_content_sha256 — re-push "
            "the code with the current hf_sync.push_code (old manifests "
            "from the Kaggle-Dataset era won't have this field)."
        )
    actual = compute_dir_hash(project_root, include)
    if actual != expected:
        raise RuntimeError(
            f"Deployed code does not match manifest!\n"
            f"  expected source_content_sha256={expected}\n"
            f"  actual   source_content_sha256={actual}\n"
            "The code on the Hub does not match what was just unzipped — "
            "push again from VS Code with hf_sync.push_code."
        )
    print(f"Code deployed to {project_root} and verified (sha256 matches).")
    return manifest


# --------------------------------------------------------------------------- #
# Checkpoint push / pull  (replaces the Google Drive / gdown flow)
# --------------------------------------------------------------------------- #

def _fetch_remote_checkpoint_manifest(repo_id, token):
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


def push_checkpoint(local_path, name=None, repo_id=None, token=None, extra_meta=None):
    """Upload one checkpoint file (best.pth / latest.pth / whatever) and
    update the shared checkpoint manifest. Manifest goes up last."""
    repo_id = ensure_repo(repo_id, token)
    token = _get_token(token)
    name = name or os.path.basename(local_path)
    sha = sha256_file(local_path)
    size = os.path.getsize(local_path)

    api = get_api(token)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{CHECKPOINT_REMOTE_DIR}/{name}",
        repo_id=repo_id, repo_type="dataset",
        commit_message=f"push {name} ({sha[:12]})",
    )

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


def pull_checkpoint(name, local_path, repo_id=None, token=None, force=False):
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
            print(f"{name}: local copy already matches remote (sha256 match), skipping download.")
            return False

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=f"{CHECKPOINT_REMOTE_DIR}/{name}",
            token=token,
        )
    except EntryNotFoundError:
        print(f"{name}: not found in {repo_id} — nothing to pull yet.")
        return False

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    shutil.copy2(downloaded, local_path)

    if remote_meta and remote_meta.get("sha256"):
        actual = sha256_file(local_path)
        if actual != remote_meta["sha256"]:
            raise RuntimeError(
                f"{name}: downloaded sha256 {actual} != manifest sha256 {remote_meta['sha256']}"
            )
    print(f"{name}: pulled to {local_path}.")
    return True


def push_json(local_path, name=None, repo_id=None, token=None):
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

    def __init__(self, repo_id=None, token=None):
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
                print(f"[hf_sync] background push OK: {kind} {args[0] if args else ''}")
            except Exception as e:
                # Never crash training because the upload failed — the local
                # checkpoint on disk is still safe. Log and move on; the
                # next checkpoint push will naturally re-sync a newer file.
                print(f"[hf_sync] WARNING: background push failed ({kind} {args[0] if args else ''}): {e}")
            finally:
                self._q.task_done()

    def enqueue_checkpoint(self, local_path, name=None, extra_meta=None):
        self._q.put(("checkpoint", (local_path,), {"name": name, "extra_meta": extra_meta}))

    def enqueue_json(self, local_path, name=None):
        self._q.put(("json", (local_path,), {"name": name}))

    def flush(self, timeout=600):
        """Block until all queued pushes complete, or timeout seconds pass.
        Always call this before the process exits."""
        start = time.time()
        while not self._q.empty() and (time.time() - start) < timeout:
            time.sleep(1)
        if not self._q.empty():
            print(f"[hf_sync] WARNING: flush timed out after {timeout}s with pushes still pending.")

    def close(self, timeout=600):
        self.flush(timeout)
        self._q.put(None)
        self._thread.join(timeout=30)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cli():
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
