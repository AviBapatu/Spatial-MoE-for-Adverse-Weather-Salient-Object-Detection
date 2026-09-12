"""Code push / pull / deploy against the ``code/`` folder of the HF dataset repo.

Replaces the manual Kaggle-Dataset zip + manifest upload flow. The
``source_content_sha256`` it writes is the same ``compute_dir_hash`` output the
Kaggle notebook verifies after unzipping, so a mismatch always means the
deployed code genuinely differs from what was uploaded.
"""

import json
import os
import shutil
import subprocess
import time
import zipfile
from typing import Optional

from src.hf_sync.hashing import (
    DEFAULT_CODE_INCLUDE,
    _get_repo_id,
    _get_token,
    compute_dir_hash,
    ensure_repo,
    get_api,
    sha256_file,
)

CODE_REMOTE_DIR = "code"
CODE_ZIP_NAME = "spatial_moe_sod_code.zip"
CODE_MANIFEST_NAME = "project_manifest.json"


def _git_commit(root_dir: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root_dir,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


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


def pull_code(dest_dir: str, repo_id: Optional[str] = None, token: Optional[str] = None):
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


def deploy_code(project_root: str, dest_dir: Optional[str] = None,
                repo_id: Optional[str] = None, token: Optional[str] = None,
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
