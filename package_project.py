"""
package_project.py — Build the code zip and push it to HuggingFace.

Always delegates to hf_sync.push_code, which computes the content hash and
builds the zip in a single pass from the same file list, guaranteeing
source_content_sha256 in project_manifest.json always matches the zip.

Usage:
    python package_project.py               # zip + upload (default)
    python package_project.py --no-push     # zip only, no upload
    python package_project.py --version v1.2

Can be run with bare `python` — it will re-exec itself under `uv run` if the
project venv is not active.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess

# ── Self-re-exec under uv if huggingface_hub is not available ────────────────
# This lets you run `python package_project.py` directly without first
# activating the venv or prefixing every command with `uv run`.
try:
    import huggingface_hub  # noqa: F401
except ImportError:
    # Not in the project venv — re-launch under uv and exit.
    script = os.path.abspath(__file__)
    print("[package_project] huggingface_hub not found — re-launching under uv run...\n")
    result = subprocess.run(["uv", "run", "python", script] + sys.argv[1:])
    sys.exit(result.returncode)

# ── Canonical include list ───────────────────────────────────────────────────
# Only files needed to train on Kaggle.
# Must stay in sync with hf_sync.DEFAULT_CODE_INCLUDE so the Kaggle notebook's
# hash-verify step (Cell 12) always agrees with what was uploaded.
INCLUDE = (
    "src",           # all model / training / eval source
    "experiments",   # experiment config JSONs (baseline_v1.json etc.)
    "requirements.txt",
    "pyproject.toml",
    "train.py",
)


def main():
    parser = argparse.ArgumentParser(description="Package project and upload to HuggingFace.")
    parser.add_argument("--no-push", action="store_true",
                        help="Build zip locally only — skip HuggingFace upload.")
    parser.add_argument("--version", type=str, default=None,
                        help="Version tag written into the manifest, e.g. v1.2")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))

    # hf_sync.push_code is the single source of truth: it computes the content
    # hash and builds the zip in one pass from the same include list, so the
    # manifest it writes can never drift from the zip it creates.
    from src.hf_sync import push_code, CODE_ZIP_NAME, CODE_MANIFEST_NAME

    work_dir = os.path.join(root, ".hf_sync_tmp")

    print("=== Packaging project ===")
    print(f"Root    : {root}")
    print(f"Include : {', '.join(INCLUDE)}\n")

    manifest = push_code(
        project_root=root,
        include=INCLUDE,
        source_version=args.version or "unversioned",
        work_dir=work_dir,
        # push_code uploads by default; if --no-push we monkey-patch after
    ) if not args.no_push else _build_only(root, work_dir, args.version)

    # Copy the canonical zip + manifest to project root for local inspection
    for name, dest_name in [
        (CODE_ZIP_NAME,      "spatial_moe_sod_code.zip"),
        (CODE_MANIFEST_NAME, "project_manifest.json"),
    ]:
        src_path = os.path.join(work_dir, name)
        dst_path = os.path.join(root, dest_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

    # Print summary from the manifest hf_sync generated
    with open(os.path.join(root, "project_manifest.json")) as f:
        m = json.load(f)

    print(f"\nProject packaged into spatial_moe_sod_code.zip")
    print(f"Manifest written to   project_manifest.json")
    print(f"SHA256 (archive):     {m['archive_sha256']}")
    print(f"SHA256 (source):      {m['source_content_sha256']}")
    if not args.no_push:
        print(f"\n✅ Uploaded to HuggingFace ({os.environ.get('HF_REPO_ID', 'see .env')})")
    else:
        print("\n⚠️  --no-push: skipped HuggingFace upload.")


def _build_only(root, work_dir, version):
    """Build zip + manifest without uploading, using hf_sync internals."""
    import time, zipfile, json
    from src.hf_sync import (
        compute_dir_hash, sha256_file, _git_commit,
        CODE_ZIP_NAME, CODE_MANIFEST_NAME,
    )

    os.makedirs(work_dir, exist_ok=True)
    zip_path = os.path.join(work_dir, CODE_ZIP_NAME)
    manifest_path = os.path.join(work_dir, CODE_MANIFEST_NAME)

    source_hash = compute_dir_hash(root, INCLUDE)

    # Build zip using the same walk as compute_dir_hash (no extra filtering)
    file_list = []
    for item in INCLUDE:
        full = os.path.join(root, item)
        if os.path.isdir(full):
            for base, _, files in os.walk(full):
                for fn in files:
                    fp = os.path.join(base, fn)
                    file_list.append(os.path.relpath(fp, root))
        elif os.path.isfile(full):
            file_list.append(item)
    file_list.sort()

    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in file_list:
            zf.write(os.path.join(root, rel), rel)

    archive_hash = sha256_file(zip_path)
    manifest = {
        "source_version": version or "unversioned",
        "git_commit": _git_commit(root),
        "creation_time": time.time(),
        "archive_name": CODE_ZIP_NAME,
        "archive_sha256": archive_hash,
        "source_content_sha256": source_hash,
        "included_paths": list(INCLUDE),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    main()
