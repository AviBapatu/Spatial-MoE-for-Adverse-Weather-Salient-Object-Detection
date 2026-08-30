import zipfile
import os
import json
import hashlib
import time
import subprocess

def package_project():
    output_filename = "spatial_moe_sod_code.zip"
    manifest_filename = "project_manifest.json"
    
    dirs_to_zip = ["src", "tests", "experiments"]
    files_to_zip = ["requirements.txt", "pyproject.toml", "train.py"]
    
    # 1. Create the zip
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for d in dirs_to_zip:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    if "__pycache__" in root: continue
                    for file in files:
                        if file.endswith(".pyc"): continue
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, file_path)
                        
        for f in files_to_zip:
            if os.path.exists(f):
                zipf.write(f, f)
                
    # 2. Compute SHA256 of the zip
    sha256 = hashlib.sha256()
    with open(output_filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    zip_hash = sha256.hexdigest()
    
    # 3. Get git commit if possible
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        pass
        
    # 4. Write manifest
    manifest = {
        "source_version": "1.0.0",
        "git_commit": git_commit,
        "creation_time": time.time(),
        "archive_sha256": zip_hash,
        "archive_name": output_filename
    }
    
    with open(manifest_filename, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Project packaged into {output_filename}")
    print(f"Manifest written to {manifest_filename}")
    print(f"SHA256: {zip_hash}")

if __name__ == "__main__":
    package_project()
