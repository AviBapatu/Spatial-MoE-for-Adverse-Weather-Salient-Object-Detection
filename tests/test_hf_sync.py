"""Tests for ``src.hf_sync``: hash determinism, atomic write + concurrent-reader
safety, checkpoint integrity verification, and the HF upload path with a mocked
Hub API (no network calls).

The snapshot-race test is the important one: it proves ``push_checkpoint``
hashes and uploads the *same* bytes even when training atomically replaces the
live checkpoint path mid-push.
"""

import hashlib
import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from src import hf_sync
from src.hf_sync import checkpoint_sync
from src.hf_sync.checkpoint_sync import (
    AsyncCheckpointPusher,
    push_checkpoint,
    push_json,
    verify_checkpoint_integrity,
)
from src.hf_sync.hashing import compute_dir_hash, sha256_file


# --------------------------------------------------------------------------- #
# Hash determinism
# --------------------------------------------------------------------------- #

def test_sha256_file_matches_hashlib(tmp_path):
    data = b"spatial-moe checkpoint payload" * 1000
    p = tmp_path / "blob.bin"
    p.write_bytes(data)

    # Hand-computed oracle + stability across repeated reads (incl. >1 chunk).
    big = tmp_path / "big.bin"
    big.write_bytes(data * 64)  # ~2 MiB, forces multi-chunk reads

    assert sha256_file(str(p)) == hashlib.sha256(data).hexdigest()
    assert sha256_file(str(p)) == sha256_file(str(p))
    assert sha256_file(str(big)) == hashlib.sha256(data * 64).hexdigest()


def test_compute_dir_hash_deterministic_and_sensitive(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "experiments").mkdir()
    (tmp_path / "src").joinpath("model.py").write_text("class Model: pass\n")
    (tmp_path / "experiments").joinpath("baseline.json").write_text('{"x": 1}')
    (tmp_path).joinpath("train.py").write_text("print('hi')\n")

    include = ("src", "experiments", "train.py")

    h1 = compute_dir_hash(str(tmp_path), include)
    h2 = compute_dir_hash(str(tmp_path), include)
    assert h1 == h2

    # Content change (not just path change) must change the hash.
    (tmp_path / "src").joinpath("model.py").write_text("class Model:\n    pass\n")
    assert compute_dir_hash(str(tmp_path), include) != h1


def test_compute_dir_hash_matches_manual_fold(tmp_path):
    files = {
        "src/a.py": "a = 1",
        "src/deep/b.py": "b = 2",
        "readme.txt": "readme",
    }
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)

    include = ("src", "readme.txt")
    assert compute_dir_hash(str(tmp_path), include) == _manual_dir_hash(str(tmp_path), include)


def _manual_dir_hash(root, include):
    """Independent reproduction of compute_dir_hash's folding."""
    hasher = hashlib.sha256()
    file_list = []
    for item in include:
        full = os.path.join(root, item)
        if os.path.isdir(full):
            for base, _, files in os.walk(full):
                for fn in files:
                    fp = os.path.join(base, fn)
                    file_list.append((os.path.relpath(fp, root), fp))
        elif os.path.isfile(full):
            file_list.append((item, full))
    file_list.sort(key=lambda x: x[0])
    for rel, fp in file_list:
        hasher.update(rel.replace(os.sep, "/").encode("utf-8"))
        with open(fp, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


# --------------------------------------------------------------------------- #
# verify_checkpoint_integrity
# --------------------------------------------------------------------------- #

def _write_valid_checkpoint(path, tag):
    torch.save({"tag": tag, "x": torch.ones(4)}, str(path))


def test_verify_checkpoint_integrity_states(tmp_path):
    good = tmp_path / "best.pth"
    _write_valid_checkpoint(good, "good")
    sha = sha256_file(str(good))

    assert verify_checkpoint_integrity(str(good)) is True
    assert verify_checkpoint_integrity(str(good), expected_sha256=sha) is True

    # Wrong expected sha -> False.
    assert verify_checkpoint_integrity(str(good), expected_sha256="0" * 64) is False

    # Missing file -> False.
    assert verify_checkpoint_integrity(str(tmp_path / "nope.pth")) is False

    # Garbage bytes (not a torch checkpoint) -> False.
    garbage = tmp_path / "garbage.pth"
    garbage.write_bytes(b"definitely not a torch.load pickled checkpoint")
    assert verify_checkpoint_integrity(str(garbage)) is False

    # Truncated valid checkpoint -> False.
    good_bytes = good.read_bytes()
    truncated = tmp_path / "trunc.pth"
    truncated.write_bytes(good_bytes[: len(good_bytes) // 2])
    assert verify_checkpoint_integrity(str(truncated)) is False


def test_concurrent_atomic_write_verified_by_reader(tmp_path):
    """Writer uses the repo's tmp -> os.replace pattern; reader (background
    thread) must never observe a torn file via sha256/verify."""
    path = str(tmp_path / "latest.pth")
    n_writes = 40
    recorded_shas = []  # sha of each complete state, recorded by the writer
    observed_shas = []
    stop = threading.Event()

    def writer():
        for i in range(n_writes):
            torch.save({"i": i, "payload": torch.randn(8, 8)}, path + ".tmp")
            os.replace(path + ".tmp", path)
            recorded_shas.append(sha256_file(path))
        stop.set()

    def reader():
        while not stop.is_set():
            if os.path.exists(path):
                observed_shas.append(sha256_file(path))
                assert verify_checkpoint_integrity(path) is True
            time.sleep(0.001)

    t_w = threading.Thread(target=writer)
    t_r = threading.Thread(target=reader)
    t_w.start()
    t_r.start()
    t_w.join()
    t_r.join()

    assert observed_shas, "reader never saw the file"
    for sha in observed_shas:
        assert sha in recorded_shas, f"reader observed a torn/partial file state: {sha}"


# --------------------------------------------------------------------------- #
# Mocked HF API
# --------------------------------------------------------------------------- #

class FakeApi:
    """Records uploads in memory; reads file-path uploads at call time."""

    def __init__(self):
        self.uploads = []  # list of {'path_in_repo', 'content', 'commit_message'}

    def upload_file(self, path_or_fileobj, path_in_repo=None, repo_id=None,
                    repo_type=None, commit_message=None):
        if isinstance(path_or_fileobj, str):
            with open(path_or_fileobj, "rb") as f:
                content = f.read()
        else:
            content = path_or_fileobj
        self.uploads.append({
            "path_in_repo": path_in_repo,
            "content": content,
            "commit_message": commit_message,
        })

    def repo_info(self, **kwargs):
        return SimpleNamespace()


@pytest.fixture()
def fake_hub(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(checkpoint_sync, "ensure_repo", lambda *a, **k: "dummy/repo")
    monkeypatch.setattr(checkpoint_sync, "get_api", lambda *a, **k: api)
    monkeypatch.setattr(
        checkpoint_sync,
        "_fetch_remote_checkpoint_manifest",
        lambda *a, **k: {},  # empty remote manifest -> manifest starts fresh
    )
    return api


def test_push_checkpoint_snapshot_race(fake_hub, tmp_path):
    """Overwriting the live path mid-push must not corrupt the manifest sha."""
    local = tmp_path / "best.pth"
    state_v1 = torch.randn(16)
    torch.save({"payload": state_v1}, str(local))
    sha_v1 = sha256_file(str(local))
    overwrote = threading.Event()

    original_upload = fake_hub.upload_file

    def racing_upload(path_or_fileobj, path_in_repo=None, **kwargs):
        # The live file is atomically replaced between our hash and upload:
        if path_in_repo == "checkpoints/best.pth":
            torch.save({"payload": torch.zeros(16)}, str(local))  # overwrite
            overwrote.set()
        original_upload(path_or_fileobj, path_in_repo=path_in_repo, **kwargs)

    fake_hub.upload_file = racing_upload

    pushed_sha = push_checkpoint(str(local), name="best.pth")

    # The sha returned matches the bytes that were actually uploaded (v1),
    # NOT the bytes now sitting at the live path.
    ckpt_upload = next(u for u in fake_hub.uploads if u["path_in_repo"] == "checkpoints/best.pth")
    manifest_upload = next(u for u in fake_hub.uploads if u["path_in_repo"] == "checkpoints/checkpoint_manifest.json")
    assert overwrote.is_set()
    assert hashlib.sha256(ckpt_upload["content"]).hexdigest() == sha_v1
    assert pushed_sha == sha_v1
    assert hashlib.sha256(ckpt_upload["content"]).hexdigest() != sha256_file(str(local))
    # Manifest sha points at the uploaded (v1) bytes.
    import json

    manifest = json.loads(manifest_upload["content"].decode())
    assert manifest["files"]["best.pth"]["sha256"] == sha_v1
    # No snapshot temp files left behind.
    assert not [f for f in os.listdir(str(tmp_path)) if ".snapshot.pth" in f]


def test_push_checkpoint_manifest_last_and_cleaned(fake_hub, tmp_path):
    local = tmp_path / "latest.pth"
    torch.save({"t": 1}, str(local))

    pushed_sha = push_checkpoint(str(local), name="latest.pth", extra_meta={"epoch": 7})

    remote_names = [u["path_in_repo"] for u in fake_hub.uploads]
    assert remote_names == ["checkpoints/latest.pth", "checkpoints/checkpoint_manifest.json"]
    assert pushed_sha == sha256_file(str(local))
    # No temp files (snapshot or manifest) left behind.
    leftovers = [f for f in os.listdir(str(tmp_path)) if f != "latest.pth"]
    assert leftovers == []


def test_push_checkpoint_verify_true_valid(fake_hub, tmp_path):
    local = tmp_path / "verified.pth"
    torch.save({"ok": 1}, str(local))
    push_checkpoint(str(local), name="verified.pth", verify=True)
    assert [u["path_in_repo"] for u in fake_hub.uploads] == [
        "checkpoints/verified.pth",
        "checkpoints/checkpoint_manifest.json",
    ]


def test_push_checkpoint_verify_refuses_corrupt(fake_hub, tmp_path):
    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"not a torch checkpoint")

    with pytest.raises(RuntimeError, match="integrity check failed"):
        push_checkpoint(str(corrupt), name="corrupt.pth", verify=True)

    # Refused before any upload.
    assert fake_hub.uploads == []


def test_async_pusher_threads_verify_flag(fake_hub, tmp_path):
    good = tmp_path / "best.pth"
    corrupt = tmp_path / "corrupt.pth"
    torch.save({"ok": 1}, str(good))
    corrupt.write_bytes(b"garbage")

    pusher = AsyncCheckpointPusher(repo_id="dummy/repo", token="dummy")
    pusher.enqueue_checkpoint(str(good), name="best.pth", verify=True)
    pusher.enqueue_checkpoint(str(corrupt), name="corrupt.pth", verify=True)
    pusher.enqueue_json(str(good), name="training_complete.json")
    pusher.flush(timeout=60)
    pusher.close(timeout=60)

    remote_names = [u["path_in_repo"] for u in fake_hub.uploads]
    # Good checkpoint pushed (file + manifest), corrupt one refused (only its
    # warning), json marker pushed.
    assert "checkpoints/best.pth" in remote_names
    assert "checkpoints/corrupt.pth" not in remote_names
    assert "checkpoints/training_complete.json" in remote_names
    assert "checkpoints/checkpoint_manifest.json" in remote_names


def test_backward_compat_surface():
    """Attributes the rest of the repo relies on must still exist."""
    assert hasattr(hf_sync, "AsyncCheckpointPusher")
    assert hasattr(hf_sync, "push_checkpoint")
    assert hasattr(hf_sync, "push_json")
    assert hasattr(hf_sync, "push_code")
    assert hasattr(hf_sync, "pull_checkpoint")
    assert hasattr(hf_sync, "CODE_ZIP_NAME")
    assert hasattr(hf_sync, "CODE_MANIFEST_NAME")