import os

import cv2
import numpy as np
import pytest
import albumentations as A
from torch.utils.data.distributed import DistributedSampler

from src.dataset import (
    WXSODDataset,
    build_transforms,
    get_dataloaders,
    get_scene_id,
    get_weather_type,
    scene_group_split,
)


def _write_pair(img_dir, gt_dir, stem, h=32, w=24):
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % (2**32))
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255

    cv2.imwrite(str(img_dir / f"{stem}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(gt_dir / f"{stem}.jpg"), gt)


@pytest.fixture
def synth_wxsod(tmp_path):
    root = tmp_path / "wxsod_synth"
    weathers = ["fog", "rain", "snow"]

    # train_sys: 6 scenes x 3 weathers (every scene has every weather).
    for i in range(6):
        for w in weathers:
            _write_pair(
                root / "train_sys" / "input",
                root / "train_sys" / "gt",
                f"{i:04d}_{w}",
                h=24 + i,
                w=32,
            )

    # test_sys / test_real: a couple of pairs with varied aspect ratios.
    for split in ["test_sys", "test_real"]:
        _write_pair(root / split / "input", root / split / "gt", f"0099_{weathers[0]}", h=40, w=16)
        _write_pair(root / split / "input", root / split / "gt", f"0099_{weathers[1]}", h=16, w=40)

    return root


def test_build_transforms_composition():
    train_t = build_transforms("train", 384)
    assert any(isinstance(t, A.HorizontalFlip) for t in train_t.transforms)
    assert any(isinstance(t, A.ColorJitter) for t in train_t.transforms)

    val_t = build_transforms("val", 384)
    assert len(val_t.transforms) == 1
    assert isinstance(val_t.transforms[0], A.Normalize)


def test_build_transforms_normalizes():
    pipe = build_transforms("val", 384)
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    gt = (np.random.default_rng(0).random((16, 16)) > 0.5).astype(np.uint8)
    res = pipe(image=img, masks=[gt, np.ones_like(gt)])

    out = res["image"]
    assert out.dtype != np.uint8
    assert out.shape == (16, 16, 3)
    # Normalized: the constant 128/255 input must differ from the raw range.
    assert np.abs(out - 0.5).max() > 0.1
    assert res["masks"][0].shape == (16, 16)


def test_scene_id_and_weather_extraction():
    assert get_scene_id("0001_fog") == "0001"
    assert get_scene_id("0002_rainafog") == "0002"
    assert get_weather_type("0001_rainafog") == "rainafog"
    with pytest.raises(ValueError):
        get_scene_id("plain")


def test_scene_group_split_disjoint_and_coverage():
    samples = [
        (f"{i:04d}_{w}.jpg", f"{i:04d}_{w}.png", f"{i:04d}_{w}")
        for i in range(6)
        for w in ("fog", "rain", "snow")
    ]

    train, val = scene_group_split(samples)

    assert len(train) + len(val) == len(samples)
    assert 0 < len(val) < len(train)  # a minority of the 6 scenes, all grouped

    train_scenes = {get_scene_id(s[2]) for s in train}
    val_scenes = {get_scene_id(s[2]) for s in val}
    assert train_scenes.isdisjoint(val_scenes)

    train_weather = {get_weather_type(s[2]) for s in train}
    val_weather = {get_weather_type(s[2]) for s in val}
    assert train_weather == {"fog", "rain", "snow"}
    assert val_weather == {"fog", "rain", "snow"}


def test_scene_group_split_rejects_unparsable_stems():
    bad = [("a.jpg", "a.png", "no_underscore")]
    with pytest.raises(ValueError):
        scene_group_split(bad)


def test_wxsod_dataset_item(tmp_path):
    root = tmp_path / "one"
    _write_pair(root / "test_sys" / "input", root / "test_sys" / "gt", "0007_light", h=24, w=32)

    ds = WXSODDataset(root_dir=str(root), split="test_sys", image_size=16)
    assert len(ds) == 1
    assert ds[0]["name"] == "0007_light"

    item = ds[0]
    img, mask, edge, pad = item["image"], item["mask"], item["edge_map"], item["pad_mask"]
    assert img.shape == (3, 16, 16)
    assert mask.shape == (1, 16, 16)
    assert edge.shape == (1, 16, 16)
    assert pad.shape == (1, 16, 16)

    assert 0.0 <= mask.min() and mask.max() <= 1.0
    assert len(torch_unique(edge)) <= 2  # binary boundary map
    # Padding regions are zero in mask and pad_mask; the 12x16 content is 1.
    assert mask[0, :2, :].sum() == 0
    assert pad[0, :2, :].sum() == 0
    assert pad[0, 2:14, :].sum() == 12 * 16


def torch_unique(t):
    return t.unique()


def test_wxsod_dataset_rejects_unpaired_gt(tmp_path):
    root = tmp_path / "broken"
    in_dir = root / "train_sys" / "input"
    gt_dir = root / "train_sys" / "gt"
    in_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    _write_pair(in_dir, gt_dir, "0001_fog")
    # An extra input image with no GT -> the 1:1 mapping check must fail.
    _write_pair(in_dir, gt_dir, "0001_rain")
    os.remove(gt_dir / "0001_rain.jpg")

    with pytest.raises(RuntimeError):
        WXSODDataset(root_dir=str(root), split="train", image_size=16)


def test_get_dataloaders_synthetic_samplers_and_splits(synth_wxsod):
    train_loader, val_loader, synth_loader, real_loader = get_dataloaders(
        root_dir=str(synth_wxsod),
        image_size=32,
        batch_size=4,
        num_workers=0,
        distributed=True,
        rank=0,
        world_size=2,
    )

    for loader in [train_loader, val_loader, synth_loader, real_loader]:
        assert isinstance(loader.sampler, DistributedSampler)

    train_ds, val_ds = train_loader.dataset, val_loader.dataset
    train_scenes = {get_scene_id(s[2]) for s in train_ds.samples}
    val_scenes = {get_scene_id(s[2]) for s in val_ds.samples}
    assert train_scenes.isdisjoint(val_scenes)

    # A full batch from each loader can be produced.
    for loader in [train_loader, val_loader, synth_loader, real_loader]:
        batch = next(iter(loader))
        assert batch["image"].shape[1:] == (3, 32, 32)
        assert batch["mask"].shape[1:] == (1, 32, 32)


def test_get_dataloaders_non_distributed_has_no_samplers(synth_wxsod):
    train_loader, val_loader, synth_loader, real_loader = get_dataloaders(
        root_dir=str(synth_wxsod),
        image_size=32,
        batch_size=4,
        num_workers=0,
        distributed=False,
    )
    for loader in [train_loader, val_loader, synth_loader, real_loader]:
        assert not isinstance(loader.sampler, DistributedSampler)