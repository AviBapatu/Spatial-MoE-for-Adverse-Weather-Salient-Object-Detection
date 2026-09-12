"""WXSOD dataset loading and scene-aware splitting.

Provides:

- :class:`WXSODDataset`: per-image reader that applies aspect-preserving
  resize/pad, computes boundary targets, and applies the albumentations
  ``build_transforms`` pipeline.
- :func:`build_transforms`: the single source of truth for the train/val/test
  augmentation pipelines.
- :func:`scene_group_split`: scene-aware ``GroupShuffleSplit`` with a hard
  train/val scene-disjointness assertion (data-leakage prevention —
  correctness-critical, do not weaken).
- :func:`get_dataloaders`: builds the four loaders (train / val / test_sys /
  test_real) with rank-aware ``DistributedSampler`` support.

``cv2.setNumThreads(0)`` is intentional: it stops OpenCV's internal thread pool
from competing with DataLoader worker processes.
"""
import glob
import hashlib
import os
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

cv2.setNumThreads(0)  # Prevent OpenCV thread pool from competing with DataLoader workers
import albumentations as A
import numpy as np
import torch
from albumentations import Compose
from albumentations.pytorch import ToTensorV2  # noqa: F401  (kept for parity with the original import)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.boundary import compute_boundary, get_boundary_kernel
from src.log import get_logger

log = get_logger(__name__)


def build_transforms(split: str, image_size: int = 384) -> Compose:
    """Build the albumentations pipeline for a dataset split.

    Train gets horizontal flip and color jitter before normalization; every other
    split gets normalization only.  The aspect-preserving resize/pad geometry is
    applied separately in :meth:`WXSODDataset.__getitem__`; ``image_size`` is kept
    on the signature so the split pipelines stay consistent with the dataset's
    configured output resolution, and so ``build_transforms`` remains the sole
    place that defines what differs between splits.

    Args:
        split: One of ``"train"`` (augmented) or ``"val"``/``"test_sys"``/
            ``"test_real"`` (inference-style).
        image_size: Output resolution the dataset resizes to (reserved for
            pipeline/dataset consistency; the current transform list does not
            resize).

    Returns:
        The ``albumentations.Compose`` to apply to ``image`` and masks.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if split == "train":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=mean, std=std),
        ]
    else:
        transforms = [
            A.Normalize(mean=mean, std=std),
        ]
    return A.Compose(transforms, is_check_shapes=False)


class WXSODDataset(Dataset):
    """A single-scene-ready image/mask dataset with boundary and pad-mask outputs.

    Args:
        root_dir: Directory containing the ``train_sys``/``test_sys``/``test_real``
            subfolders.
        split: ``"train"``, ``"val"``, ``"test_sys"`` or ``"test_real"``.
        image_size: Output spatial size (aspect-preserving resize + pad).
        max_samples: Optional truncation of the sample list.
        transform: Optional override; defaults to :func:`build_transforms`.
        base_seed: Anchor for the per-sample deterministic augmentation seed.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 384,
        max_samples: Optional[int] = None,
        transform: Optional[Compose] = None,
        base_seed: int = 42,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.max_samples = max_samples
        self.transform = transform
        self.base_seed = base_seed
        self.epoch = 0

        # Determine paths based on split
        if split in ["train", "val"]:
            self.input_dir = os.path.join(root_dir, "train_sys", "input")
            self.gt_dir = os.path.join(root_dir, "train_sys", "gt")
        elif split == "test_sys":
            self.input_dir = os.path.join(root_dir, "test_sys", "input")
            self.gt_dir = os.path.join(root_dir, "test_sys", "gt")
        elif split == "test_real":
            self.input_dir = os.path.join(root_dir, "test_real", "input")
            self.gt_dir = os.path.join(root_dir, "test_real", "gt")
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"Missing input directory: {self.input_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"Missing GT directory: {self.gt_dir}")

        # Get all input images
        input_pattern = os.path.join(self.input_dir, "*")
        self.image_paths = sorted([p for p in glob.glob(input_pattern) if os.path.isfile(p)])

        # Create a mapping of stem (basename without extension) to gt path
        gt_pattern = os.path.join(self.gt_dir, "*")
        gt_paths = [p for p in glob.glob(gt_pattern) if os.path.isfile(p)]
        self.gt_map = {Path(p).stem: p for p in gt_paths}

        # Filter valid pairs and check for strict 1:1 mapping
        self.samples: List[Tuple[str, str, str]] = []
        input_stems = [Path(p).stem for p in self.image_paths]

        # Validation checks
        if len(set(input_stems)) != len(input_stems):
            raise RuntimeError("Duplicate stems found in input images.")

        for img_path in self.image_paths:
            stem = Path(img_path).stem
            if stem not in self.gt_map:
                raise RuntimeError(f"Input image {img_path} has no corresponding GT mask.")
            self.samples.append((img_path, self.gt_map[stem], stem))

        for gt_stem in self.gt_map.keys():
            if gt_stem not in set(input_stems):
                raise RuntimeError(f"GT mask {self.gt_map[gt_stem]} has no corresponding input image.")

        # Setup albumentations transforms if none provided
        if self.transform is None:
            self.transform = build_transforms(split, self.image_size)

        # Morphological parameters for boundary target
        self.boundary_kernel = get_boundary_kernel()

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch (used for deterministic per-epoch seeds)."""
        self.epoch = epoch

    def set_samples(self, samples: List[Tuple[str, str, str]]) -> None:
        """Override the sample list, optionally truncated to ``max_samples``."""
        self.samples = samples
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _compute_edge_map(self, mask_binary: np.ndarray) -> np.ndarray:
        """Compute the boundary map of a binary mask."""
        # Ensure it is purely binary (0 or 1)
        mask_u8 = (mask_binary > 0.5).astype(np.uint8)
        return compute_boundary(mask_u8, self.boundary_kernel)

    def _aspect_preserving_resize_pad(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Resize to the longest-side ``image_size`` then centre-pad to square."""
        orig_h, orig_w = image.shape[:2]

        # Longest max size
        scale = self.image_size / max(orig_h, orig_w)
        resized_h = int(orig_h * scale)
        resized_w = int(orig_w * scale)

        # Resize
        image_resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)

        # Pad if needed
        pad_h = self.image_size - resized_h
        pad_w = self.image_size - resized_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Reflect padding for image, constant 0 for mask
        image_padded = cv2.copyMakeBorder(
            image_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        mask_padded = cv2.copyMakeBorder(
            mask_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0]
        )

        valid_mask_ones = np.ones((resized_h, resized_w), dtype=np.float32)
        valid_mask = cv2.copyMakeBorder(
            valid_mask_ones, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0]
        )

        meta = {
            'orig_h': orig_h,
            'orig_w': orig_w,
            'resized_h': resized_h,
            'resized_w': resized_w,
            'pad_top': pad_top,
            'pad_bottom': pad_bottom,
            'pad_left': pad_left,
            'pad_right': pad_right,
        }

        return image_padded, mask_padded, valid_mask, meta

    def __getitem__(self, index: int) -> dict:
        img_path, gt_path, name = self.samples[index]

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {gt_path}")

        # Normalize mask to 0-1 and binarize
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        # 1. Geometric transforms (aspect-preserving resize & pad)
        image_padded, mask_padded, valid_mask, meta = self._aspect_preserving_resize_pad(image, mask)
        meta['gt_path'] = gt_path

        # 2. Compute edge AFTER geometry transforms on the binary mask
        edge_map = self._compute_edge_map(mask_padded)

        # Determine unique deterministic seed for this exact sample and epoch
        sample_seed_str = f"{self.base_seed}_{self.epoch}_{index}_{name}"
        sample_seed = int(hashlib.md5(sample_seed_str.encode('utf-8')).hexdigest()[:8], 16)

        # 3. Apply color/flip augmentations and normalization
        if self.transform is not None:
            # Re-seed purely for this augmentation call
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)

            augmented = self.transform(image=image_padded, masks=[mask_padded, valid_mask])
            img_tensor = augmented['image']
            mask_tensor, valid_mask = augmented['masks']
        else:
            img_tensor = image_padded
            mask_tensor = mask_padded

        # Convert to tensor manually if ToTensorV2 not in transform
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask_tensor).float()

        valid_mask_tensor = torch.from_numpy(valid_mask).float()
        edge_tensor = torch.from_numpy(edge_map).float().unsqueeze(0)

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        if valid_mask_tensor.ndim == 2:
            valid_mask_tensor = valid_mask_tensor.unsqueeze(0)

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'edge_map': edge_tensor,
            'pad_mask': valid_mask_tensor,
            'name': name,
            'meta': meta,
        }


def get_scene_id(stem: str) -> str:
    """Extract the scene ID (first ``_``-separated token) from a sample stem.

    Assumes the format ``{scene_id}_{weather}``, where ``scene_id`` itself
    contains no underscores (everything before the first ``_`` is the scene,
    everything after is the weather suffix — see :func:`get_weather_type`).
    Raises if the stem does not match, so unreliable grouping can never
    silently degrade into a random split.

    Args:
        stem: Image stem, e.g. ``scene12_fog``.

    Returns:
        The scene ID, e.g. ``"scene12"``.

    Raises:
        ValueError: If fewer than two underscore-separated parts exist.
    """
    parts = stem.split('_')
    if len(parts) < 2:
        raise ValueError(f"Stem '{stem}' does not match expected pattern sceneID_weather. Cannot reliably extract scene ID.")
    return parts[0]


def get_weather_type(stem: str) -> str:
    """Extract the weather suffix (everything after the first ``_``) from a stem."""
    parts = stem.split('_')
    if len(parts) >= 2:
        return "_".join(parts[1:])
    return "unknown"


def scene_group_split(
    samples: List[Tuple[str, str, str]], test_size: float = 0.2, random_state: int = 42
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Scene-aware train/val split that guarantees scene disjointness.

    Groups samples by ``{scene_id}`` and uses :class:`sklearn.GroupShuffleSplit`
    so no scene appears in both splits — preventing data leakage between
    semantically duplicate views of the same scene.  A hard assertion enforces
    the disjointness invariant.

    Args:
        samples: ``(img_path, gt_path, stem)`` tuples, each stem of the form
            ``{scene_id}_{weather}``.
        test_size: Validation fraction of scenes.
        random_state: RNG seed for the grouping split.

    Returns:
        ``(train_samples, val_samples)`` — two disjoint scene-grouped lists.

    Raises:
        ValueError: If any stem cannot be resolved to a scene ID (a random-split
            fallback is forbidden deliberately to prevent leakage).
        AssertionError: If the resulting split is not scene-disjoint.
    """
    scene_ids = []
    for _, _, stem in samples:
        try:
            sid = get_scene_id(stem)
            scene_ids.append(sid)
        except ValueError as e:
            log.info("CRITICAL: Failed to reliably establish scene grouping.")
            log.info(e)
            log.info("Aborting dataset creation. A random split fallback is forbidden to prevent data leakage.")
            raise

    # Scene-aware group split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(samples, groups=scene_ids))

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    # Hard assertion for disjointness
    train_scenes_set = set([get_scene_id(s[2]) for s in train_samples])
    val_scenes_set = set([get_scene_id(s[2]) for s in val_samples])
    assert train_scenes_set.isdisjoint(val_scenes_set), (
        "CRITICAL: Data leakage detected! Train and Val scenes are not disjoint."
    )

    return train_samples, val_samples


def verify_weather_distribution(
    train_samples: List[Tuple[str, str, str]], val_samples: List[Tuple[str, str, str]]
) -> None:
    """Print train/val weather distributions and fail if a train category is absent in val.

    Raises:
        RuntimeError: If any weather condition present in train is missing from val.
    """
    train_weather = [get_weather_type(s[2]) for s in train_samples]
    val_weather = [get_weather_type(s[2]) for s in val_samples]

    train_dist = Counter(train_weather)
    val_dist = Counter(val_weather)

    log.info("\n--- Weather Distribution ---")
    log.info(f"TRAIN: {dict(train_dist)}")
    log.info(f"VALIDATION: {dict(val_dist)}")

    # Check if any weather condition in train is missing from validation completely
    missing_in_val = set(train_dist.keys()) - set(val_dist.keys())
    if missing_in_val:
        log.info(f"⚠️ WARNING/FAILURE: Validation split is imbalanced! Missing weather categories: {missing_in_val}")
        raise RuntimeError(f"Validation split lacks representation for weather categories: {missing_in_val}")
    log.info("----------------------------\n")


def get_dataloaders(
    root_dir: str = "data/WXSOD",
    image_size: int = 384,
    batch_size: int = 4,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    base_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Build the train/val/test_sys/test_real loaders.

    When ``distributed`` is ``True`` every loader gets a ``DistributedSampler``
    (train shuffles; the eval loaders do not).

    Args:
        root_dir: Dataset root containing ``train_sys``/``test_sys``/``test_real``.
        image_size: Square output resolution.
        batch_size: Per-``DataLoader`` batch size.
        num_workers: Dataloader workers (also enables persistent workers).
        max_samples: Optional sample cap passed to each dataset.
        distributed: If ``True`` use ``DistributedSampler`` on all four loaders.
        rank: Process rank (when distributed).
        world_size: World size (when distributed).
        base_seed: Deterministic augmentation seed base.

    Returns:
        ``(train_loader, val_loader, test_synth_loader, test_real_loader)``.
    """
    # Base datasets
    train_sys_full = WXSODDataset(root_dir=root_dir, split="train", image_size=image_size, base_seed=base_seed)

    # Scene-aware group split (with hard scene-disjointness check)
    train_samples, val_samples = scene_group_split(train_sys_full.samples)

    verify_weather_distribution(train_samples, val_samples)

    # Create final dataset objects
    train_ds = WXSODDataset(root_dir=root_dir, split="train", image_size=image_size, max_samples=max_samples, base_seed=base_seed)
    train_ds.set_samples(train_samples)

    val_ds = WXSODDataset(root_dir=root_dir, split="val", image_size=image_size, max_samples=max_samples, base_seed=base_seed)
    val_ds.set_samples(val_samples)
    # Val transform should not have horizontal flip or color jitter
    val_ds.transform = build_transforms("val", image_size)

    test_synth_ds = WXSODDataset(root_dir=root_dir, split="test_sys", image_size=image_size, max_samples=max_samples, base_seed=base_seed)
    test_real_ds = WXSODDataset(root_dir=root_dir, split="test_real", image_size=image_size, max_samples=max_samples, base_seed=base_seed)

    _pin = True
    _persist = num_workers > 0

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_synth_sampler = DistributedSampler(test_synth_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_real_sampler = DistributedSampler(test_real_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_synth_sampler = None
        test_real_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, drop_last=True,
        pin_memory=_pin, persistent_workers=_persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler,
        num_workers=num_workers, pin_memory=_pin, persistent_workers=_persist,
    )
    test_synth_loader = DataLoader(
        test_synth_ds, batch_size=batch_size, shuffle=False, sampler=test_synth_sampler,
        num_workers=num_workers, pin_memory=_pin, persistent_workers=_persist,
    )
    test_real_loader = DataLoader(
        test_real_ds, batch_size=batch_size, shuffle=False, sampler=test_real_sampler,
        num_workers=num_workers, pin_memory=_pin, persistent_workers=_persist,
    )

    return train_loader, val_loader, test_synth_loader, test_real_loader
