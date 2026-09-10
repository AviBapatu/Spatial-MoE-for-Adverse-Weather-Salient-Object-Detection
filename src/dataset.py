import os
import glob
from pathlib import Path
import cv2
cv2.setNumThreads(0)  # Prevent OpenCV thread pool from competing with DataLoader workers
import numpy as np
import random
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

from src.boundary import compute_boundary, get_boundary_kernel

class WXSODDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=384, max_samples=None, transform=None, base_seed=42):
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
        self.samples = []
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
            if split == "train":
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ], is_check_shapes=False)
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ], is_check_shapes=False)
                
        # Morphological parameters for boundary target
        self.boundary_kernel = get_boundary_kernel()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_samples(self, samples):
        self.samples = samples
        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]

    def __len__(self):
        return len(self.samples)
        
    def _compute_edge_map(self, mask_binary):
        # Ensure it is purely binary (0 or 1)
        mask_u8 = (mask_binary > 0.5).astype(np.uint8)
        return compute_boundary(mask_u8, self.boundary_kernel)

    def _aspect_preserving_resize_pad(self, image, mask):
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
        image_padded = cv2.copyMakeBorder(image_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        mask_padded = cv2.copyMakeBorder(mask_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0])
        
        valid_mask_ones = np.ones((resized_h, resized_w), dtype=np.float32)
        valid_mask = cv2.copyMakeBorder(valid_mask_ones, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0])
        
        meta = {
            'orig_h': orig_h,
            'orig_w': orig_w,
            'resized_h': resized_h,
            'resized_w': resized_w,
            'pad_top': pad_top,
            'pad_bottom': pad_bottom,
            'pad_left': pad_left,
            'pad_right': pad_right
        }
        
        return image_padded, mask_padded, valid_mask, meta

    def __getitem__(self, index):
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
        mask = (mask.astype(np.float32) / 255.0)
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
            'meta': meta
        }

def get_scene_id(stem):
    """
    Extracts scene ID from stem. Assumes format {scene_id}_{weather}.
    If it doesn't match this pattern, raises an Error as requested.
    """
    parts = stem.split('_')
    if len(parts) < 2:
        raise ValueError(f"Stem '{stem}' does not match expected pattern sceneID_weather. Cannot reliably extract scene ID.")
    return parts[0]

def get_weather_type(stem):
    parts = stem.split('_')
    if len(parts) >= 2:
        return "_".join(parts[1:])
    return "unknown"

def verify_weather_distribution(train_samples, val_samples):
    train_weather = [get_weather_type(s[2]) for s in train_samples]
    val_weather = [get_weather_type(s[2]) for s in val_samples]
    
    train_dist = Counter(train_weather)
    val_dist = Counter(val_weather)
    
    print("\n--- Weather Distribution ---")
    print(f"TRAIN: {dict(train_dist)}")
    print(f"VALIDATION: {dict(val_dist)}")
    
    # Check if any weather condition in train is missing from validation completely
    # (Assuming all weather conditions in train_sys should ideally be represented)
    missing_in_val = set(train_dist.keys()) - set(val_dist.keys())
    if missing_in_val:
        print(f"⚠️ WARNING/FAILURE: Validation split is imbalanced! Missing weather categories: {missing_in_val}")
        raise RuntimeError(f"Validation split lacks representation for weather categories: {missing_in_val}")
    print("----------------------------\n")

from torch.utils.data.distributed import DistributedSampler

def get_dataloaders(root_dir="data/WXSOD", image_size=384, batch_size=4, num_workers=4, max_samples=None, distributed=False, rank=0, world_size=1, base_seed=42):
    # Base datasets
    train_sys_full = WXSODDataset(root_dir=root_dir, split="train", image_size=image_size, base_seed=base_seed)
    
    # Extract scene IDs
    scene_ids = []
    for _, _, stem in train_sys_full.samples:
        try:
            sid = get_scene_id(stem)
            scene_ids.append(sid)
        except ValueError as e:
            print("CRITICAL: Failed to reliably establish scene grouping.")
            print(e)
            print("Aborting dataset creation. A random split fallback is forbidden to prevent data leakage.")
            raise

    # Scene-aware group split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(train_sys_full.samples, groups=scene_ids))
    
    train_samples = [train_sys_full.samples[i] for i in train_idx]
    val_samples = [train_sys_full.samples[i] for i in val_idx]
    
    # Hard assertion for disjointness
    train_scenes_set = set([get_scene_id(s[2]) for s in train_samples])
    val_scenes_set = set([get_scene_id(s[2]) for s in val_samples])
    assert train_scenes_set.isdisjoint(val_scenes_set), "CRITICAL: Data leakage detected! Train and Val scenes are not disjoint."
    
    verify_weather_distribution(train_samples, val_samples)
    
    # Create final dataset objects
    train_ds = WXSODDataset(root_dir=root_dir, split="train", image_size=image_size, max_samples=max_samples, base_seed=base_seed)
    train_ds.set_samples(train_samples)
    
    val_ds = WXSODDataset(root_dir=root_dir, split="val", image_size=image_size, max_samples=max_samples, base_seed=base_seed)
    val_ds.set_samples(val_samples)
    # Val transform should not have horizontal flip or color jitter
    val_ds.transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))], is_check_shapes=False)
    
    test_synth_ds = WXSODDataset(root_dir=root_dir, split="test_sys", image_size=image_size, max_samples=max_samples, base_seed=base_seed)
    test_real_ds = WXSODDataset(root_dir=root_dir, split="test_real", image_size=image_size, max_samples=max_samples, base_seed=base_seed)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    train_shuffle = (train_sampler is None)
    
    _pin = True
    _persist = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=num_workers, drop_last=True, pin_memory=_pin, persistent_workers=_persist)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=_pin, persistent_workers=_persist)
    test_synth_loader = DataLoader(test_synth_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=_pin, persistent_workers=_persist)
    test_real_loader = DataLoader(test_real_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=_pin, persistent_workers=_persist)
    
    return train_loader, val_loader, test_synth_loader, test_real_loader
