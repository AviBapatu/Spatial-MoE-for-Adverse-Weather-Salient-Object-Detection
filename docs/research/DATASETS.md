# Datasets

> **Note on citations.** Line-number citations below predate the current source
> layout (the decoder is now the `src/decoder/` package, the training loop lives under
> `src/training/`, the evaluation artifacts are under `results/`). Treat module and
> function names as authoritative and `RESEARCH_TRUTH.md` as the verified reference;
> numbers live in `RESULTS.md`, which is generated from the result files.

## Primary Dataset: WXSOD

**Source:** `src/dataset.py:19`, `spatial-moe-adverse-weather-sod-blueprint.md:10-11`

**Description:** Weather-aware Salient Object Detection dataset with pixel-wise saliency ground truth plus per-image weather labels. Contains both synthetic and real-world adverse weather test splits.

### Directory Structure
```
data/WXSDO_data/
├── train_sys/
│   ├── input/   # Training images
│   └── gt/      # Ground truth masks
├── test_sys/
│   ├── input/   # Synthetic test images
│   └── gt/      # Ground truth masks
└── test_real/
    ├── input/   # Real-world test images
    └── gt/      # Ground truth masks
```

### Splits
| Split | Images | Purpose |
|-------|--------|---------|
| train_sys | ~12,891 | Training (from notebook validation) |
| test_sys | 1,500 | Synthetic weather test |
| test_real | 554 | Real-world weather test |

### Filename Convention
Pattern: `{scene_id}_{weather}.{ext}`

Examples:
- `001_fog.png` → scene_id=001, weather=fog
- `042_rainafog.jpg` → scene_id=042, weather=rainafog

**Scene ID extraction** (`dataset.py:206-214`):
```python
def get_scene_id(stem):
    parts = stem.split('_')
    if len(parts) < 2:
        raise ValueError(f"Stem '{stem}' does not match expected pattern sceneID_weather")
    return parts[0]
```

**Weather type extraction** (`dataset.py:216-220`):
```python
def get_weather_type(stem):
    parts = stem.split('_')
    if len(parts) >= 2:
        return "_".join(parts[1:])
    return "unknown"
```

### Weather Categories (from evaluation results)

**test_sys categories:** clean, dark, fog, light, rain, rainafog, rainasnow, snow, snowafog

**test_real categories:** dark, fog, light, rain, snow

Note: test_real has fewer categories than test_sys.

## Data Loading (`src/dataset.py`)

### WXSODDataset Class (`dataset.py:19-204`)

**Initialization:**
1. Determine paths based on split (train/val → train_sys, test_sys, test_real)
2. Glob all input images, create GT mapping by stem
3. Validate strict 1:1 mapping (every input has GT, every GT has input)
4. Setup albumentations transforms

**Key methods:**
- `set_epoch(epoch)`: Sets epoch for deterministic augmentation seeding
- `set_samples(samples)`: Overrides sample list (used after splitting)
- `_aspect_preserving_resize_pad(image, mask)`: Resize to max_dim=384, pad to square
- `_compute_edge_map(mask_binary)`: Morphological boundary from binary mask

### Scene-Aware Splitting (`dataset.py:243-271`)

```python
# Extract scene IDs
scene_ids = [get_scene_id(stem) for _, _, stem in train_sys_full.samples]

# GroupShuffleSplit: 80% train, 20% val, grouped by scene
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(samples, groups=scene_ids))

# Hard assertion for disjointness
assert train_scenes_set.isdisjoint(val_scenes_set)
```

**Purpose:** Prevents data leakage where images from the same scene appear in both train and val.

**Weather distribution verification** (`dataset.py:222-239`):
```python
verify_weather_distribution(train_samples, val_samples)
```
Checks that all weather categories in train are represented in val. Raises RuntimeError if any are missing.

### Augmentation

**Train transforms** (`dataset.py:78-82`):
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
```

**Val/test transforms** (`dataset.py:84-86`):
```python
A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
```

### Deterministic Augmentation Seeding (`dataset.py:170-179`)

```python
sample_seed_str = f"{self.base_seed}_{self.epoch}_{index}_{name}"
sample_seed = int(hashlib.md5(sample_seed_str.encode('utf-8')).hexdigest()[:8], 16)

random.seed(sample_seed)
np.random.seed(sample_seed)
torch.manual_seed(sample_seed)
```

**Guarantee:** Same sample + same epoch → same augmentation every time.

### Aspect-Preserving Resize + Pad (`dataset.py:107-143`)

1. Compute scale = `image_size / max(orig_h, orig_w)`
2. Resize image (bilinear) and mask (nearest) to `(resized_h, resized_w)`
3. Pad to `(image_size, image_size)`:
   - Image: `cv2.BORDER_REFLECT_101`
   - Mask: `cv2.BORDER_CONSTANT, value=0`

**Metadata stored for geometry reversal during evaluation:**
```python
meta = {
    'orig_h', 'orig_w', 'resized_h', 'resized_w',
    'pad_top', 'pad_bottom', 'pad_left', 'pad_right'
}
```

### Boundary Computation (`src/boundary.py`)

**Kernel:** 5×5 ellipse (`cv2.MORPH_ELLIPSE`)

**Method:**
```python
dilated = cv2.dilate(mask, kernel, iterations=1)
eroded = cv2.erode(mask, kernel, iterations=1)
boundary = dilated - eroded
```

Produces a boundary band (not single-pixel edge) from the binary mask.

## DataLoader Configuration

```python
get_dataloaders(
    root_dir="data/WXSDO",
    image_size=384,
    batch_size=4,
    num_workers=4,
    max_samples=None,    # None = full dataset
    distributed=False,
    rank=0,
    world_size=1,
    base_seed=42
)
```

**Distributed:** Uses `DistributedSampler` with `shuffle=True` when `distributed=True`.

**Return order:** `train_loader, val_loader, test_synth_loader, test_real_loader`

## Dataset in Notebook Validation (`generate_notebook.py`)

Expected counts for validation gate:
```python
splits = {"train_sys": 12891, "test_sys": 1500, "test_real": 554}
```

Required subdirectories per split: `{split}/input/`, `{split}/gt/`

## Blueprint Dataset Context (`spatial-moe-adverse-weather-sod-blueprint.md`)

### Related Benchmarks Mentioned

**Direct SOD targets:**
- WXSOD: 14,945 images with weather labels + synthetic/real test splits
- RGBT-SOD (VT5000, VT1000, VT821): thermal-paired SOD

**Weather understanding (transferable):**
- Foggy Cityscapes / Foggy Driving
- RainCityscapes / Rain-KITTI / RainDrop
- Snow100K / CSD
- ExDark (low-light)
- ACDC (fog/rain/snow/night with pixel-level labels)
- XWOD (2026, 10K extreme weather images)
