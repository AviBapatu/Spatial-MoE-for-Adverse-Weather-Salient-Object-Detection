import os
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WXSODDataset(Dataset):
    def __init__(self, root_dir="data/WXSOD", split="train", image_size=384):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # Determine paths based on split
        if split == "train":
            self.input_dir = os.path.join(root_dir, "train", "input")
            self.gt_dir = os.path.join(root_dir, "train", "gt")
        elif split == "test/synthetic":
            self.input_dir = os.path.join(root_dir, "test", "synthetic", "input")
            self.gt_dir = os.path.join(root_dir, "test", "synthetic", "gt")
        elif split == "test/real":
            self.input_dir = os.path.join(root_dir, "test", "real", "input")
            self.gt_dir = os.path.join(root_dir, "test", "real", "gt")
        else:
            raise ValueError(f"Unknown split: {split}")
            
        # Get all input images
        input_pattern = os.path.join(self.input_dir, "*")
        self.image_paths = sorted([p for p in glob.glob(input_pattern) if os.path.isfile(p)])
        
        # Create a mapping of stem (basename without extension) to gt path
        gt_pattern = os.path.join(self.gt_dir, "*")
        gt_paths = [p for p in glob.glob(gt_pattern) if os.path.isfile(p)]
        self.gt_map = {Path(p).stem: p for p in gt_paths}
        
        # Filter valid pairs
        self.samples = []
        for img_path in self.image_paths:
            stem = Path(img_path).stem
            if stem in self.gt_map:
                self.samples.append((img_path, self.gt_map[stem], stem))
        
        # Setup albumentations transforms
        if self.split == "train":
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)
        
    def _compute_edge_map(self, mask):
        """
        Compute edge map from mask using Laplacian operator.
        """
        # Ensure it's numpy for OpenCV
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = mask
            
        # Compute Laplacian
        laplacian = cv2.Laplacian(mask_np, cv2.CV_32F)
        edge_map = np.abs(laplacian)
        
        # Normalize edge map to [0, 1]
        max_val = edge_map.max()
        if max_val > 0:
            edge_map = edge_map / max_val
            
        return torch.from_numpy(edge_map).unsqueeze(0).float()

    def __getitem__(self, idx):
        img_path, gt_path, name = self.samples[idx]
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {gt_path}")
            
        # Normalize mask to 0-1 for processing (if it's 0-255)
        # Assuming ground truth masks are 0 (bg) and 255 (fg)
        mask = mask.astype(np.float32) / 255.0
        
        # Apply transforms
        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented['image']
        mask_tensor = augmented['mask']
        
        # mask_tensor is [H, W] since ToTensorV2 converts 2D mask to 2D tensor
        # Add channel dimension to mask -> [1, H, W]
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
            
        # Compute edge map using the augmented mask
        edge_map = self._compute_edge_map(mask_tensor.squeeze(0))
        
        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'edge_map': edge_map,
            'name': name
        }

def get_dataloaders(root_dir="data/WXSOD", image_size=384, batch_size=4, num_workers=4):
    """
    Returns data loaders for train, test_synthetic, and test_real.
    """
    train_dataset = WXSODDataset(root_dir=root_dir, split="train", image_size=image_size)
    test_synth_dataset = WXSODDataset(root_dir=root_dir, split="test/synthetic", image_size=image_size)
    test_real_dataset = WXSODDataset(root_dir=root_dir, split="test/real", image_size=image_size)
    
    # drop_last=True for train to handle batch norm issues with size 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_synth_loader = DataLoader(test_synth_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_real_loader = DataLoader(test_real_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_synth_loader, test_real_loader

if __name__ == '__main__':
    # Test the dataset
    print("Testing WXSODDataset...")
    
    # Try testing with the provided data directory
    try:
        dataset = WXSODDataset(root_dir="data/WXSOD", split="train")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSingle sample shapes:")
            print(f"  Image: {sample['image'].shape}")
            print(f"  Mask: {sample['mask'].shape}")
            print(f"  Edge Map: {sample['edge_map'].shape}")
            print(f"  Name: {sample['name']}")
            
            print("\nTesting DataLoader...")
            loader = DataLoader(dataset, batch_size=2, shuffle=True)
            batch = next(iter(loader))
            print("Batch shapes:")
            print(f"  Images: {batch['image'].shape}")
            print(f"  Masks: {batch['mask'].shape}")
            print(f"  Edge Maps: {batch['edge_map'].shape}")
            print(f"  Names: {batch['name']}")
        else:
            print("Dataset is empty. Directories might not have any valid image/mask pairs yet.")
            print(f"Checked image dir: {dataset.input_dir}")
            print(f"Checked GT dir: {dataset.gt_dir}")
    except Exception as e:
        print(f"Failed to run test block: {e}")
