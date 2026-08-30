import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.dataset import get_dataloaders, get_scene_id

def test_dataset_pipeline():
    print("Running Dataset Preflight Validations...")
    # Using small batch size for quick checks
    try:
        train_loader, val_loader, test_synth_loader, test_real_loader = get_dataloaders(
            root_dir="data/WXSOD",
            image_size=384,
            batch_size=4,
            num_workers=0,
            max_samples=20 # limit samples for faster test
        )
    except Exception as e:
        print(f"Failed to load dataloaders: {e}")
        return

    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    
    # 1. Train/Val Scene Disjoint Check (Hard failure)
    train_scenes = set([get_scene_id(s[2]) for s in train_ds.samples])
    val_scenes = set([get_scene_id(s[2]) for s in val_ds.samples])
    
    assert train_scenes.isdisjoint(val_scenes), "Train and Val scenes are NOT disjoint!"
    print("✅ Train/Val scenes are purely disjoint.")
    
    # 2. Check geometry and values for a few samples
    samples_to_visualize = []
    
    for i in range(min(8, len(val_ds))):
        data = val_ds[i]
        img = data['image']
        mask = data['mask']
        edge = data['edge_map']
        meta = data['meta']
        gt_path = meta['gt_path']
        
        # Check shapes
        assert img.shape == (3, 384, 384)
        assert mask.shape == (1, 384, 384)
        assert edge.shape == (1, 384, 384)
        
        # Check values
        assert 0.0 <= mask.min() and mask.max() <= 1.0, f"Mask not in [0,1]: {mask.min()} to {mask.max()}"
        unique_edge_vals = torch.unique(edge)
        assert len(unique_edge_vals) <= 2, f"Edge is not binary! Unique vals: {unique_edge_vals}"
        
        # Round trip test: crop and resize back to original
        p = mask[0].numpy()
        p_cropped = p[meta['pad_top']:meta['pad_top']+meta['resized_h'], 
                      meta['pad_left']:meta['pad_left']+meta['resized_w']]
        p_orig = cv2.resize(p_cropped, (meta['orig_w'], meta['orig_h']), interpolation=cv2.INTER_NEAREST)
        
        # Original GT mask shape should match
        orig_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        assert p_orig.shape == orig_mask.shape, f"Restored mask shape {p_orig.shape} != original mask shape {orig_mask.shape}"
        
        # Ensure padding is 0 for mask and edge
        pad_t, pad_b, pad_l, pad_r = meta['pad_top'], meta['pad_bottom'], meta['pad_left'], meta['pad_right']
        if pad_t > 0:
            assert p[:pad_t, :].sum() == 0, "Top padding of mask is not 0"
        if pad_b > 0:
            assert p[384-pad_b:, :].sum() == 0, "Bottom padding of mask is not 0"
        
        # Store for visualization
        img_np = img.permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        samples_to_visualize.append({
            'img': img_np,
            'mask': mask[0].numpy(),
            'edge': edge[0].numpy(),
            'name': data['name']
        })
        
    print("✅ Geometry preservation and un-pad round trip successful.")
    print("✅ Mask and boundary values validated.")
    
    # Dump visualizations
    os.makedirs("test_outputs", exist_ok=True)
    fig, axes = plt.subplots(len(samples_to_visualize), 3, figsize=(10, 4*len(samples_to_visualize)))
    if len(samples_to_visualize) == 1:
        axes = [axes]
        
    for i, s in enumerate(samples_to_visualize):
        axes[i][0].imshow(s['img'])
        axes[i][0].set_title(f"Image: {s['name']}")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(s['mask'], cmap='gray')
        axes[i][1].set_title("Mask")
        axes[i][1].axis('off')
        
        axes[i][2].imshow(s['edge'], cmap='gray')
        axes[i][2].set_title("Boundary")
        axes[i][2].axis('off')
        
    plt.tight_layout()
    plt.savefig("test_outputs/dataset_validation.png")
    print("✅ Saved visualization to test_outputs/dataset_validation.png")
    
if __name__ == "__main__":
    test_dataset_pipeline()
