import os
import json
import argparse
from tqdm import tqdm
from src.dataset import WXSODDataset

def evaluate_padding(dataset_name, dataset, image_size=384):
    non_trivial_padding_count = 0
    total_samples = len(dataset)
    total_padded_fraction_sum = 0.0
    
    threshold = 0.05 * (image_size * 4)
    
    for i in tqdm(range(total_samples), desc=f"Evaluating {dataset_name} padding"):
        # We don't want to run the full augmentations, just get the meta
        # Getting an item from dataset runs everything.
        item = dataset[i]
        meta = item['meta']
        
        pad_sum = meta['pad_top'] + meta['pad_bottom'] + meta['pad_left'] + meta['pad_right']
        if pad_sum > threshold:
            non_trivial_padding_count += 1
            
        padded_pixels = (image_size * image_size) - (meta['resized_h'] * meta['resized_w'])
        padded_fraction = padded_pixels / (image_size * image_size)
        total_padded_fraction_sum += padded_fraction
        
    return {
        "non_trivial_padding_fraction": non_trivial_padding_count / total_samples if total_samples > 0 else 0,
        "mean_padded_pixel_fraction": total_padded_fraction_sum / total_samples if total_samples > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/WXSDO_data')
    parser.add_argument('--out_dir', type=str, default='evaluation')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    test_sys_ds = WXSODDataset(root_dir=args.data_dir, split="test_sys", image_size=384)
    test_real_ds = WXSODDataset(root_dir=args.data_dir, split="test_real", image_size=384)
    
    stats = {
        "test_sys": evaluate_padding("test_sys", test_sys_ds),
        "test_real": evaluate_padding("test_real", test_real_ds),
        "loss_masks_padding_region": False,
        "evidence": "SpatialMoELoss.forward (src/loss.py lines 164-167) and _soft_iou_loss (line 89) compute loss on entire [B, 1, H, W] tensors without masking."
    }
    
    out_file = os.path.join(args.out_dir, "padding_stats.json")
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"Padding stats saved to {out_file}")
    
    print("\n--- PADDING AUDIT SUMMARY ---")
    print(f"loss_masks_padding_region: {stats['loss_masks_padding_region']}")

if __name__ == "__main__":
    main()
