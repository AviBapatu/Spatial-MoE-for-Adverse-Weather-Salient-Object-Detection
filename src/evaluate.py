import os
import json
import csv
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
import hashlib

from src.dataset import get_dataloaders, get_weather_type
from src.model import SpatialMoESODNet
from src.metrics import SODMetrics, BoundaryMetrics
from src.boundary import compute_boundary, BOUNDARY_KERNEL_SIZE, BOUNDARY_KERNEL_SHAPE

def verify_parameter_consistency(model):
    names = sorted([name for name, _ in model.named_parameters()])
    name_str = ",".join(names)
    return hashlib.md5(name_str.encode('utf-8')).hexdigest()

def reverse_geometry(pred_tensor, meta):
    """
    pred_tensor: [1, 1, H, W]
    meta: dictionary with pad and size info
    Returns: [orig_h, orig_w] numpy array
    """
    pred = pred_tensor[0, 0].cpu().numpy()
    
    pad_top = meta['pad_top']
    pad_left = meta['pad_left']
    resized_h = meta['resized_h']
    resized_w = meta['resized_w']
    orig_h = meta['orig_h']
    orig_w = meta['orig_w']
    
    # 1. Unpad
    pred_cropped = pred[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]
    
    # 2. Resize to orig
    pred_original = cv2.resize(pred_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    assert pred_original.shape == (orig_h, orig_w)
    return pred_original

def evaluate(model, dataloader, output_dir, use_tta=False, ablation_cfg=None):
    model.eval()
    device = next(model.parameters()).device
    
    global_metrics = SODMetrics()
    global_boundary = BoundaryMetrics()
    
    weather_metrics = defaultdict(SODMetrics)
    weather_boundary = defaultdict(BoundaryMetrics)
    weather_counts = defaultdict(int)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=True):
            images = batch['image'].to(device)
            names = batch['name']
            
            # Forward pass
            out, _ = model(images, ablation_cfg=ablation_cfg)
            pred_prob = torch.sigmoid(out.saliency_logits)
            
            if use_tta:
                # Horizontal flip
                images_flipped = torch.flip(images, dims=[-1])
                out_flipped, _ = model(images_flipped, ablation_cfg=ablation_cfg)
                pred_prob_flipped = torch.sigmoid(out_flipped.saliency_logits)
                pred_prob_unflipped = torch.flip(pred_prob_flipped, dims=[-1])
                pred_prob = 0.5 * pred_prob + 0.5 * pred_prob_unflipped
                
            for i in range(images.size(0)):
                meta = {k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i] for k, v in batch['meta'].items()}
                stem = names[i]
                weather = get_weather_type(stem)
                
                # Reverse geometry
                pred_orig = reverse_geometry(pred_prob[i:i+1], meta)
                
                # GT
                gt_path = str(meta.get('gt_path'))
                gt_orig = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                
                if gt_orig is None:
                    raise FileNotFoundError(f"Missing GT: {gt_path}")
                assert pred_orig.shape == gt_orig.shape, f"Shape mismatch: Pred {pred_orig.shape} vs GT {gt_orig.shape}"
                
                # 8-bit prediction
                pred_uint8 = np.round(np.clip(pred_orig, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"{stem}.png"), pred_uint8)
                
                # Metrics update
                global_metrics.step(pred_uint8, gt_orig)
                weather_metrics[weather].step(pred_uint8, gt_orig)
                weather_counts[weather] += 1
                
                # Boundary metrics update
                gt_bin = (gt_orig > 127).astype(np.float32)
                gt_bound = compute_boundary(gt_bin)
                pred_bin = (pred_orig > 0.5).astype(np.float32)
                pred_bound = compute_boundary(pred_bin)
                global_boundary.step(pred_bound, gt_bound)
                weather_boundary[weather].step(pred_bound, gt_bound)
                
    results = {
        "global": {**global_metrics.get_results(), **global_boundary.get_results(), "sample_count": sum(weather_counts.values())},
        "weather": {}
    }
    
    for w in sorted(weather_counts.keys()):
        count = weather_counts[w]
        w_res = {**weather_metrics[w].get_results(), **weather_boundary[w].get_results()}
        w_res["sample_count"] = count
        if count < 10:
            w_res["warning"] = "LOW SAMPLE"
        results["weather"][w] = w_res
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, required=True, choices=['test_sys', 'test_real', 'both'], help='Dataset to evaluate')
    parser.add_argument('--tta', type=str, default='none', choices=['none', 'hflip'], help='TTA mode')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/wxsod-dataset')
    parser.add_argument('--out_dir', type=str, default='evaluation')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Checkpoint Integrity
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Extract config from checkpoint if available to initialize model properly
    model_cfg = checkpoint.get('config', {}).get('model', {})
    use_deep_supervision = model_cfg.get('deep_supervision', False)
    num_experts = model_cfg.get('num_experts', 6) # Default to 6 if missing
    window_size = model_cfg.get('window_size', 8) # Default to 8 if missing
    
    model = SpatialMoESODNet(
        use_deep_supervision=use_deep_supervision,
        num_experts=num_experts,
        window_size=window_size
    ).to(device)
    
    model_hash = verify_parameter_consistency(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    eval_dir = os.path.join(args.out_dir, ckpt_name)
    
    print(f"--- EVALUATION CONFIGURATION ---")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Best Metric: {checkpoint.get('best_metric', 'Unknown')}")
    print(f"Model Hash: {model_hash}")
    print(f"TTA: {args.tta}")
    print(f"--------------------------------\n")
    
    # Disable router noise and random augmentation implicitly by eval()
    model.eval()
    
    _, _, test_sys_loader, test_real_loader = get_dataloaders(
        root_dir=args.data_dir, 
        batch_size=1, # Strict batch size 1 for geometry extraction ease
        num_workers=2, 
        distributed=False
    )
    
    datasets_to_run = []
    if args.dataset in ['test_sys', 'both']:
        datasets_to_run.append(('test_sys', test_sys_loader))
    if args.dataset in ['test_real', 'both']:
        datasets_to_run.append(('test_real', test_real_loader))
        
    for ds_name, loader in datasets_to_run:
        print(f"Evaluating {ds_name}...")
        ds_out_dir = os.path.join(eval_dir, ds_name, args.tta)
        
        results = evaluate(model, loader, ds_out_dir, use_tta=(args.tta == 'hflip'))
        
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save manifest
        manifest = {
            "checkpoint": args.checkpoint,
            "dataset": ds_name,
            "tta": args.tta,
            "num_images": results["global"]["sample_count"],
            "prediction_resolution": "original",
            "boundary_kernel": BOUNDARY_KERNEL_SIZE,
            "boundary_shape": "ellipse",
            "model_hash": model_hash
        }
        
        manifest_filename = f"evaluation_manifest_{ds_name}_{timestamp}.json"
        with open(os.path.join(ds_out_dir, manifest_filename), "w") as f:
            json.dump(manifest, f, indent=4)
            
        metrics_filename = f"metrics_{ds_name}_{timestamp}.json"
        with open(os.path.join(ds_out_dir, metrics_filename), "w") as f:
            json.dump(results, f, indent=4)
            
        # Summary TXT
        summary_filename = f"summary_{ds_name}_{timestamp}.txt"
        with open(os.path.join(ds_out_dir, summary_filename), "w") as f:
            f.write(f"--- {ds_name.upper()} RESULTS ---\n")
            f.write(f"Global Sample Count: {results['global']['sample_count']}\n")
            for k, v in results['global'].items():
                if isinstance(v, float):
                    f.write(f"{k}: {v:.4f}\n")
                    
            f.write(f"\n--- WEATHER-WISE RESULTS ---\n")
            for w, w_res in results['weather'].items():
                f.write(f"\nWeather: {w} (Count: {w_res['sample_count']})\n")
                if "warning" in w_res:
                    f.write(f"WARNING: {w_res['warning']}\n")
                for k, v in w_res.items():
                    if isinstance(v, float):
                        f.write(f"  {k}: {v:.4f}\n")
                        
        print(f"Completed {ds_name}. Results saved to {ds_out_dir}")

if __name__ == '__main__':
    main()
