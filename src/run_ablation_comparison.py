import json
import os
import sys

def main():
    checkpoint_root = os.environ.get("CHECKPOINT_ROOT")
    if not checkpoint_root:
        if os.path.isdir("/kaggle/working"):
            checkpoint_root = "/kaggle/working/WXSOD_Checkpoints"
        else:
            checkpoint_root = "checkpoints"

    experiments = ["ablation_full_moe", "ablation_moe16_dense"]
    
    # Headers
    print(f"{'Config':<20} | {'Split':<9} | {'Ep1 MAE':<7} | {'Ep4 MAE':<7} | {'Ep8 MAE':<7} | {'Ep8 S-meas':<10} | {'Params':<8} | {'Ep-time':<7}")
    print("-" * 88)

    for exp in experiments:
        exp_dir = os.path.join(checkpoint_root, exp)
        metrics_path = os.path.join(exp_dir, "epoch_metrics.json")
        info_path = os.path.join(exp_dir, "run_info.json")

        if not os.path.exists(metrics_path) or not os.path.exists(info_path):
            print(f"{exp:<20} | Missing data in {exp_dir}")
            continue

        with open(info_path, "r") as f:
            info = json.load(f)
            params = f"{info['total_params'] / 1e6:.1f}M"

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        if not metrics:
            print(f"{exp:<20} | No metrics found")
            continue

        # Extract stats for Ep1, Ep4, Ep8
        def get_metric(epoch_idx, key):
            for m in metrics:
                if m["epoch"] == epoch_idx:
                    return m["val"][key]
            return None

        ep1_mae = get_metric(1, "MAE")
        ep4_mae = get_metric(4, "MAE")
        ep8_mae = get_metric(8, "MAE")
        ep8_smeas = get_metric(8, "S_measure")
        
        ep1_mae_str = f"{ep1_mae:.3f}" if ep1_mae is not None else "N/A"
        ep4_mae_str = f"{ep4_mae:.3f}" if ep4_mae is not None else "N/A"
        ep8_mae_str = f"{ep8_mae:.3f}" if ep8_mae is not None else "N/A"
        ep8_smeas_str = f"{ep8_smeas:.3f}" if ep8_smeas is not None else "N/A"

        # Calculate average epoch time
        times = [m.get("wall_clock_s", 0) for m in metrics if "wall_clock_s" in m]
        avg_time = sum(times) / len(times) if times else 0
        time_str = f"{avg_time:.0f}s"

        # Note: Since the training loop validation only provides one generic 'val' metric right now 
        # (derived from train_sys split), we just output 'val' for the split name instead of 'test_sys' / 'test_real'
        print(f"{exp:<20} | {'val':<9} | {ep1_mae_str:<7} | {ep4_mae_str:<7} | {ep8_mae_str:<7} | {ep8_smeas_str:<10} | {params:<8} | {time_str:<7}")

if __name__ == "__main__":
    main()
