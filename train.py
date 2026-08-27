import os
import argparse
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import SpatialMoESODNet
from src.losses import CombinedMoELoss

def calculate_mae(pred, target):
    """Mean Absolute Error"""
    return torch.abs(pred - target).mean().item()

def calculate_f_beta(pred, target, beta_sq=0.3, threshold=0.5):
    """F-beta score widely used for Salient Object Detection"""
    pred_bin = (pred > threshold).float()
    
    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    f_beta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall + 1e-8)
    return f_beta.mean().item()

def main():
    parser = argparse.ArgumentParser(description="Train Spatial-MoE for SOD")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", type=str, default="data/WXSOD")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit dataset to N samples per split for smoke testing (e.g. --max_samples 50)"
    )
    args = parser.parse_args()
    
    # Smoke-test mode: auto-reduce epochs, grad accumulation, and batch size for speed/memory
    if args.max_samples is not None:
        print(f"\n*** SMOKE TEST MODE: {args.max_samples} samples, overriding epochs=3, grad_accum=1, batch_size=2 ***\n")
        args.epochs = min(args.epochs, 3)
        args.grad_accum_steps = 1
        args.batch_size = min(args.batch_size, 2)
    
    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    print(f"Training on {device} | AMP: {use_amp} | Grad Accum: {args.grad_accum_steps}")
    
    # 1. Setup Dataloaders
    train_loader, test_synth_loader, test_real_loader = get_dataloaders(
        root_dir=args.data_dir, batch_size=args.batch_size,
        image_size=args.image_size, num_workers=args.num_workers,
        max_samples=args.max_samples
    )
    
    if len(train_loader) == 0:
        print(f"Warning: Train dataloader is empty. Please check data directory: {args.data_dir}")
        return
        
    # 2. Model, Loss, Optimizer
    model = SpatialMoESODNet().to(device)
    criterion = CombinedMoELoss(moe_weight=0.01)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    writer = SummaryWriter(log_dir="runs/spatial_moe_sod")
    
    best_mae = float('inf')
    os.makedirs("checkpoints", exist_ok=True)
    global_step = 0
    
    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            edges = batch['edge_map'].to(device)
            
            # Forward with mixed precision
            with autocast(device_type=device.type, enabled=use_amp):
                sal_pred, edge_pred, routing_probs = model(images)
            
            loss, loss_dict = criterion(sal_pred, edge_pred, masks, edges, routing_probs)
            loss = loss / args.grad_accum_steps
            
            # Backward with gradient scaling
            assert isinstance(loss, torch.Tensor)  # Explicit type for GradScaler
            scaled_loss = scaler.scale(loss)
            assert isinstance(scaled_loss, torch.Tensor)
            scaled_loss.backward()
            
            # Step optimizer every grad_accum_steps
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
            
            train_loss += loss.item() * args.grad_accum_steps
            
            # Compute Metrics (detach to save memory)
            with torch.no_grad():
                mae = calculate_mae(sal_pred, masks)
                f_beta = calculate_f_beta(sal_pred, masks)
            train_mae += mae
            
            # Logging
            pbar.set_postfix({
                'Loss': f"{loss.item() * args.grad_accum_steps:.4f}", 
                'MAE': f"{mae:.4f}",
                'F_beta': f"{f_beta:.4f}",
                'LB': f"{loss_dict['moe_load_balancing']:.4f}"
            })
            
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        scheduler.step()
        
        # 4. Validation on Synthetic Split
        model.eval()
        val_mae = 0.0
        val_fbeta = 0.0
        
        if len(test_synth_loader) > 0:
            with torch.no_grad():
                for batch in tqdm(test_synth_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    
                    with autocast(device_type=device.type, enabled=use_amp):
                        sal_pred, _, _ = model(images)
                    
                    val_mae += calculate_mae(sal_pred, masks)
                    val_fbeta += calculate_f_beta(sal_pred, masks)
                    
            val_mae /= len(test_synth_loader)
            val_fbeta /= len(test_synth_loader)
            
            print(f"Epoch {epoch+1} Summary: Train Loss={train_loss:.4f}, Val MAE={val_mae:.4f}, Val F-beta={val_fbeta:.4f}")
            
            # Tensorboard Logging
            writer.add_scalar('Val/MAE', val_mae, epoch)
            writer.add_scalar('Val/F_beta', val_fbeta, epoch)
            
            # Save Best Model
            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), "checkpoints/best_model.pth")
                print(f"  -> Saved new best model with MAE: {best_mae:.4f}")
        else:
            print(f"Epoch {epoch+1} Summary: Train Loss={train_loss:.4f} (No validation data)")
            
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/MAE', train_mae, epoch)
        
    writer.close()
    print(f"\nTraining complete. Best Val MAE: {best_mae:.4f}")

if __name__ == '__main__':
    main()
