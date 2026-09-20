import os
import torch
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from src.dataset import WXSODDataset

def test_flip_alignment():
    print("Testing Flip Alignment...")
    dataset = WXSODDataset(root_dir="data/WXSDO_data", split="train", image_size=384)
    # Use p=1.0 to force flip
    dataset.transform = A.Compose([
        A.HorizontalFlip(p=1.0)
    ], is_check_shapes=False)
    
    # Find a sample that will be padded asymmetrically if possible, or just any sample
    sample = dataset[0]
    
    img = sample['image'].permute(1, 2, 0).numpy()
    # Denormalize image for plotting (wait, we didn't add Normalize to our transform!
    # So it's not normalized in this test. Let's just clip it to 0-1 if it is > 1.0, wait dataset reads image as uint8, then converts it?
    # No, dataset returns it as it is if no ToTensor. Oh wait, if no ToTensor it does torch.from_numpy. So it's 0-255 float32.)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0, 1)
    
    pad_mask = sample['pad_mask'][0].numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title('Flipped Image')
    axes[1].imshow(pad_mask, cmap='gray')
    axes[1].set_title('Pad Mask (Is it flipped?)')
    
    os.makedirs("tests/output", exist_ok=True)
    plt.savefig("tests/output/flip_alignment_test.png")
    print("Saved visualization to tests/output/flip_alignment_test.png")
    
def test_end_to_end_criterion():
    print("\nTesting End-to-End Criterion Call...")
    # This requires running train_ddp.py with small settings or just directly here
    from src.model import SpatialMoESODNet
    from src.loss import SpatialMoELoss
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpatialMoESODNet().to(device)
    criterion = SpatialMoELoss()
    
    B, C, H, W = 1, 3, 128, 128
    
    images = torch.randn(B, C, H, W, device=device)
    targets = torch.zeros(B, 1, H, W, device=device)
    gt_boundaries = torch.zeros(B, 1, H, W, device=device)
    pad_masks = torch.ones(B, 1, H, W, device=device)
    
    try:
        out, moe_outputs = model(images)
        loss, loss_dict = criterion(
            saliency_logits=out.saliency_logits, 
            edge_logits=out.boundary_logits, 
            moe_outputs=moe_outputs, 
            target=targets, 
            gt_boundary=gt_boundaries,
            aux_logits_16=out.aux_logits_16,
            aux_logits_8=out.aux_logits_8,
            aux_logits_4=out.aux_logits_4,
            pad_mask=pad_masks
        )
        print("Criterion success! Total loss:", loss.item())
    except Exception as e:
        print("Criterion failed!")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_flip_alignment()
    test_end_to_end_criterion()
