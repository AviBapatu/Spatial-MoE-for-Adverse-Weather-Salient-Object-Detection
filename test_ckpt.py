import torch
import shutil
import os
import zipfile

# 1. Create a dummy checkpoint
ckpt = {'model': torch.nn.Linear(10, 10).state_dict(), 'epoch': 1}
torch.save(ckpt, 'test_checkpoint.pth')

# 2. Extract it (like Kaggle does)
os.makedirs('test_unzipped', exist_ok=True)
with zipfile.ZipFile('test_checkpoint.pth', 'r') as zip_ref:
    zip_ref.extractall('test_unzipped')

# 3. Re-zip it
shutil.make_archive('test_rezipped', 'zip', 'test_unzipped')
os.rename('test_rezipped.zip', 'test_rezipped.pth')

# 4. Try loading it
try:
    torch.load('test_rezipped.pth', weights_only=False)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
