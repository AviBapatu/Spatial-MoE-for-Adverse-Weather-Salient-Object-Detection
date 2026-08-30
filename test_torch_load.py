import torch
import os
import shutil
import zipfile

# Create a simple tensor and save it
data = {'weight': torch.randn(10, 10)}
torch.save(data, 'test.pth')

# Unzip it
os.makedirs('test_unzipped', exist_ok=True)
with zipfile.ZipFile('test.pth', 'r') as zip_ref:
    zip_ref.extractall('test_unzipped')

# Try loading the unzipped folder
try:
    loaded = torch.load('test_unzipped')
    print("Successfully loaded unzipped folder")
except Exception as e:
    print(f"Failed to load unzipped folder: {e}")

