import torch
import torch.nn as nn
from src.decoder import WindowedCrossAttention

model = WindowedCrossAttention(dim=64, window_size=8, num_heads=8)
q = torch.randn(1, 64, 64)
kv = torch.randn(1, 64, 64)
model(q, kv)
