import torch
import torch.nn as nn
from src.decoder import WindowedCrossAttention

model = WindowedCrossAttention(dim=64, window_size=8, num_heads=8).cuda()
dp_model = nn.DataParallel(model, device_ids=[0, 1])
q = torch.randn(2, 64, 64).cuda()
kv = torch.randn(2, 64, 64).cuda()
dp_model(q, kv)
