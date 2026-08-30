import torch
from src.decoder import WindowedCrossAttention

model = WindowedCrossAttention(dim=64, window_size=8, num_heads=8).cuda()
model = torch.compile(model)
q = torch.randn(1, 64, 64).cuda()
kv = torch.randn(1, 64, 64).cuda()
model(q, kv)
