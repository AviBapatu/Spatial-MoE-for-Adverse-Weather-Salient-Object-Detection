import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("relative_position_index", torch.zeros(10, dtype=torch.long))
        self.relative_position_bias_table = nn.Parameter(torch.zeros(10, 5))
        self.window_size = 2

    def forward(self, x):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        return relative_position_bias

m = M()
print(m(torch.randn(1)))

m_dp = nn.DataParallel(m)
print(m_dp(torch.randn(1)))
