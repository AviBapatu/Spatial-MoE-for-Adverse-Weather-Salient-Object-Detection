import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import NamedTuple, Optional

class DecoderOutput(NamedTuple):
    saliency_logits: torch.Tensor
    boundary_logits: torch.Tensor
    aux_logits_16: Optional[torch.Tensor]
    aux_logits_8: Optional[torch.Tensor]
    aux_logits_4: Optional[torch.Tensor]

class EntropyFusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj_y = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.proj_entropy = nn.Conv2d(1, out_dim, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, Y, entropy, disable_entropy: bool = False):
        y_proj = self.proj_y(Y)
        if disable_entropy:
            return y_proj
        # Normalize entropy approximately to [0, 1] assuming K=2
        entropy_norm = entropy / math.log(2.0 + 1e-8)
        e_proj = self.proj_entropy(entropy_norm)
        return y_proj + self.scale * e_proj

class GlobalCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim)
        )
        
    def forward(self, q_x, kv_x):
        B, C, H_q, W_q = q_x.shape
        _, _, H_kv, W_kv = kv_x.shape
        
        q = q_x.flatten(2).transpose(1, 2)
        kv = kv_x.flatten(2).transpose(1, 2)
        
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        
        attn_out, _ = self.mha(q_norm, kv_norm, kv_norm)
        q = q + attn_out
        
        q_norm = self.norm_ffn(q)
        ffn_out = self.ffn(q_norm)
        q = q + ffn_out
        
        return q.transpose(1, 2).reshape(B, C, H_q, W_q)

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowedCrossAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.relative_position_index: torch.Tensor
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, q_win, kv_win, mask=None):
        B_, N, C = q_win.shape
        
        q = self.q_proj(q_win).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(kv_win).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(kv_win).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class WindowedCrossAttentionBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8, ffn_expansion=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = WindowedCrossAttention(dim, window_size, num_heads)
        
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim)
        )
        
    def forward(self, q_x, kv_x):
        B, C, H, W = q_x.shape
        
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        
        if pad_r > 0 or pad_b > 0:
            q_x = F.pad(q_x, (0, pad_r, 0, pad_b))
            kv_x = F.pad(kv_x, (0, pad_r, 0, pad_b))
            
        _, _, Hp, Wp = q_x.shape
        
        q = q_x.permute(0, 2, 3, 1)
        kv = kv_x.permute(0, 2, 3, 1)
        
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        
        q_windows = window_partition(q_norm, self.window_size)
        kv_windows = window_partition(kv_norm, self.window_size)
        
        q_windows = q_windows.view(-1, self.window_size * self.window_size, C)
        kv_windows = kv_windows.view(-1, self.window_size * self.window_size, C)
        
        if pad_r > 0 or pad_b > 0:
            img_mask = torch.zeros((1, Hp, Wp, 1), device=q_x.device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -pad_b), slice(-pad_b, None)) if pad_b > 0 else (slice(0, None),)
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -pad_r), slice(-pad_r, None)) if pad_r > 0 else (slice(0, None),)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100)).masked_fill(attn_mask == 0, float(0))
        else:
            attn_mask = None
            
        attn_windows = self.attn(q_windows, kv_windows, mask=attn_mask)
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        q_attn = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        q = q + q_attn
        
        q_norm_ffn = self.norm_ffn(q)
        ffn_out = self.ffn(q_norm_ffn)
        q = q + ffn_out
        
        if pad_r > 0 or pad_b > 0:
            q = q[:, :H, :W, :].contiguous()
            
        return q.permute(0, 3, 1, 2)

class RefinementBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(32, dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(32, dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return x + self.conv(x)

class SpatialMoEDecoder(nn.Module):
    def __init__(self, dim=256, window_size=8, use_deep_supervision=True):
        super().__init__()
        self.dim = dim
        self.use_deep_supervision = use_deep_supervision
        
        self.fuse_16 = EntropyFusionBlock(dim, dim)
        self.fuse_8 = EntropyFusionBlock(dim, dim)
        self.fuse_4 = EntropyFusionBlock(dim, dim)
        
        self.global_context_pool = nn.AdaptiveAvgPool2d(1)
        self.global_context_proj = nn.Conv2d(dim, dim, 1)
        
        self.cross_attn_16_to_8 = GlobalCrossAttentionBlock(dim)
        self.cross_attn_8_to_4 = WindowedCrossAttentionBlock(dim, window_size=window_size)
        
        self.refine_4 = RefinementBlock(dim)
        self.refine_2 = RefinementBlock(dim)
        self.refine_1 = RefinementBlock(dim)
        
        self.saliency_head = nn.Conv2d(dim, 1, 1)
        self.edge_head = nn.Conv2d(dim, 1, 1)
        
        if self.use_deep_supervision:
            self.aux_head_16 = nn.Conv2d(dim, 1, 1)
            self.aux_head_8 = nn.Conv2d(dim, 1, 1)
            self.aux_head_4 = nn.Conv2d(dim, 1, 1)
            
    def forward(self, Y_4, Y_8, Y_16, H_4, H_8, H_16, disable_entropy: bool = False):
        F16 = self.fuse_16(Y_16, H_16, disable_entropy=disable_entropy)
        F8_local = self.fuse_8(Y_8, H_8, disable_entropy=disable_entropy)
        F4_local = self.fuse_4(Y_4, H_4, disable_entropy=disable_entropy)
        
        F16_up = F.interpolate(F16, size=F8_local.shape[-2:], mode='bilinear', align_corners=False)
        F8 = self.cross_attn_16_to_8(q_x=F16_up, kv_x=F8_local)
        
        g_ctx = self.global_context_proj(self.global_context_pool(F8))
        F4_ctx = F4_local + g_ctx
        
        F8_up = F.interpolate(F8, size=F4_ctx.shape[-2:], mode='bilinear', align_corners=False)
        F4 = self.cross_attn_8_to_4(q_x=F8_up, kv_x=F4_ctx)
        
        x4 = self.refine_4(F4)
        x2 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = self.refine_2(x2)
        x1 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = self.refine_1(x1)
        
        saliency_logits = self.saliency_head(x1)
        boundary_logits = self.edge_head(x1)
        
        aux_16 = self.aux_head_16(F16) if self.use_deep_supervision else None
        aux_8 = self.aux_head_8(F8) if self.use_deep_supervision else None
        aux_4 = self.aux_head_4(F4) if self.use_deep_supervision else None
        
        return DecoderOutput(
            saliency_logits=saliency_logits,
            boundary_logits=boundary_logits,
            aux_logits_16=aux_16,
            aux_logits_8=aux_8,
            aux_logits_4=aux_4
        )
