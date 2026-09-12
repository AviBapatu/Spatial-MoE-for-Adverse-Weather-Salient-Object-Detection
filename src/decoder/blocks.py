"""Reusable building blocks for the Spatial-MoE decoder.

These primitives (entropy fusion, global/windowed cross-attention, window
partition helpers, and residual refinement) are shared by the top-level
``SpatialMoEDecoder`` assembly in :mod:`src.decoder.decoder` and are usable
independently (e.g. in unit tests).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyFusionBlock(nn.Module):
    """Adds a learnable-scaled routing-entropy channel onto the input features.

    Multi-scale fusion step of the decoder: each scale's features ``Y`` are
    combined with a normalized entropy map ``H`` so the decoder can use routing
    uncertainty as an extra contextual cue.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the entropy fusion block.

        Args:
            in_dim: Input feature channel count.
            out_dim: Output feature channel count.
        """
        super().__init__()
        self.proj_y = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.proj_entropy = nn.Conv2d(1, out_dim, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        Y: torch.Tensor,
        entropy: torch.Tensor,
        disable_entropy: bool = False,
    ) -> torch.Tensor:
        """Fuse features ``Y`` with the (optionally disabled) entropy map.

        Args:
            Y: Features ``[B, in_dim, H, W]``.
            entropy: Routing entropy map ``[B, 1, H, W]``.
            disable_entropy: If ``True`` (ablation), skip the entropy channel.

        Returns:
            Fused features ``[B, out_dim, H, W]``.
        """
        y_proj = self.proj_y(Y)
        if disable_entropy:
            return y_proj
        # Normalize entropy approximately to [0, 1] assuming K=2 experts.
        entropy_norm = entropy / math.log(2.0 + 1e-8)
        e_proj = self.proj_entropy(entropy_norm)
        return y_proj + self.scale * e_proj


class GlobalCrossAttentionBlock(nn.Module):
    """Global cross-attention with a twin-FFN over flattened spatial tokens.

    ``q_x`` attends over ``kv_x`` using global (full-sequence)
    ``nn.MultiheadAttention``, followed by residual FFN blocks.
    """

    def __init__(self, dim: int, num_heads: int = 8, ffn_expansion: int = 4) -> None:
        """Initialize the global cross-attention block.

        Args:
            dim: Feature dimensionality.
            num_heads: Number of attention heads.
            ffn_expansion: Hidden width ratio of the FFN.
        """
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim),
        )

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> torch.Tensor:
        """Apply global cross-attention with *q_x* as query, *kv_x* as key/value.

        Args:
            q_x: Query features ``[B, C, H_q, W_q]``.
            kv_x: Key/value features ``[B, C, H_kv, W_kv]``.

        Returns:
            Refined features ``[B, C, H_q, W_q]``.
        """
        B, C, H_q, W_q = q_x.shape

        q = q_x.flatten(2).transpose(1, 2)  # [B, H_q*W_q, C]
        kv = kv_x.flatten(2).transpose(1, 2)  # [B, H_kv*W_kv, C]

        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)

        attn_out, _ = self.mha(q_norm, kv_norm, kv_norm)
        q = q + attn_out

        q_norm = self.norm_ffn(q)
        ffn_out = self.ffn(q_norm)
        q = q + ffn_out

        return q.transpose(1, 2).reshape(B, C, H_q, W_q)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Split a ``[B, H, W, C]`` tensor into ``[nW, win, win, C]`` windows.

    Args:
        x: Input tensor laid out as ``[B, H, W, C]``.
        window_size: Side length of each square window.

    Returns:
        Windows tensor of shape ``[nW, window_size, window_size, C]``.
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Inverse of :func:`window_partition`.

    Args:
        windows: Windows ``[nW, window_size, window_size, C]``.
        window_size: Side length of each square window.
        H: Original spatial height.
        W: Original spatial width.

    Returns:
        Reconstructed tensor ``[B, H, W, C]``.
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowedCrossAttention(nn.Module):
    """Windowed multi-head self/cross-attention with relative position bias.

    Operates on ``[num_windows, win*win, C]`` tokens produced by
    :func:`window_partition`.
    """

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8) -> None:
        """Initialize windowed attention.

        Args:
            dim: Feature dimensionality.
            window_size: Side length of each attention window.
            num_heads: Number of attention heads.
        """
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
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        q_win: torch.Tensor,
        kv_win: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply windowed attention to token windows.

        Args:
            q_win: Query tokens ``[nW, win*win, C]``.
            kv_win: Key/value tokens ``[nW, win*win, C]``.
            mask: Optional additive attention mask ``[nW, win*win, win*win]``
                used to suppress padded positions.

        Returns:
            Attended tokens ``[nW, win*win, C]``.
        """
        B_, N, _ = q_win.shape

        q = self.q_proj(q_win).reshape(B_, N, self.num_heads, self.dim // self.num_heads)
        k = self.k_proj(kv_win).reshape(B_, N, self.num_heads, self.dim // self.num_heads)
        v = self.v_proj(kv_win).reshape(B_, N, self.num_heads, self.dim // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        x = self.proj(x)
        return x


class WindowedCrossAttentionBlock(nn.Module):
    """Windowed cross-attention + FFN over a full ``[B, C, H, W]`` feature map.

    Performs window-partitioning, windowed attention, and reverse
    reconstruction, padding non-divisible spatial sizes so the output matches
    the input resolution exactly.
    """

    def __init__(
        self, dim: int, window_size: int = 8, num_heads: int = 8, ffn_expansion: int = 4
    ) -> None:
        """Initialize the windowed attention block.

        Args:
            dim: Feature dimensionality.
            window_size: Side length of each attention window.
            num_heads: Number of attention heads.
            ffn_expansion: Hidden width ratio of the FFN.
        """
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
            nn.Linear(dim * ffn_expansion, dim),
        )

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> torch.Tensor:
        """Apply windowed cross-attention between two feature maps.

        Args:
            q_x: Query features ``[B, C, H, W]``.
            kv_x: Key/value features ``[B, C, H, W]``.

        Returns:
            Refined features ``[B, C, H, W]`` (same spatial size as *q_x*).
        """
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
            h_slices = (
                (slice(0, -self.window_size), slice(-self.window_size, -pad_b), slice(-pad_b, None))
                if pad_b > 0
                else (slice(0, None),)
            )
            w_slices = (
                (slice(0, -self.window_size), slice(-self.window_size, -pad_r), slice(-pad_r, None))
                if pad_r > 0
                else (slice(0, None),)
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100)).masked_fill(
                attn_mask == 0, float(0)
            )
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
    """Residual double-conv feature refinement block.

    ``x + Conv2d -> GroupNorm -> ReLU -> Conv2d -> GroupNorm -> ReLU(x)``.
    """

    def __init__(self, dim: int) -> None:
        """Initialize the refinement block.

        Args:
            dim: Input/output channel count.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(32, dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(32, dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual refinement.

        Args:
            x: Features ``[B, dim, H, W]``.

        Returns:
            Refined features of the same shape.
        """
        return x + self.conv(x)
