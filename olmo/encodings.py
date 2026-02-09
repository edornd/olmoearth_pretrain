import numpy as np
import torch


def get_1d_sincos_pos_encoding(pos: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    assert encoding_dim % 2 == 0, f"encoding_dim must be even, got {encoding_dim}"
    omega = torch.arange(encoding_dim // 2, device=pos.device) / encoding_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("l,d->ld", pos, omega)
    encoding = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return encoding


def get_2d_sincos_pos_encoding(grid: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    assert encoding_dim % 2 == 0
    encoding_dim_1d = encoding_dim // 2
    emb_h = get_1d_sincos_pos_encoding(grid[0], encoding_dim_1d)
    emb_w = get_1d_sincos_pos_encoding(grid[1], encoding_dim_1d)
    return torch.cat([emb_h, emb_w], dim=1)


def get_2d_sincos_pos_encoding_with_resolution(
    grid_size: int,
    res: torch.Tensor,
    encoding_dim: int,
    device: torch.device,
    cls_token: bool = False,
) -> torch.Tensor:
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)

    grid = torch.einsum("chw,n->cnhw", grid, res)
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_encoding(grid, encoding_dim)
    pos_embed = pos_embed.reshape(n, h * w, encoding_dim)
    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([n, 1, encoding_dim], device=pos_embed.device), pos_embed],
            dim=1,
        )
    return pos_embed


def get_month_encoding_table(encoding_dim: int) -> torch.Tensor:
    assert encoding_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))
    dim_per_table = encoding_dim // 2
    sin_table = torch.sin(torch.stack([angles for _ in range(dim_per_table)], dim=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(dim_per_table)], dim=-1))
    return torch.concatenate([sin_table[:-1], cos_table[:-1]], dim=-1)
