from collections.abc import Iterable
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from olmo.constants import ModalitySpec


class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        modality_spec: ModalitySpec,
        patch_size_at_16: int | tuple[int, int],
        in_chans: int = 3,
        embedding_size: int = 128,
        norm_layer: type[nn.Module] | None = None,
        bias: bool = True,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.modality_spec = modality_spec
        self.patch_size = _to_2tuple(patch_size_at_16 * modality_spec.image_tile_size_factor)

        self.proj = nn.Conv2d(in_chans, embedding_size, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = norm_layer(embedding_size) if norm_layer else nn.Identity()

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, x: Tensor, patch_size: int | tuple[int, int] | None = None) -> Tensor:
        batch_size = x.shape[0]
        has_time_dimension = len(x.shape) == 5

        if has_time_dimension:
            num_timesteps = x.shape[3]
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")

        if not patch_size:
            patch_size = self.patch_size
        else:
            if isinstance(patch_size, tuple):
                patch_size = (
                    patch_size[0] * self.modality_spec.image_tile_size_factor,
                    patch_size[1] * self.modality_spec.image_tile_size_factor,
                )
            else:
                patch_size = patch_size * self.modality_spec.image_tile_size_factor
        patch_size = _to_2tuple(patch_size)

        if patch_size != self.patch_size:
            shape = x.shape[-2:]
            new_shape = (
                shape[0] // patch_size[0] * self.patch_size[0],
                shape[1] // patch_size[1] * self.patch_size[1],
            )
            x = F.interpolate(x, size=new_shape, mode=self.interpolation, antialias=self.antialias)

        x = self.proj(x)

        if has_time_dimension:
            _, d, h, w = x.shape
            x = rearrange(x, "(b t) d h w -> b h w t d", b=batch_size, t=num_timesteps, d=d, h=h, w=w)
        else:
            x = rearrange(x, "b d h w -> b h w d")

        x = self.norm(x)
        return x


def _to_2tuple(x: Any) -> tuple[int, int]:
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    return (x, x)
