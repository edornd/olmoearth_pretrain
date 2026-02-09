import logging
import math

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from olmo.attention import Block
from olmo.constants import BASE_GSD, Modality, ModalitySpec
from olmo.encodings import (
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
)
from olmo.patch_embed import FlexiPatchEmbed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# lightweight input container (replaces 30-field MaskedOlmoEarthSample)
# ---------------------------------------------------------------------------
class EncoderInput:
    """Lightweight input container for the encoder.

    Holds per-modality data tensors, masks, and timestamps.
    Supports attribute access: ``getattr(inp, 'sentinel2_l2a')`` returns the data tensor,
    ``getattr(inp, 'sentinel2_l2a_mask')`` returns the mask tensor.
    """

    def __init__(self, data: dict[str, Tensor], masks: dict[str, Tensor], timestamps: Tensor):
        self._data = data
        self._masks = masks
        self.timestamps = timestamps

    @property
    def modalities(self) -> list[str]:
        return list(self._data.keys())

    def __getattr__(self, name: str) -> Tensor:
        if name.startswith("_") or name == "timestamps":
            raise AttributeError(name)
        if name.endswith("_mask"):
            modality = name[: -len("_mask")]
            if modality in self._masks:
                return self._masks[modality]
        elif name in self._data:
            return self._data[name]
        raise AttributeError(name)

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        return f"{modality}_mask"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _get_modalities_to_process(available: list[str], supported: list[str]) -> list[str]:
    return sorted(set(supported).intersection(available))


def _return_modalities_from_dict(d: dict[str, Tensor]) -> list[str]:
    return [k for k in d if not k.endswith("_mask")]


# ---------------------------------------------------------------------------
# ProjectAndAggregate (kept for weight-loading; unused at inference)
# ---------------------------------------------------------------------------
class ProjectAndAggregate(nn.Module):
    def __init__(self, embedding_size: int, num_layers: int, aggregate_then_project: bool = True):
        super().__init__()
        projections: list[nn.Module] = [nn.Linear(embedding_size, embedding_size)]
        for _ in range(1, num_layers):
            projections.append(nn.ReLU())
            projections.append(nn.Linear(embedding_size, embedding_size))
        self.projection = nn.Sequential(*projections)
        self.aggregate_then_project = aggregate_then_project


# ---------------------------------------------------------------------------
# MultiModalPatchEmbeddings
# ---------------------------------------------------------------------------
class MultiModalPatchEmbeddings(nn.Module):
    def __init__(self, supported_modality_names: list[str], max_patch_size: int, embedding_size: int):
        super().__init__()
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        self.supported_modality_names = supported_modality_names

        self.per_modality_embeddings = nn.ModuleDict()
        for modality in self.supported_modality_names:
            self.per_modality_embeddings[modality] = self._build_embed(modality)

        # register non-persistent buffers for band-index selection
        for modality in self.supported_modality_names:
            for idx, bandset_indices in enumerate(Modality.get(modality).bandsets_as_indices()):
                self.register_buffer(
                    self._buffer_name(modality, idx),
                    torch.tensor(bandset_indices, dtype=torch.long),
                    persistent=False,
                )

    # ---- naming helpers (must match original) ----
    @staticmethod
    def _buffer_name(modality: str, idx: int) -> str:
        return f"{modality}__{idx}_buffer"

    @staticmethod
    def _embed_name(modality: str, idx: int) -> str:
        return f"{modality}__{idx}"

    # ---- build per-modality embed modules ----
    def _build_embed(self, modality: str) -> nn.ModuleDict:
        spec = Modality.get(modality)
        bandset_indices = spec.bandsets_as_indices()
        if not spec.is_spatial:
            return nn.ModuleDict(
                {
                    self._embed_name(modality, idx): nn.Linear(len(idxs), self.embedding_size)
                    for idx, idxs in enumerate(bandset_indices)
                }
            )
        return nn.ModuleDict(
            {
                self._embed_name(modality, idx): FlexiPatchEmbed(
                    in_chans=len(idxs),
                    embedding_size=self.embedding_size,
                    patch_size_at_16=self.max_patch_size,
                    modality_spec=spec,
                )
                for idx, idxs in enumerate(bandset_indices)
            }
        )

    def _embed_one_bandset(
        self,
        modality: str,
        idx: int,
        data: Tensor,
        mask: Tensor,
        spec: ModalitySpec,
        patch_size: int,
    ) -> tuple[Tensor, Tensor]:
        buf = getattr(self, self._buffer_name(modality, idx))
        band_data = torch.index_select(data, -1, buf)
        embed = self.per_modality_embeddings[modality][self._embed_name(modality, idx)]  # type: ignore

        if not spec.is_spatial:
            token_mask = mask[..., idx]
            tokens = embed(band_data)
        else:
            stride = patch_size * spec.image_tile_size_factor
            token_mask = mask[:, 0::stride, 0::stride, ..., idx]
            tokens = embed(band_data, patch_size=patch_size)
        return tokens, token_mask

    def forward(self, input_data: EncoderInput, patch_size: int) -> dict[str, Tensor]:
        output: dict[str, Tensor] = {}
        modalities = _get_modalities_to_process(input_data.modalities, self.supported_modality_names)
        for modality in modalities:
            spec = Modality.get(modality)
            num_bs = spec.num_band_sets
            data = getattr(input_data, modality)
            mask = getattr(input_data, EncoderInput.get_masked_modality_name(modality))

            tokens_list, masks_list = [], []
            for idx in range(num_bs):
                tok, msk = self._embed_one_bandset(modality, idx, data, mask, spec, patch_size)
                tokens_list.append(tok)
                masks_list.append(msk)
            output[modality] = torch.stack(tokens_list, dim=-2)
            output[EncoderInput.get_masked_modality_name(modality)] = torch.stack(masks_list, dim=-1)
        return output


# ---------------------------------------------------------------------------
# CompositeEncodings
# ---------------------------------------------------------------------------
class CompositeEncodings(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        supported_modalities: list[ModalitySpec],
        max_sequence_length: int,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.supported_modalities = supported_modalities
        self.supported_modality_names = [m.name for m in supported_modalities]
        self.max_sequence_length = max_sequence_length
        self.embedding_dim_per_embedding_type = int(embedding_size * 0.25)

        n = self.embedding_dim_per_embedding_type

        # 1D sinusoidal position encoding for time
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_encoding(torch.arange(max_sequence_length), n),
            requires_grad=False,
        )
        # month encoding
        self.month_embed = nn.Embedding.from_pretrained(get_month_encoding_table(n), freeze=True)

        # per-modality channel embeddings
        self.per_modality_channel_embeddings = nn.ParameterDict()
        for mod in self.supported_modalities:
            num_bs = mod.num_band_sets
            shape = (num_bs, n)
            if not learnable_channel_embeddings and not random_channel_embeddings:
                self.per_modality_channel_embeddings[mod.name] = nn.Parameter(torch.zeros(shape), requires_grad=False)
            elif random_channel_embeddings:
                self.per_modality_channel_embeddings[mod.name] = nn.Parameter(
                    torch.rand(shape), requires_grad=not learnable_channel_embeddings
                )
            else:
                self.per_modality_channel_embeddings[mod.name] = nn.Parameter(
                    torch.zeros(shape), requires_grad=learnable_channel_embeddings
                )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _gsd_ratio(input_res: float, patch_size: int) -> float:
        return input_res * patch_size / BASE_GSD

    def _apply_per_modality(
        self,
        modality_name: str,
        tokens: Tensor,
        timestamps: Tensor | None,
        patch_size: int | None,
        input_res: int | None,
    ) -> Tensor:
        spec = Modality.get(modality_name)
        n = self.embedding_dim_per_embedding_type
        device = tokens.device

        # build einops pattern based on ndim
        if tokens.ndim == 3:
            b, b_s, _ = tokens.shape
            ein, kw = "b b_s d", {"b": b, "b_s": b_s}
        elif tokens.ndim == 4:
            b, t, b_s, _ = tokens.shape
            ein, kw = "b t b_s d", {"b": b, "t": t, "b_s": b_s}
        elif tokens.ndim == 5:
            b, h, w, b_s, _ = tokens.shape
            ein, kw = "b h w b_s d", {"b": b, "h": h, "w": w, "b_s": b_s}
        elif tokens.ndim == 6:
            b, h, w, t, b_s, _ = tokens.shape
            ein, kw = "b h w t b_s d", {"b": b, "h": h, "w": w, "t": t, "b_s": b_s}
        else:
            raise ValueError(f"Unsupported token shape: {tokens.shape}")

        embed = torch.zeros(tokens.shape, device=device)

        # channel
        ch_embed = self.per_modality_channel_embeddings[spec.name]
        embed[..., :n] += repeat(ch_embed, f"b_s d -> {ein}", **kw).to(device)

        # temporal
        if spec.is_multitemporal:
            t_val = tokens.shape[3] if tokens.ndim == 6 else (tokens.shape[1] if tokens.ndim == 4 else 1)
            embed[..., n : n * 2] += repeat(self.pos_embed[:t_val], f"t d -> {ein}", **kw).to(device)
            assert timestamps is not None
            months = timestamps[:, :, 1]
            month_e = self.month_embed(months)
            embed[..., n * 2 : n * 3] += repeat(month_e, f"b t d -> {ein}", **kw).to(device)

        # spatial
        if spec.is_spatial:
            assert input_res is not None and patch_size is not None
            h = tokens.shape[1] if tokens.ndim >= 5 else tokens.shape[1]
            if tokens.ndim >= 5:
                h = tokens.shape[1]
            gsd = self._gsd_ratio(input_res, patch_size)
            spatial_e = get_2d_sincos_pos_encoding_with_resolution(
                grid_size=h,
                res=torch.ones(b, device=device) * gsd,
                encoding_dim=n,
                device=device,
            )
            spatial_e = rearrange(spatial_e, "b (h w) d -> b h w d", h=h)
            embed[..., n * 3 : n * 4] += repeat(spatial_e, f"b h w d -> {ein}", **kw)

        return tokens + embed

    def forward(
        self,
        tokens_dict: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        available = _return_modalities_from_dict(tokens_dict)
        for mod in _get_modalities_to_process(available, self.supported_modality_names):
            out[mod] = self._apply_per_modality(mod, tokens_dict[mod], timestamps, patch_size, input_res)
        return out


# ---------------------------------------------------------------------------
# FlexiVitBase
# ---------------------------------------------------------------------------
class FlexiVitBase(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        max_sequence_length: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        supported_modalities: list[ModalitySpec],
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.supported_modalities = supported_modalities
        self.supported_modality_names = [m.name for m in supported_modalities]
        self.max_sequence_length = max_sequence_length

        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=qk_norm,
                    norm_layer=nn.LayerNorm,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )

        self.composite_encodings = CompositeEncodings(
            embedding_size,
            self.supported_modalities,
            max_sequence_length,
            learnable_channel_embeddings,
            random_channel_embeddings,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def collapse_and_combine(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Flatten per-modality tokens/masks into single (B, N, D) and (B, N) tensors."""
        tokens, masks = [], []
        available = _return_modalities_from_dict(x)
        for mod in _get_modalities_to_process(available, self.supported_modality_names):
            mask_name = EncoderInput.get_masked_modality_name(mod)
            tokens.append(rearrange(x[mod], "b ... d -> b (...) d"))
            masks.append(rearrange(x[mask_name], "b ... -> b (...)"))
        return torch.cat(tokens, dim=1), torch.cat(masks, dim=1)

    def split_tokens_masks_and_dims(
        self,
        x: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, tuple]]:
        tokens_d: dict[str, Tensor] = {}
        masks_d: dict[str, Tensor] = {}
        dims_d: dict[str, tuple] = {}
        available = _return_modalities_from_dict(x)
        for mod in _get_modalities_to_process(available, self.supported_modality_names):
            tokens_d[mod] = x[mod]
            dims_d[mod] = x[mod].shape
            masks_d[EncoderInput.get_masked_modality_name(mod)] = x[EncoderInput.get_masked_modality_name(mod)]
        return tokens_d, masks_d, dims_d

    @staticmethod
    def split_and_expand(x: Tensor, dims_dict: dict[str, tuple]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        offset = 0
        for mod, dims in dims_dict.items():
            mid = dims[1:-1]
            n = math.prod(mid)
            out[mod] = x[:, offset : offset + n].view(x.shape[0], *mid, x.shape[-1])
            offset += n
        return out


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(FlexiVitBase):
    def __init__(
        self,
        embedding_size: int,
        max_patch_size: int,
        min_patch_size: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        supported_modalities: list[ModalitySpec],
        max_sequence_length: int,
        num_register_tokens: int = 0,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        num_projection_layers: int = 1,
        aggregate_then_project: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__(
            embedding_size=embedding_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            learnable_channel_embeddings=learnable_channel_embeddings,
            drop_path=drop_path,
            supported_modalities=supported_modalities,
            random_channel_embeddings=random_channel_embeddings,
            qk_norm=qk_norm,
        )
        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0
        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(torch.zeros(num_register_tokens, embedding_size))
            nn.init.xavier_uniform_(self.register_tokens)
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size

        self.patch_embeddings = MultiModalPatchEmbeddings(
            self.supported_modality_names,
            self.max_patch_size,
            self.embedding_size,
        )
        self.project_and_aggregate = ProjectAndAggregate(
            embedding_size=self.embedding_size,
            num_layers=num_projection_layers,
            aggregate_then_project=aggregate_then_project,
        )
        self.norm = nn.LayerNorm(self.embedding_size)
        self.apply(self._init_weights)

    def forward(
        self,
        x: EncoderInput,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> dict[str, Tensor]:
        """Inference-only forward (equivalent to original fast_pass=True).

        Returns a dict mapping modality names to feature tensors and ``<modality>_mask`` to masks.
        """
        # 1. patchify
        patched = self.patch_embeddings(x, patch_size)

        # 2. split tokens / masks / dims
        tokens_only, masks_only, dims_dict = self.split_tokens_masks_and_dims(patched)

        # 3. add composite encodings
        encoded = self.composite_encodings(tokens_only, x.timestamps, patch_size, input_res)
        encoded.update(masks_only)

        # 4. flatten all modalities into a single sequence
        tokens, _mask = self.collapse_and_combine(encoded)

        # 5. (optionally) prepend register tokens
        if self.has_register_tokens:
            reg = self.register_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([reg, tokens], dim=1)

        # 6. transformer blocks (no masking / no attention mask for fast inference)
        for blk in self.blocks:
            tokens = blk(tokens)

        # 7. pop register tokens
        if self.has_register_tokens:
            tokens = tokens[:, self.num_register_tokens :]

        # 8. layer norm
        tokens = self.norm(tokens)

        # 9. split back per modality and merge masks
        per_mod = self.split_and_expand(tokens, dims_dict)
        per_mod.update(masks_only)
        return per_mod
