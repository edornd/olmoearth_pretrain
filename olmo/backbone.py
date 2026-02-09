import logging

import torch
from torch import Tensor, nn

from olmo.constants import BASE_GSD, NORMALIZATION_STATS, MaskValue, Modality
from olmo.encoder import Encoder, EncoderInput
from olmo.loader import load_encoder

logger = logging.getLogger(__name__)

try:
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY  # type: ignore

    _HAS_TERRATORCH = True
except ImportError:
    _HAS_TERRATORCH = False


def _maybe_register(cls: type) -> type:
    if _HAS_TERRATORCH:
        TERRATORCH_BACKBONE_REGISTRY.register(cls)  # type: ignore
    return cls


@_maybe_register
class OlmoEarthBackbone(nn.Module):
    """TerraTorch-compatible backbone wrapper for the OlmoEarth encoder.

    Accepts ``[B, C, H, W]`` input and returns ``list[Tensor]`` with a single
    feature map of shape ``[B, D, H', W']``.

    Args:
        model_id: HuggingFace model name (e.g. ``"OlmoEarth-v1-Base"``).
        modality: which modality the input corresponds to.
        bands: optional subset of bands in the input (in order). If ``None``,
            all bands of the modality are expected.
        patch_size: patchification size (1–8, smaller = finer but slower).
        input_res: ground-sample distance of the input in metres.
        month: 0-indexed month for the temporal encoding (default 6 = July).
        normalize: if ``True``, apply the OlmoEarth normalization inside ``forward``
            (expects raw sensor values). If ``False``, the caller must pre-normalize.
    """

    def __init__(
        self,
        model_id: str = "OlmoEarth-v1-Base",
        modality: str = "sentinel2_l2a",
        bands: list[str] | None = None,
        patch_size: int = 4,
        input_res: int = BASE_GSD,
        month: int = 6,
        normalize: bool = True,
    ):
        super().__init__()
        self.encoder: Encoder = load_encoder(model_id, load_weights=True)
        self.modality = modality
        self.patch_size = patch_size
        self.input_res = input_res
        self.month = month

        spec = Modality.get(modality)
        full_band_order = spec.band_order

        # if the user passes a subset of bands, build an index mapping
        if bands is not None:
            self.band_indices = [full_band_order.index(b) for b in bands]
            self.num_input_bands = len(bands)
            self.num_full_bands = spec.num_bands
            self._needs_reindex = True
            norm_bands = bands
        else:
            self.band_indices = list(range(spec.num_bands))
            self.num_input_bands = spec.num_bands
            self.num_full_bands = spec.num_bands
            self._needs_reindex = False
            norm_bands = full_band_order

        self.num_band_sets = spec.num_band_sets
        self.embedding_size = self.encoder.embedding_size

        # normalization buffers
        self._do_normalize = normalize
        if normalize and modality in NORMALIZATION_STATS:
            stats = NORMALIZATION_STATS[modality]
            mins, maxs = [], []
            for band_name in norm_bands:
                mean, std = stats[band_name]
                mins.append(mean - 2 * std)
                maxs.append(mean + 2 * std)
            # shape [1, C, 1, 1] for broadcasting over [B, C, H, W]
            self.register_buffer("_norm_min", torch.tensor(mins, dtype=torch.float32).view(1, -1, 1, 1))
            self.register_buffer(
                "_norm_range",
                torch.tensor(
                    [mx - mn for mn, mx in zip(mins, maxs)],
                    dtype=torch.float32,
                ).view(1, -1, 1, 1),
            )
        else:
            self._do_normalize = False

    def forward(self, x: Tensor) -> list[Tensor]:
        """Run inference.

        Args:
            x: ``[B, C, H, W]`` input tensor.

        Returns:
            Single-element list containing ``[B, D, H', W']`` feature map.
        """
        b, _c, h, w = x.shape

        # normalize
        if self._do_normalize:
            x = (x - self._norm_min) / self._norm_range  # type: ignore

        # reindex bands if a subset was provided → pad to full band count
        if self._needs_reindex:
            full = x.new_zeros(b, self.num_full_bands, h, w)
            for dst, src in enumerate(self.band_indices):
                full[:, src] = x[:, dst]
            x = full

        # [B, C, H, W] → [B, H, W, 1, C]
        x = x.permute(0, 2, 3, 1).unsqueeze(3)

        # mask: [B, H, W, 1, num_band_sets]  — all ONLINE_ENCODER (0)
        mask = x.new_full((b, h, w, 1, self.num_band_sets), MaskValue.ONLINE_ENCODER)

        # timestamps: [B, 1, 3]  (day, month_0idx, year)
        # day and year are never used by the encoder
        timestamps = torch.tensor([[[15, self.month, 2024]]], device=x.device, dtype=torch.long).expand(b, -1, -1)

        inp = EncoderInput(
            data={self.modality: x},
            masks={self.modality: mask},
            timestamps=timestamps,
        )

        out = self.encoder(inp, patch_size=self.patch_size, input_res=self.input_res)
        features = out[self.modality]  # [B, H', W', T=1, BandSets, D]

        # pool over timestep (dim 3) and band-sets (dim 4) → [B, H', W', D]
        features = features.mean(dim=(3, 4))
        # → [B, D, H', W']
        features = features.permute(0, 3, 1, 2)
        return [features]
