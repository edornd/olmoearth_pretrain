import json
import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from olmo.constants import get_modality_specs_from_names
from olmo.encoder import Encoder

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "weights.pth"

# valid HuggingFace model ids
MODEL_IDS = {
    "OlmoEarth-v1-Nano",
    "OlmoEarth-v1-Tiny",
    "OlmoEarth-v1-Base",
    "OlmoEarth-v1-Large",
}


def load_encoder(
    model_id: str = "OlmoEarth-v1-Base",
    load_weights: bool = True,
) -> Encoder:
    """Build and optionally load pretrained weights for the OlmoEarth encoder.

    Args:
        model_id: one of the OlmoEarth model names (e.g. ``"OlmoEarth-v1-Base"``).
        load_weights: if True, download and load the pretrained weights from HuggingFace.

    Returns:
        The ``Encoder`` module (inference-ready).
    """
    repo_id = f"allenai/{model_id}"
    config_path = Path(hf_hub_download(repo_id=repo_id, filename=CONFIG_FILENAME))
    encoder = _build_encoder_from_config(config_path)

    if load_weights:
        weights_path = Path(hf_hub_download(repo_id=repo_id, filename=WEIGHTS_FILENAME))
        _load_encoder_weights(encoder, weights_path)

    return encoder


def load_encoder_from_path(
    model_dir: str | Path,
    load_weights: bool = True,
) -> Encoder:
    """Build encoder from a local directory containing config.json and weights.pth."""
    model_dir = Path(model_dir)
    encoder = _build_encoder_from_config(model_dir / CONFIG_FILENAME)
    if load_weights:
        _load_encoder_weights(encoder, model_dir / WEIGHTS_FILENAME)
    return encoder


def _build_encoder_from_config(config_path: Path) -> Encoder:
    with open(config_path) as f:
        cfg = json.load(f)

    enc = cfg["model"]["encoder_config"]
    supported_modalities = get_modality_specs_from_names(enc["supported_modality_names"])

    return Encoder(
        embedding_size=enc["embedding_size"],
        max_patch_size=enc["max_patch_size"],
        min_patch_size=enc["min_patch_size"],
        num_heads=enc["num_heads"],
        mlp_ratio=enc["mlp_ratio"],
        depth=enc["depth"],
        drop_path=enc["drop_path"],
        supported_modalities=supported_modalities,
        max_sequence_length=enc["max_sequence_length"],
        num_register_tokens=enc.get("num_register_tokens", 0),
        learnable_channel_embeddings=enc.get("learnable_channel_embeddings", True),
        random_channel_embeddings=enc.get("random_channel_embeddings", False),
        num_projection_layers=enc.get("num_projection_layers", 1),
        aggregate_then_project=enc.get("aggregate_then_project", True),
        qk_norm=enc.get("qk_norm", False),
    )


def _load_encoder_weights(encoder: Encoder, weights_path: Path) -> None:
    full_sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    prefix = "encoder."
    encoder_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}

    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing:
        logger.warning("Missing keys when loading encoder weights: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading encoder weights: %s", unexpected)
