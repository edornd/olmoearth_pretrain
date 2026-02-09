import torch

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

# Load model from HuggingFace
model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
model.eval()

# Create synthetic input (B=1, H=64, W=64, T=1, C=12 for Sentinel-2)
dummy_image = torch.randn(1, 64, 64, 1, 12)
dummy_mask = torch.ones(1, 64, 64, 1, 3) * MaskValue.ONLINE_ENCODER.value
dummy_timestamps = torch.tensor([[[15, 6, 2024]]])  # day, month (0-indexed), year

sample = MaskedOlmoEarthSample(
    sentinel2_l2a=dummy_image,
    sentinel2_l2a_mask=dummy_mask,
    timestamps=dummy_timestamps,
)

with torch.no_grad():
    output = model.encoder(sample, fast_pass=True, patch_size=4)
    features = output["tokens_and_masks"].sentinel2_l2a
    print(f"Output shape: {features.shape}")  # (B, H', W', T, S, D)
