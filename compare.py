import torch

torch.manual_seed(0)

# deterministic input
image = torch.randn(1, 64, 64, 1, 12)
mask = torch.zeros(1, 64, 64, 1, 3)
timestamps = torch.tensor([[[15, 6, 2024]]])

# --- original ---
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

orig_model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
orig_model.eval()

sample = MaskedOlmoEarthSample(sentinel2_l2a=image, sentinel2_l2a_mask=mask, timestamps=timestamps)
with torch.no_grad():
    orig_features = orig_model.encoder(sample, fast_pass=True, patch_size=4)["tokens_and_masks"].sentinel2_l2a

# --- extracted ---
from olmo.encoder import EncoderInput
from olmo.loader import load_encoder

new_encoder = load_encoder("OlmoEarth-v1-Base")
new_encoder.eval()

inp = EncoderInput(data={"sentinel2_l2a": image}, masks={"sentinel2_l2a": mask}, timestamps=timestamps)
with torch.no_grad():
    new_features = new_encoder(inp, patch_size=4)["sentinel2_l2a"]

# --- compare ---
print(f"Original shape:  {orig_features.shape}")
print(f"Extracted shape: {new_features.shape}")
print(f"Max abs diff:    {(orig_features - new_features).abs().max().item():.2e}")
print(f"Allclose:        {torch.allclose(orig_features, new_features, atol=1e-6)}")
