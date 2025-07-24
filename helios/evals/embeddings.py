"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from helios.evals.datasets.configs import TaskType
from helios.nn.flexihelios import Encoder, PoolingType, TokensAndMasks
from helios.train.masking import MaskedHeliosSample
from einops import rearrange
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader,
    task_type: TaskType,
    model: Encoder,
    patch_size: int,
    pooling_type: PoolingType = PoolingType.MAX,
    concat_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader."""
    embeddings = []
    labels = []
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    torchhub_id = "dinov2_vitb14"
    model = torch.hub.load("facebookresearch/dinov2", torchhub_id)
    model = model.eval()
    device = next(model.parameters()).device
    total_samples = len(data_loader)
    with torch.no_grad():
        for i, (masked_helios_sample, label) in enumerate(data_loader):
            masked_helios_sample_dict = masked_helios_sample.as_dict(return_none=False)
            for key, val in masked_helios_sample_dict.items():
                if key == "timestamps":
                    masked_helios_sample_dict[key] = val.to(device=device)
                else:
                    masked_helios_sample_dict[key] = val.to(
                        device=device, dtype=torch.bfloat16
                    )

            masked_helios_sample = MaskedHeliosSample.from_dict(
                masked_helios_sample_dict
            )
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                # TODO: Model expects masked helios sample we need to pass empty masks
                # Likely we want to have a flag that checks for eval mode and passes empty masks
                # batch_embeddings: TokensAndMasks = model(
                #     masked_helios_sample, patch_size=patch_size
                # )[0]  # (bsz, dim)
                # create rgb only s2 data
                s2_data = masked_helios_sample.sentinel2_l2a
                logger.info(f"s2_data: {s2_data.shape}")
                # DinoV2 exbects B, C , H , W
                # channels first

                s2_data = rearrange(s2_data, "b h w t c -> b (c t) h w")
                s2_data = s2_data[:, [3,2,1], :, :]
                # Resize the image to 224x224
                s2_data = F.interpolate(s2_data, size=(224, 224), mode="bilinear", align_corners=False)
                batch_embeddings = model.forward_features(s2_data)
            spatial_pool = True if task_type == TaskType.SEGMENTATION else False
            # Concat features across modalities in space averaged across time
            averaged_embeddings = batch_embeddings.pool_unmasked_tokens(
                pooling_type,
                spatial_pooling=spatial_pool,
                concat_features=concat_features,
            )
            embeddings.append(averaged_embeddings.cpu())
            labels.append(label)
            logger.debug(f"Processed {i} / {total_samples}")

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels
