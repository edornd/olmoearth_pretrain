"""Unit tests for HeliosSample."""

import torch

from helios.data.dataset import HeliosSample


def test_all_attrs_have_bands() -> None:
    """Test all attributes are described in attribute_to_bands."""
    for attribute_name in HeliosSample._fields:
        _ = HeliosSample.num_bands(attribute_name)


def test_subsetting() -> None:
    """Test subsetting works."""
    (
        b,
        h,
        w,
        t,
    ) = 1, 16, 16, 100
    sample = HeliosSample(
        sentinel2=torch.ones((b, h, w, t, HeliosSample.num_bands("sentinel2"))),
        timestamps=torch.ones((b, t, HeliosSample.num_bands("timestamps"))),
    )
    subsetted_sample = sample.subset(
        patch_size=4, max_tokens_per_instance=100, hw_to_sample=[4]
    )

    # 16 / 4 = 4 tokens along the height and width dimension
    # total s2 tokens = t * 4 * 4 * 3 (band sets) = 48
    # so a token budget of floor(100 / 48) = 2
    assert subsetted_sample.time == 2


def test_subsetting_worldcover_too() -> None:
    """Test subsetting works."""
    (
        b,
        h,
        w,
        t,
    ) = 1, 16, 16, 100
    sample = HeliosSample(
        sentinel2=torch.ones((b, h, w, t, HeliosSample.num_bands("sentinel2"))),
        worldcover=torch.ones((b, h, w, HeliosSample.num_bands("worldcover"))),
        timestamps=torch.ones((b, t, HeliosSample.num_bands("timestamps"))),
    )
    subsetted_sample = sample.subset(
        patch_size=4, max_tokens_per_instance=100, hw_to_sample=[4]
    )

    # 16 / 4 = 4 tokens along the height and width dimension
    # total s2 tokens = t * 4 * 4 * 3 (band sets) = 48
    # total worldcover tokens = 4 * 4 * 1 (band set) = 16
    # so a token budget of floor((100 - 16) / 48 = 1)

    assert subsetted_sample.time == 1
