"""Test Pastis dataset."""

from pathlib import Path

import pytest
import torch

from helios.evals.datasets.pastis_dataset import PASTISRDataset


@pytest.fixture
def mock_pastis_data(tmp_path: Path) -> Path:
    """Create mock PASTIS-R data for testing."""
    # Create mock data with small dimensions
    data = {
        "s2_images": torch.randn(2, 12, 13, 64, 64),  # 2 samples
        "s1_images": torch.randn(2, 12, 2, 64, 64),
        "targets": torch.randint(0, 2, (2, 64, 64)),
        "months": torch.arange(1, 13).repeat(
            2, 1
        ),  # Shape: (2, 12) - one sequence per sample
    }

    # Save mock data
    save_path = tmp_path / "pastis_r_train.pt"
    torch.save(data, save_path)
    return tmp_path


def test_pastis_dataset_initialization(mock_pastis_data: Path) -> None:
    """Test basic initialization and functionality of PASTISRDataset."""
    # Test multimodal initialization
    dataset = PASTISRDataset(
        path_to_splits=mock_pastis_data, split="train", is_multimodal=True
    )

    assert len(dataset) == 2  # Should have 2 samples

    # Test single sample access
    sample, label = dataset[0]

    # Check basic properties
    assert isinstance(sample.sentinel2_l2a, torch.Tensor)
    assert isinstance(sample.sentinel1, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    # Check shapes
    assert sample.sentinel2_l2a.shape[2] == 12  # 12 timestamps
    assert sample.sentinel1.shape[2] == 12  # 12 timestamps
    assert label.shape == (64, 64)  # Label should be 64x64

    # Test non-multimodal initialization
    dataset_s2_only = PASTISRDataset(
        path_to_splits=mock_pastis_data, split="train", is_multimodal=False
    )

    sample_s2, label_s2 = dataset_s2_only[0]
    assert sample_s2.sentinel1 is None  # Should not have S1 data
    assert sample_s2.sentinel2_l2a is not None  # Should have S2 data
