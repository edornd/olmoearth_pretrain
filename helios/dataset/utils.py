"""Utility functions for the data module."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from helios.dataset.schemas import (Sentinel2FrequencyMetadataDataModel,
                                    Sentinel2MonthlyMetadataDataModel,
                                    TrainingDataIndexDataModel)

T = TypeVar("T")
FrequencyType = Literal["monthly", "freq"]


def load_metadata(path: Path | str, schema: type[T], **kwargs: Any) -> DataFrame[T]:
    """Load metadata from a file with the specified schema.

    Args:
        path: Path to the file
        schema: Pandera schema class to validate against
        **kwargs: Additional arguments passed to pd.read_csv
    """
    # check the extension of the file say not implemented if not csv
    file_extension = (
        path.split(".")[-1] if isinstance(path, str) else path.name.split(".")[-1]
    )
    if file_extension not in ["csv", "parquet"]:
        raise NotImplementedError(f"File extension {file_extension} not supported")
    if file_extension == "csv":
        return pd.read_csv(path, **kwargs)
    elif file_extension == "parquet":
        return pd.read_parquet(path, **kwargs)
    else:
        raise NotImplementedError(f"File extension {file_extension} not supported")


@pa.check_types
def load_data_index(
    data_index_path: Path | str, **kwargs: Any
) -> DataFrame[TrainingDataIndexDataModel]:
    """Load the data index from a csv file."""
    return load_metadata(data_index_path, TrainingDataIndexDataModel, **kwargs)


class DataSourceMetadataRegistry:
    """Registry for data source metadata loading functions."""

    _registry: dict[str, dict[FrequencyType, Callable[..., DataFrame[T]]]] = {}

    @classmethod
    def register(
        cls, data_source: str, frequency_type: Optional[FrequencyType]
    ) -> Callable:
        """Register a metadata loading function for a data source and frequency type.

        If frequency_type is None, the function is registered as a static data source.
        """

        def decorator(func: Callable[..., DataFrame[T]]) -> Callable[..., DataFrame[T]]:
            if data_source not in cls._registry:
                cls._registry[data_source] = {}
            if frequency_type is not None:
                cls._registry[data_source][frequency_type] = func
            else:
                cls._registry[data_source] = func
            return func

        return decorator

    @classmethod
    def load_and_validate(
        cls, data_source: str, frequency_type: Optional[FrequencyType], **kwargs: Any
    ) -> DataFrame[T]:
        """Load and validate metadata for a given data source and frequency type."""
        if data_source not in cls._registry:
            raise ValueError(f"Unknown data source: {data_source}")
        if frequency_type is None:
            func = cls._registry[data_source]
        elif frequency_type not in cls._registry[data_source]:
            raise ValueError(
                f"Unknown frequency type {frequency_type} for data source {data_source}"
            )
        else:
            func = cls._registry[data_source][frequency_type]

        # Get the registered function and call it with the provided kwargs
        return func(**kwargs)


@DataSourceMetadataRegistry.register("sentinel2", "monthly")
@pa.check_types
def load_sentinel2_monthly_metadata(
    path: Path | str, **kwargs: Any
) -> DataFrame[Sentinel2MonthlyMetadataDataModel]:
    """Load the Sentinel-2 Monthly metadata from a csv file."""
    return load_metadata(path, Sentinel2MonthlyMetadataDataModel, **kwargs)


@DataSourceMetadataRegistry.register("sentinel2", "freq")
@pa.check_types
def load_sentinel2_frequency_metadata(
    path: Path | str, **kwargs: Any
) -> DataFrame[Sentinel2FrequencyMetadataDataModel]:
    """Load the Sentinel-2 Frequency metadata from a csv file."""
    return load_metadata(path, Sentinel2FrequencyMetadataDataModel, **kwargs)


# maybe I want a single loading function per data source
