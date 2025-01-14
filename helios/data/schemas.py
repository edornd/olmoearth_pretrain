"""Data schemas for helios training data index and metadata files."""

import pandera as pa
from pandera import DataFrameModel
from pandera.typing import Series


class BaseDataModel(DataFrameModel):
    """Base schema for accompanying data files."""

    class Config:
        """Config for BaseDataModel."""

        coerce = True  # ensure that columns are coerced to the correct type
        strict = False  # allow extra columns


class TrainingDataIndexDataModel(BaseDataModel):
    """Schema for training data index files.

    This file contains metadata about the training data, including the example_id, projection,
    resolution, start_column, start_row, and time.
    """

    example_id: Series[str] = pa.Field(
        description="Unique identifier for the example",
        nullable=False,
    )

    projection: Series[str] = pa.Field(
        description="EPSG projection code",
        nullable=False,
    )

    resolution: Series[int] = pa.Field(
        description="Resolution in meters",
        nullable=False,
        isin=[1, 10, 250],  # Based on the values in the CSV
    )

    start_column: Series[int] = pa.Field(
        description="Starting column UTM coordinate",
        nullable=False,
    )

    start_row: Series[int] = pa.Field(
        description="Starting row UTM coordinate",
        nullable=False,
    )

    time: Series[str] = pa.Field(
        description="Timestamp in ISO format",
        nullable=False,
        regex=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}",  # ISO timestamp format
    )

    sentinel2_freq: Series[str] = pa.Field(
        description="Whether the example_id is available in the Sentinel-2 Frequency dataset",
        nullable=False,
        isin=["y", "n"],
    )

    sentinel2_monthly: Series[str] = pa.Field(
        description="Whether the example_id is available in the Sentinel-2 Monthly dataset",
        nullable=False,
        isin=["y", "n"],
    )


class Sentinel2FrequencyMetadataDataModel(BaseDataModel):
    """Schema for Sentinel-2 Frequency metadata files.

    This file contains metadata about the Sentinel-2 Frequency dataset, including the example_id,
    image_idx, start_time, and end_time.
    """

    example_id: Series[str] = pa.Field(
        description="Unique identifier for the example, name of the file",
        nullable=False,
    )

    image_idx: Series[int] = pa.Field(
        description="Index of the image on the time axis for the geotiff",
        nullable=False,
    )

    start_time: Series[str] = pa.Field(
        description="Start time of the image in ISO format",
        nullable=False,
        regex=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}",  # ISO timestamp format
    )

    end_time: Series[str] = pa.Field(
        description="End time of the image in ISO format",
        nullable=False,
        regex=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}",  # ISO timestamp format
    )


class Sentinel2MonthlyMetadataDataModel(BaseDataModel):
    """Schema for Sentinel-2 Monthly metadata files.

    This file contains metadata about the Sentinel-2 Monthly dataset, including the example_id,
    image_idx, start_time, and end_time.
    """

    example_id: Series[str] = pa.Field(
        description="Unique identifier for the example, name of the file",
        nullable=False,
    )

    image_idx: Series[int] = pa.Field(
        description="Index of the image on the time axis for the geotiff",
        nullable=False,
    )

    start_time: Series[str] = pa.Field(
        description="Start time of the image in ISO format",
        nullable=False,
        regex=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}",  # ISO timestamp format
    )

    end_time: Series[str] = pa.Field(
        description="End time of the image in ISO format",
        nullable=False,
        regex=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}",  # ISO timestamp format
    )
