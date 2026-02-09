from dataclasses import dataclass
from enum import IntEnum

BASE_RESOLUTION = 0.625
BASE_GSD = 10


class MaskValue(IntEnum):
    ONLINE_ENCODER = 0
    TARGET_ENCODER_ONLY = 1
    DECODER = 2
    MISSING = 3


@dataclass(frozen=True)
class BandSet:
    bands: list[str]
    resolution_factor: int

    def __hash__(self) -> int:
        return hash((tuple(self.bands), self.resolution_factor))

    def get_resolution(self) -> float:
        resolution = BASE_RESOLUTION * self.resolution_factor
        return int(resolution) if float(int(resolution)) == resolution else resolution


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    tile_resolution_factor: int
    band_sets: list[BandSet]
    is_multitemporal: bool
    image_tile_size_factor: int = 1

    def __hash__(self) -> int:
        return hash(self.name)

    def get_tile_resolution(self) -> float:
        resolution = BASE_RESOLUTION * self.tile_resolution_factor
        return int(resolution) if float(int(resolution)) == resolution else resolution

    def bandsets_as_indices(self) -> list[list[int]]:
        indices = []
        offset = 0
        for band_set in self.band_sets:
            num_bands = len(band_set.bands)
            indices.append(list(range(offset, offset + num_bands)))
            offset += num_bands
        return indices

    @property
    def band_order(self) -> list[str]:
        return sum((list(bs.bands) for bs in self.band_sets), [])

    @property
    def num_band_sets(self) -> int:
        return len(self.band_sets)

    @property
    def num_bands(self) -> int:
        return sum(len(bs.bands) for bs in self.band_sets)

    @property
    def is_spatial(self) -> bool:
        return self.get_tile_resolution() > 0


class Modality:
    SENTINEL2_L2A = ModalitySpec(
        name="sentinel2_l2a",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(["B02", "B03", "B04", "B08"], 16),
            BandSet(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            BandSet(["B01", "B09"], 64),
        ],
        is_multitemporal=True,
    )

    SENTINEL1 = ModalitySpec(
        name="sentinel1",
        tile_resolution_factor=16,
        band_sets=[BandSet(["vv", "vh"], 16)],
        is_multitemporal=True,
    )

    LANDSAT = ModalitySpec(
        name="landsat",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(["B8"], 16),
            BandSet(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"], 32),
        ],
        is_multitemporal=True,
    )

    WORLDCOVER = ModalitySpec(
        name="worldcover",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
    )

    SRTM = ModalitySpec(
        name="srtm",
        tile_resolution_factor=16,
        band_sets=[BandSet(["srtm"], 16)],
        is_multitemporal=False,
    )

    OPENSTREETMAP_RASTER = ModalitySpec(
        name="openstreetmap_raster",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "aerialway_pylon",
                    "aerodrome",
                    "airstrip",
                    "amenity_fuel",
                    "building",
                    "chimney",
                    "communications_tower",
                    "crane",
                    "flagpole",
                    "fountain",
                    "generator_wind",
                    "helipad",
                    "highway",
                    "leisure",
                    "lighthouse",
                    "obelisk",
                    "observatory",
                    "parking",
                    "petroleum_well",
                    "power_plant",
                    "power_substation",
                    "power_tower",
                    "river",
                    "runway",
                    "satellite_dish",
                    "silo",
                    "storage_tank",
                    "taxiway",
                    "water_tower",
                    "works",
                ],
                4,
            )
        ],
        is_multitemporal=False,
    )

    WRI_CANOPY_HEIGHT_MAP = ModalitySpec(
        name="wri_canopy_height_map",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
    )

    CDL = ModalitySpec(
        name="cdl",
        tile_resolution_factor=16,
        band_sets=[BandSet(["cdl"], 16)],
        is_multitemporal=False,
    )

    WORLDCEREAL = ModalitySpec(
        name="worldcereal",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "tc-annual-temporarycrops-classification",
                    "tc-maize-main-irrigation-classification",
                    "tc-maize-main-maize-classification",
                    "tc-maize-second-irrigation-classification",
                    "tc-maize-second-maize-classification",
                    "tc-springcereals-springcereals-classification",
                    "tc-wintercereals-irrigation-classification",
                    "tc-wintercereals-wintercereals-classification",
                ],
                16,
            )
        ],
        is_multitemporal=False,
    )

    @classmethod
    def get(cls, name: str) -> ModalitySpec:
        modality = getattr(cls, name.upper())
        assert modality.name == name
        return modality

    @classmethod
    def values(cls) -> list[ModalitySpec]:
        return [getattr(cls, k) for k in dir(cls) if isinstance(getattr(cls, k), ModalitySpec)]


def get_modality_specs_from_names(names: list[str]) -> list[ModalitySpec]:
    return [Modality.get(name) for name in names]


# normalization statistics from computed.json: {modality: {band: (mean, std)}}
# normalization formula (std_multiplier=2): normalized = (x - (mean - 2*std)) / (4*std)
NORMALIZATION_STATS: dict[str, dict[str, tuple[float, float]]] = {
    "sentinel2_l2a": {
        "B02": (1188.94, 1859.19),
        "B03": (1407.77, 1727.74),
        "B04": (1513.06, 1740.78),
        "B08": (2755.48, 1612.26),
        "B05": (1890.99, 1754.73),
        "B06": (2483.78, 1622.12),
        "B07": (2722.73, 1621.82),
        "B8A": (2885.57, 1611.36),
        "B11": (2562.85, 1441.55),
        "B12": (1914.14, 1328.89),
        "B01": (1115.85, 1955.70),
        "B09": (3269.81, 2651.09),
    },
    "sentinel1": {
        "vv": (-11.65, 10.84),
        "vh": (-17.75, 10.22),
    },
    "landsat": {
        "B8": (10132.09, 7788.73),
        "B1": (11000.96, 7857.92),
        "B2": (10493.42, 7872.39),
        "B3": (10146.32, 7676.28),
        "B4": (10236.45, 8038.22),
        "B5": (14427.02, 9302.43),
        "B6": (12164.08, 7442.39),
        "B7": (9712.03, 6037.71),
        "B9": (4585.47, 2549.75),
        "B10": (21347.89, 10957.28),
        "B11": (19686.23, 9911.42),
    },
    "srtm": {
        "srtm": (677.02, 993.17),
    },
}
