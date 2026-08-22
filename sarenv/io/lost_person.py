"""Save and load projected lost-person locations with their SAR contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from pyproj import CRS
from shapely.geometry import Point

from ..core.loading import SARDatasetItem
from ..utils.lost_person_behavior import get_available_sizes


LOST_PERSON_FILENAME = "lost_persons.json"
LOST_PERSON_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LostPersonLocations:
    """Projected lost-person points restored from one SAR scenario."""

    points: tuple[Point, ...]
    projected_crs: str
    environment_size: str
    bounds: tuple[float, float, float, float]
    center_point_wgs84: tuple[float, float]
    radius_km: float
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.environment_size not in get_available_sizes():
            raise ValueError("Unsupported lost-person environment_size.")
        if not CRS.from_user_input(self.projected_crs).is_projected:
            raise ValueError("Lost-person coordinates require a projected CRS.")
        if len(self.bounds) != 4 or not np.isfinite(self.bounds).all():
            raise ValueError("Lost-person bounds must contain four finite values.")
        minx, miny, maxx, maxy = self.bounds
        if maxx <= minx or maxy <= miny:
            raise ValueError("Lost-person bounds are invalid.")
        for point in self.points:
            if not isinstance(point, Point) or point.is_empty:
                raise TypeError("Lost-person locations must be non-empty Points.")
            if not (minx <= point.x <= maxx and miny <= point.y <= maxy):
                raise ValueError("A lost-person point lies outside the SAR bounds.")


def save_lost_person_locations(
    points: list[Point] | tuple[Point, ...],
    dataset_item: SARDatasetItem,
    output_path: str | Path,
) -> Path:
    """Save exact projected coordinates and their authoritative SAR metadata."""
    if not isinstance(dataset_item, SARDatasetItem):
        raise TypeError("dataset_item must be a SARDatasetItem.")
    projected_crs = dataset_item.projected_crs or str(dataset_item.features.crs)
    point_tuple = tuple(points)
    metadata = {
        "schema_version": LOST_PERSON_SCHEMA_VERSION,
        "dataset_type": "sarenv_lost_person_locations",
        "quantity": "projected_lost_person_coordinates",
        "coordinate_unit": "m",
        "projected_crs": projected_crs,
        "environment_size": dataset_item.size,
        "bounds_projected": [float(value) for value in dataset_item.bounds],
        "center_point_wgs84": [
            float(value) for value in dataset_item.center_point
        ],
        "radius_km": float(dataset_item.radius_km),
        "count": len(point_tuple),
        "coordinates_projected": [
            [float(point.x), float(point.y)] for point in point_tuple
        ],
    }
    LostPersonLocations(
        points=point_tuple,
        projected_crs=projected_crs,
        environment_size=dataset_item.size,
        bounds=tuple(float(value) for value in dataset_item.bounds),
        center_point_wgs84=tuple(
            float(value) for value in dataset_item.center_point
        ),
        radius_km=float(dataset_item.radius_km),
        metadata=metadata,
    )

    path = Path(output_path)
    if path.suffix.lower() != ".json":
        path = path / LOST_PERSON_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def load_lost_person_locations(input_path: str | Path) -> LostPersonLocations:
    """Load lost-person coordinates and reject incompatible metadata."""
    path = Path(input_path)
    if path.is_dir():
        path = path / LOST_PERSON_FILENAME
    with path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("Lost-person file must contain a JSON object.")
    if metadata.get("schema_version") != LOST_PERSON_SCHEMA_VERSION:
        raise ValueError("Unsupported lost-person schema version.")
    if metadata.get("dataset_type") != "sarenv_lost_person_locations":
        raise ValueError("Lost-person dataset_type is incompatible.")
    if metadata.get("quantity") != "projected_lost_person_coordinates":
        raise ValueError("Lost-person coordinate quantity is incompatible.")
    if metadata.get("coordinate_unit") != "m":
        raise ValueError("Lost-person coordinate unit must be 'm'.")
    coordinates = metadata.get("coordinates_projected")
    if not isinstance(coordinates, list):
        raise ValueError("Lost-person coordinates_projected must be a list.")
    if metadata.get("count") != len(coordinates):
        raise ValueError("Lost-person count does not match saved coordinates.")
    return LostPersonLocations(
        points=tuple(Point(float(x), float(y)) for x, y in coordinates),
        projected_crs=str(metadata["projected_crs"]),
        environment_size=str(metadata["environment_size"]),
        bounds=tuple(float(value) for value in metadata["bounds_projected"]),
        center_point_wgs84=tuple(
            float(value) for value in metadata["center_point_wgs84"]
        ),
        radius_km=float(metadata["radius_km"]),
        metadata=metadata,
    )


__all__ = [
    "LOST_PERSON_FILENAME",
    "LOST_PERSON_SCHEMA_VERSION",
    "LostPersonLocations",
    "load_lost_person_locations",
    "save_lost_person_locations",
]
