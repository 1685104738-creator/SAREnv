"""Lightweight shared geography for radiation alignment and visualisation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from ...utils.geo import get_utm_epsg
from .grid import GridSpec, validate_projected_metre_crs


@dataclass(frozen=True)
class TerrainContext:
    """
    Existing SAREnv geography shared with radiation without sharing rasters.

    Features are projected for alignment and visualisation only. They do not
    attenuate or otherwise modify radiation in this implementation stage.
    """

    features: gpd.GeoDataFrame
    projected_crs: str
    bounds: tuple[float, float, float, float]
    center_point_wgs84: tuple[float, float]
    dataset_directory: Path

    @classmethod
    def from_dataset(cls, dataset_path: str | Path) -> "TerrainContext":
        """Load features and public spatial metadata without reading heatmap.npy."""
        dataset_directory = Path(dataset_path)
        features_path = dataset_directory / "features.geojson"
        if not features_path.exists():
            raise FileNotFoundError(f"features.geojson not found: {features_path}")

        with features_path.open("r", encoding="utf-8") as stream:
            geojson_data = json.load(stream)
        if not isinstance(geojson_data, dict):
            raise ValueError("features.geojson must contain a JSON object.")

        metadata_path = dataset_directory / "metadata.json"
        metadata: dict[str, object] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            if not isinstance(metadata, dict):
                raise ValueError("metadata.json must contain a JSON object.")

        center_value = metadata.get(
            "center_point", geojson_data.get("center_point")
        )
        if center_value is None or len(center_value) != 2:
            raise KeyError("Dataset metadata is missing center_point.")
        center_point = tuple(float(value) for value in center_value)

        projected_crs = str(
            metadata.get(
                "projected_crs",
                get_utm_epsg(center_point[0], center_point[1]),
            )
        )
        validate_projected_metre_crs(projected_crs)

        bounds_value = metadata.get(
            "bounds_projected", geojson_data.get("bounds")
        )
        if bounds_value is None:
            raise KeyError("Dataset metadata is missing projected bounds.")
        bounds = tuple(float(value) for value in bounds_value)
        if len(bounds) != 4 or not np.isfinite(bounds).all():
            raise ValueError("Terrain bounds must contain four finite values.")
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError("Terrain bounds are invalid.")

        feature_values = geojson_data.get("features", [])
        if not isinstance(feature_values, list):
            raise ValueError("features.geojson 'features' must be a list.")
        if feature_values:
            features = gpd.GeoDataFrame.from_features(
                feature_values, crs="EPSG:4326"
            )
        else:
            features = gpd.GeoDataFrame(
                {"geometry": []}, geometry="geometry", crs="EPSG:4326"
            )
        features = features.to_crs(projected_crs)
        return cls(
            features=features,
            projected_crs=projected_crs,
            bounds=bounds,
            center_point_wgs84=center_point,
            dataset_directory=dataset_directory,
        )

    def contains_geometry(self, geometry: BaseGeometry) -> bool:
        """Return whether a projected geometry lies within the terrain bounds."""
        if geometry is None or geometry.is_empty:
            return False
        minx, miny, maxx, maxy = geometry.bounds
        return (
            self.bounds[0] <= minx
            and self.bounds[1] <= miny
            and maxx <= self.bounds[2]
            and maxy <= self.bounds[3]
        )

    def validate_grid(self, grid: GridSpec) -> None:
        """Raise if a radiation grid is outside or in a different CRS."""
        if not CRS.from_user_input(grid.crs).equals(
            CRS.from_user_input(self.projected_crs)
        ):
            raise ValueError("Radiation grid CRS does not match TerrainContext CRS.")
        minx, miny, maxx, maxy = grid.bounds
        if not (
            self.bounds[0] <= minx
            and self.bounds[1] <= miny
            and maxx <= self.bounds[2]
            and maxy <= self.bounds[3]
        ):
            raise ValueError("Radiation grid lies outside the shared map bounds.")


__all__ = ["TerrainContext"]
