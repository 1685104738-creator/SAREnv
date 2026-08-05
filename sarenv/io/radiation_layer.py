"""Offline generation of synthetic radiation layers for base SAREnv datasets."""

import json
from pathlib import Path
import re

import numpy as np

from ..core.radiation import (
    RadiationConfig,
    RadiationGenerationResult,
    generate_radiation_field,
)
from ..utils.geo import get_utm_epsg
from ..utils.logging_setup import get_logger


log = get_logger()
BASE_METADATA_FILENAME = "metadata.json"
FEATURES_FILENAME = "features.geojson"
HEATMAP_FILENAME = "heatmap.npy"
DEFAULT_RADIATION_SCENARIO_NAME = "radiation"
_SCENARIO_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _read_json_object(path: Path) -> dict:
    """Read one JSON object and reject other top-level JSON types."""
    with path.open("r", encoding="utf-8") as json_file:
        value = json.load(json_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _load_base_metadata(dataset_directory: Path) -> tuple[dict, Path]:
    """Load canonical base metadata, with legacy GeoJSON metadata fallback."""
    metadata_path = dataset_directory / BASE_METADATA_FILENAME
    if metadata_path.exists():
        return _read_json_object(metadata_path), metadata_path

    features_path = dataset_directory / FEATURES_FILENAME
    if not features_path.exists():
        raise FileNotFoundError(
            "Base dataset metadata was not found. Expected either "
            f"{metadata_path} or legacy metadata in {features_path}."
        )
    log.info(
        "Base metadata.json is absent; reading legacy metadata from "
        f"{features_path}."
    )
    return _read_json_object(features_path), features_path


def _get_required_metadata_value(metadata: dict, key: str):
    """Return one required metadata value with a clear corruption error."""
    if key not in metadata:
        raise KeyError(f"Base dataset metadata is missing required key '{key}'.")
    return metadata[key]


def _validate_scenario_name(scenario_name: str) -> str:
    """Validate a safe output stem that can support future named scenarios."""
    if (
        not isinstance(scenario_name, str)
        or _SCENARIO_NAME_PATTERN.fullmatch(scenario_name) is None
    ):
        raise ValueError(
            "scenario_name must start with an alphanumeric character and "
            "contain only letters, numbers, dots, underscores, or hyphens."
        )
    return scenario_name


def generate_radiation_layer(
    dataset_path: str | Path,
    radiation_config: RadiationConfig,
    *,
    scenario_name: str = DEFAULT_RADIATION_SCENARIO_NAME,
) -> RadiationGenerationResult:
    """
    Generate and save a radiation layer using an existing SAREnv master grid.

    The function is fully offline: it reads base metadata and ``heatmap.npy``
    for grid alignment, then writes ``<scenario_name>.npy`` and
    ``<scenario_name>_meta.json``. It never generates features or probability.
    """
    if not isinstance(radiation_config, RadiationConfig):
        raise TypeError("radiation_config must be a RadiationConfig instance.")

    scenario_name = _validate_scenario_name(scenario_name)
    dataset_directory = Path(dataset_path)
    if not dataset_directory.is_dir():
        raise FileNotFoundError(
            f"Base SAREnv dataset directory not found: {dataset_directory}"
        )

    heatmap_path = dataset_directory / HEATMAP_FILENAME
    if not heatmap_path.exists():
        raise FileNotFoundError(
            f"Base probability raster not found: {heatmap_path}"
        )
    heatmap = np.load(heatmap_path, allow_pickle=False)
    if heatmap.ndim != 2:
        raise ValueError(
            f"Base heatmap must be two-dimensional, found shape {heatmap.shape}."
        )

    base_metadata, metadata_source_path = _load_base_metadata(
        dataset_directory
    )
    center_point = tuple(
        float(value)
        for value in _get_required_metadata_value(
            base_metadata, "center_point"
        )
    )
    if len(center_point) != 2:
        raise ValueError(
            "Base center_point must be a (longitude, latitude) pair."
        )

    environment_type = str(
        _get_required_metadata_value(base_metadata, "environment_type")
    )
    environment_climate = str(
        _get_required_metadata_value(base_metadata, "climate")
    )
    meter_per_bin = float(
        _get_required_metadata_value(base_metadata, "meter_per_bin")
    )
    if not np.isfinite(meter_per_bin) or meter_per_bin <= 0:
        raise ValueError(
            "Base meter_per_bin must be finite and greater than zero."
        )
    base_radius_km = float(
        _get_required_metadata_value(base_metadata, "radius_km")
    )
    if not np.isfinite(base_radius_km) or base_radius_km <= 0:
        raise ValueError(
            "Base radius_km must be finite and greater than zero."
        )

    bounds_value = base_metadata.get(
        "bounds_projected", base_metadata.get("bounds")
    )
    if bounds_value is None:
        raise KeyError(
            "Base dataset metadata is missing projected raster bounds."
        )
    master_bounds = tuple(float(value) for value in bounds_value)
    if len(master_bounds) != 4 or not np.isfinite(master_bounds).all():
        raise ValueError(
            "Base projected bounds must contain four finite values."
        )
    minx, miny, maxx, maxy = master_bounds
    if maxx <= minx or maxy <= miny:
        raise ValueError("Base projected raster bounds are invalid.")

    metadata_shape = tuple(
        int(value)
        for value in base_metadata.get("raster_shape", heatmap.shape)
    )
    if metadata_shape != heatmap.shape:
        raise ValueError(
            f"Base metadata raster shape {metadata_shape} does not match "
            f"heatmap shape {heatmap.shape}."
        )

    raster_origin = str(base_metadata.get("raster_origin", "lower"))
    if raster_origin != "lower":
        raise ValueError(
            "Only base rasters with raster_origin='lower' are supported."
        )
    raster_axis_order = str(
        base_metadata.get("raster_axis_order", "row_y_column_x")
    )
    if raster_axis_order != "row_y_column_x":
        raise ValueError(
            "Only base rasters with raster_axis_order="
            "'row_y_column_x' are supported."
        )

    projected_crs = str(
        base_metadata.get(
            "projected_crs",
            get_utm_epsg(center_point[0], center_point[1]),
        )
    )
    rows, columns = heatmap.shape
    if rows == 0 or columns == 0:
        raise ValueError("Base heatmap must contain at least one raster cell.")
    xedges = np.linspace(minx, maxx, columns + 1)
    yedges = np.linspace(miny, maxy, rows + 1)
    pixel_size_x_m = float((maxx - minx) / columns)
    pixel_size_y_m = float((maxy - miny) / rows)
    resolution_tolerance_m = meter_per_bin / max(1, min(rows, columns))
    if not np.isclose(
        pixel_size_x_m,
        meter_per_bin,
        atol=resolution_tolerance_m,
        rtol=0.0,
    ) or not np.isclose(
        pixel_size_y_m,
        meter_per_bin,
        atol=resolution_tolerance_m,
        rtol=0.0,
    ):
        raise ValueError(
            "Base bounds and heatmap shape are inconsistent with "
            f"meter_per_bin={meter_per_bin}."
        )

    result = generate_radiation_field(
        radiation_config,
        center_point=center_point,
        environment_type=environment_type,
        environment_climate=environment_climate,
        meter_per_bin=meter_per_bin,
        projected_crs=projected_crs,
        xedges=xedges,
        yedges=yedges,
        master_shape=heatmap.shape,
        master_bounds=master_bounds,
    )
    if result.field.shape != heatmap.shape:
        raise ValueError(
            f"Radiation shape {result.field.shape} does not match base "
            f"heatmap shape {heatmap.shape}."
        )

    result.metadata.update(
        {
            "scenario_name": scenario_name,
            "base_metadata_source": metadata_source_path.name,
            "base_radius_km": base_radius_km,
            "raster_origin": raster_origin,
            "raster_axis_order": raster_axis_order,
            "pixel_size_x_m": pixel_size_x_m,
            "pixel_size_y_m": pixel_size_y_m,
        }
    )

    radiation_path = dataset_directory / f"{scenario_name}.npy"
    radiation_metadata_path = (
        dataset_directory / f"{scenario_name}_meta.json"
    )
    np.save(radiation_path, result.field)
    with radiation_metadata_path.open(
        "w", encoding="utf-8"
    ) as metadata_file:
        json.dump(result.metadata, metadata_file, indent=2, sort_keys=True)

    log.info(
        "Generated offline radiation scenario '%s' with shape %s from %s.",
        scenario_name,
        result.field.shape,
        metadata_source_path,
    )
    log.info(
        "Saved radiation layer to %s and metadata to %s.",
        radiation_path,
        radiation_metadata_path,
    )
    return result


__all__ = [
    "BASE_METADATA_FILENAME",
    "DEFAULT_RADIATION_SCENARIO_NAME",
    "FEATURES_FILENAME",
    "HEATMAP_FILENAME",
    "generate_radiation_layer",
]
