"""Shared helpers for isolated, read-only OSM acquisition diagnostics."""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO

import osmnx as ox
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.ops import transform


DIAGNOSTIC_DIR = Path(__file__).resolve().parent
REPO_ROOT = DIAGNOSTIC_DIR.parent
CACHE_DIR = DIAGNOSTIC_DIR / "cache"
FEATURES_DIR = DIAGNOSTIC_DIR / "features"
SMALL_METADATA_PATH = (
    REPO_ROOT
    / "examples"
    / "sarenv_dataset"
    / "sarenv_outputs"
    / "radiation_area_01_small_20m"
    / "metadata.json"
)
SMALL_FEATURES_PATH = SMALL_METADATA_PATH.with_name("features.geojson")
FALLBACK_CENTER = (-2.66962, 51.42351)
FALLBACK_PROJECTED_CRS = "EPSG:32630"


class Tee:
    """Write text to each supplied stream."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_output(log_path: Path) -> Iterator[None]:
    """Mirror stdout and stderr to a diagnostic log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        logger_objects = [logging.getLogger()]
        logger_objects.extend(
            logger
            for logger in logging.root.manager.loggerDict.values()
            if isinstance(logger, logging.Logger)
        )
        rebound_handlers = []
        seen_handler_ids = set()
        for logger in logger_objects:
            for handler in logger.handlers:
                if (
                    isinstance(handler, logging.StreamHandler)
                    and not isinstance(handler, logging.FileHandler)
                    and id(handler) not in seen_handler_ids
                ):
                    seen_handler_ids.add(id(handler))
                    original_stream = handler.stream
                    handler.setStream(sys.stderr)
                    rebound_handlers.append((handler, original_stream))
        try:
            yield
        finally:
            for handler, original_stream in rebound_handlers:
                handler.setStream(original_stream)
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def load_small_metadata() -> tuple[dict[str, Any], str]:
    """Load authoritative Small metadata, or return an explicit fallback."""

    if SMALL_METADATA_PATH.exists():
        metadata = json.loads(SMALL_METADATA_PATH.read_text(encoding="utf-8"))
        return metadata, str(SMALL_METADATA_PATH)
    center_lon, center_lat = FALLBACK_CENTER
    to_projected = Transformer.from_crs(
        "EPSG:4326", FALLBACK_PROJECTED_CRS, always_xy=True
    )
    center_x, center_y = to_projected.transform(center_lon, center_lat)
    metadata = {
        "center_point": [center_lon, center_lat],
        "projected_crs": FALLBACK_PROJECTED_CRS,
        "bounds_projected": [
            center_x - 600.0,
            center_y - 600.0,
            center_x + 600.0,
            center_y + 600.0,
        ],
        "environment_size": "small",
        "radius_km": 0.6,
    }
    return metadata, "fallback centre + +/-600 m (metadata file not found)"


def projected_to_wgs84(
    projected_geometry: Polygon, projected_crs: str
) -> Polygon:
    transformer = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    return transform(transformer.transform, projected_geometry)


def build_tiny_polygon(size_m: float = 150.0) -> tuple[Polygon, dict[str, Any]]:
    """Build a square tiny polygon centered on the authoritative Bristol point."""

    metadata, metadata_source = load_small_metadata()
    center_lon, center_lat = metadata["center_point"]
    projected_crs = metadata["projected_crs"]
    to_projected = Transformer.from_crs(
        "EPSG:4326", projected_crs, always_xy=True
    )
    center_x, center_y = to_projected.transform(center_lon, center_lat)
    half_size = size_m / 2.0
    projected_polygon = box(
        center_x - half_size,
        center_y - half_size,
        center_x + half_size,
        center_y + half_size,
    )
    wgs84_polygon = projected_to_wgs84(projected_polygon, projected_crs)
    details = {
        "metadata_source": metadata_source,
        "center_wgs84": [center_lon, center_lat],
        "projected_crs": projected_crs,
        "size_m": size_m,
        "projected_bounds": list(projected_polygon.bounds),
        "wgs84_bounds": list(wgs84_polygon.bounds),
    }
    return wgs84_polygon, details


def build_full_small_polygon() -> tuple[Polygon, dict[str, Any]]:
    """Build the exact 1200 m square from authoritative Small metadata."""

    metadata, metadata_source = load_small_metadata()
    projected_crs = metadata["projected_crs"]
    projected_bounds = [float(value) for value in metadata["bounds_projected"]]
    projected_polygon = box(*projected_bounds)
    wgs84_polygon = projected_to_wgs84(projected_polygon, projected_crs)
    width = projected_bounds[2] - projected_bounds[0]
    height = projected_bounds[3] - projected_bounds[1]
    details = {
        "metadata_source": metadata_source,
        "center_wgs84": list(metadata["center_point"]),
        "projected_crs": projected_crs,
        "projected_bounds": projected_bounds,
        "wgs84_bounds": list(wgs84_polygon.bounds),
        "width_m": width,
        "height_m": height,
    }
    return wgs84_polygon, details


def configure_isolated_osmnx() -> dict[str, Any]:
    """Disable cache reads/writes and isolate the configured cache path."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    original = {
        "overpass_url": ox.settings.overpass_url,
        "requests_timeout": ox.settings.requests_timeout,
        "use_cache": ox.settings.use_cache,
        "cache_folder": str(ox.settings.cache_folder),
        "overpass_rate_limit": ox.settings.overpass_rate_limit,
        "overpass_settings": ox.settings.overpass_settings,
    }
    ox.settings.cache_folder = CACHE_DIR
    ox.settings.use_cache = False
    effective = {
        "overpass_url": ox.settings.overpass_url,
        "requests_timeout": ox.settings.requests_timeout,
        "use_cache": ox.settings.use_cache,
        "cache_folder": str(ox.settings.cache_folder),
        "overpass_rate_limit": ox.settings.overpass_rate_limit,
        "overpass_settings": ox.settings.overpass_settings,
    }
    return {"original": original, "effective": effective}


def walk_geometry(
    geometry: Any,
    recursive_counts: Counter[str] | None = None,
    path_counts: Counter[str] | None = None,
    parent_path: tuple[str, ...] = (),
) -> tuple[Counter[str], Counter[str]]:
    """Recursively count all geometry layers and their complete type paths."""

    if recursive_counts is None:
        recursive_counts = Counter()
    if path_counts is None:
        path_counts = Counter()
    if geometry is None:
        recursive_counts["None"] += 1
        path_counts[" -> ".join((*parent_path, "None"))] += 1
        return recursive_counts, path_counts

    geometry_type = geometry.geom_type
    recursive_counts[geometry_type] += 1
    current_path = (*parent_path, geometry_type)
    if geometry_type in {
        "GeometryCollection",
        "MultiPolygon",
        "MultiLineString",
        "MultiPoint",
    }:
        children = list(geometry.geoms)
        if children:
            for child in children:
                walk_geometry(child, recursive_counts, path_counts, current_path)
        else:
            path_counts[" -> ".join(current_path)] += 1
    else:
        path_counts[" -> ".join(current_path)] += 1
    return recursive_counts, path_counts


def summarize_geometries(geometries: Any) -> dict[str, Any]:
    """Return top-level and recursive geometry diagnostics."""

    geometry_list = list(geometries)
    top_level_counts = Counter(
        geometry.geom_type if geometry is not None else "None"
        for geometry in geometry_list
    )
    recursive_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    empty_count = 0
    invalid_count = 0
    for geometry in geometry_list:
        if geometry is None:
            walk_geometry(geometry, recursive_counts, path_counts)
            empty_count += 1
            invalid_count += 1
            continue
        empty_count += int(geometry.is_empty)
        invalid_count += int(not geometry.is_valid)
        walk_geometry(geometry, recursive_counts, path_counts)
    return {
        "top_level_geometry_count": len(geometry_list),
        "top_level_geometry_type_counts": dict(sorted(top_level_counts.items())),
        "recursive_geometry_type_counts": dict(sorted(recursive_counts.items())),
        "geometry_type_paths": dict(sorted(path_counts.items())),
        "empty_geometry_count": empty_count,
        "invalid_geometry_count": invalid_count,
        "has_MultiPolygon": recursive_counts["MultiPolygon"] > 0,
        "has_MultiLineString": recursive_counts["MultiLineString"] > 0,
        "has_GeometryCollection": recursive_counts["GeometryCollection"] > 0,
    }


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def run_logged(log_name: str, main: Callable[[], None]) -> None:
    with tee_output(DIAGNOSTIC_DIR / log_name):
        main()
