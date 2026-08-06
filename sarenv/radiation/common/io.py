"""File I/O for independent point sources and local surface simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..point.config import PointSourceConfig
from ..point.source import PointSource
from ..surface.dose_patch import DoseRatePatch
from ..surface.simulator import SurfaceSimulationResult
from .grid import GridSpec
from .metadata import read_metadata, write_metadata


NOMINAL_FIELD_FILENAME = "nominal_surface_field.npy"
NOMINAL_METADATA_FILENAME = "nominal_surface_meta.json"
DOSE_PATCH_FILENAME = "dose_rate_patch.npy"
DOSE_METADATA_FILENAME = "dose_rate_meta.json"
POINT_METADATA_FILENAME = "point_source_meta.json"


@dataclass(frozen=True)
class SavedSurfacePaths:
    """Paths written for one independent surface-source simulation."""

    nominal_field: Path
    nominal_metadata: Path
    dose_rate_patch: Path
    dose_rate_metadata: Path


@dataclass(frozen=True)
class LoadedSurfaceSimulation:
    """Arrays, grids, and metadata restored from a saved simulation."""

    nominal_surface_field: np.ndarray
    nominal_grid: GridSpec
    dose_rate_patch: DoseRatePatch
    nominal_metadata: dict[str, object]
    dose_rate_metadata: dict[str, object]


def save_surface_simulation(
    result: SurfaceSimulationResult,
    output_directory: str | Path,
) -> SavedSurfacePaths:
    """Save nominal input and excess output as separate local raster products."""
    if not isinstance(result, SurfaceSimulationResult):
        raise TypeError("result must be a SurfaceSimulationResult.")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = SavedSurfacePaths(
        nominal_field=directory / NOMINAL_FIELD_FILENAME,
        nominal_metadata=directory / NOMINAL_METADATA_FILENAME,
        dose_rate_patch=directory / DOSE_PATCH_FILENAME,
        dose_rate_metadata=directory / DOSE_METADATA_FILENAME,
    )
    np.save(paths.nominal_field, result.nominal_surface_field)
    np.save(paths.dose_rate_patch, result.dose_rate_patch.excess_uSv_h)
    write_metadata(paths.nominal_metadata, result.nominal_metadata)
    write_metadata(paths.dose_rate_metadata, result.dose_rate_metadata)
    return paths


def load_surface_simulation(
    input_directory: str | Path,
) -> LoadedSurfaceSimulation:
    """Load and validate an independent local surface-source simulation."""
    directory = Path(input_directory)
    nominal = np.load(directory / NOMINAL_FIELD_FILENAME, allow_pickle=False)
    excess = np.load(directory / DOSE_PATCH_FILENAME, allow_pickle=False)
    nominal_metadata = read_metadata(directory / NOMINAL_METADATA_FILENAME)
    dose_metadata = read_metadata(directory / DOSE_METADATA_FILENAME)
    nominal_grid_value = nominal_metadata.get("grid")
    output_grid_value = dose_metadata.get("output_grid")
    if not isinstance(nominal_grid_value, dict) or not isinstance(
        output_grid_value, dict
    ):
        raise ValueError("Radiation metadata is missing a GridSpec definition.")
    nominal_grid = GridSpec.from_metadata(nominal_grid_value)
    output_grid = GridSpec.from_metadata(output_grid_value)
    if nominal.shape != nominal_grid.shape:
        raise ValueError("Saved nominal array shape does not match its metadata.")
    if not np.isfinite(nominal).all() or (nominal < 0).any():
        raise ValueError("Saved nominal surface field is invalid.")
    patch = DoseRatePatch(
        excess_uSv_h=excess,
        grid=output_grid,
        metadata=dose_metadata,
    )
    return LoadedSurfaceSimulation(
        nominal_surface_field=nominal,
        nominal_grid=nominal_grid,
        dose_rate_patch=patch,
        nominal_metadata=nominal_metadata,
        dose_rate_metadata=dose_metadata,
    )


def save_point_source(
    source: PointSource,
    output_path: str | Path,
) -> Path:
    """Save one analytic point-source definition without rasterising it."""
    if not isinstance(source, PointSource):
        raise TypeError("source must be a PointSource.")
    path = Path(output_path)
    if path.is_dir():
        path = path / POINT_METADATA_FILENAME
    write_metadata(path, source.to_metadata())
    return path


def load_point_source(input_path: str | Path) -> PointSource:
    """Restore one analytic point source from its metadata JSON file."""
    path = Path(input_path)
    if path.is_dir():
        path = path / POINT_METADATA_FILENAME
    metadata = read_metadata(path)
    fields = {
        name: metadata[name]
        for name in PointSourceConfig.__dataclass_fields__
        if name in metadata
    }
    missing = {
        "source_id",
        "x_m",
        "y_m",
        "reference_excess_uSv_h",
        "crs",
    } - fields.keys()
    if missing:
        raise ValueError(
            "Point-source metadata is missing required fields: "
            + ", ".join(sorted(missing))
        )
    return PointSource(PointSourceConfig(**fields))


__all__ = [
    "DOSE_METADATA_FILENAME",
    "DOSE_PATCH_FILENAME",
    "LoadedSurfaceSimulation",
    "NOMINAL_FIELD_FILENAME",
    "NOMINAL_METADATA_FILENAME",
    "POINT_METADATA_FILENAME",
    "SavedSurfacePaths",
    "load_point_source",
    "load_surface_simulation",
    "save_point_source",
    "save_surface_simulation",
]
