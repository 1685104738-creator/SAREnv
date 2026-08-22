"""File I/O for independent point sources and local surface simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..point.config import PointSourceConfig
from ..point.source import PointSource
from ..surface.config import (
    CELL_ACTIVITY_UNIT,
    COLLISION_AIR_KERMA_RATE_UNIT,
    PHOTON_FLUENCE_RATE_UNIT,
    SURFACE_ACTIVITY_DENSITY_UNIT,
)
from ..surface.dose_patch import (
    CollisionAirKermaRatePatch,
    PhotonFluenceRatePatch,
)
from ..surface.simulator import SurfaceSimulationResult
from .grid import GridSpec
from .metadata import read_metadata, write_metadata


ACTIVITY_DENSITY_FILENAME = "activity_density_bq_m2.npy"
CELL_ACTIVITY_FILENAME = "cell_activity_bq.npy"
PHOTON_FLUENCE_FILENAME = "photon_fluence_rate_ph_m2_s.npy"
COLLISION_AIR_KERMA_FILENAME = "collision_air_kerma_rate_uGy_h.npy"
SURFACE_METADATA_FILENAME = "surface_metadata.json"
POINT_METADATA_FILENAME = "point_source_meta.json"
SURFACE_IO_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SavedSurfacePaths:
    """Physical surface-radiation files written by the current saver."""

    activity_density: Path
    cell_activity: Path
    photon_fluence_rate: Path
    collision_air_kerma_rate: Path
    metadata: Path


@dataclass(frozen=True)
class LoadedSurfaceSimulation:
    """Current physical surface fields restored without unit conversion."""

    activity_density_bq_m2: np.ndarray
    cell_activity_bq: np.ndarray
    grid: GridSpec
    photon_fluence_rate_patch: PhotonFluenceRatePatch
    collision_air_kerma_rate_patch: CollisionAirKermaRatePatch
    activity_metadata: dict[str, object]
    photon_fluence_metadata: dict[str, object]
    collision_air_kerma_metadata: dict[str, object]
    metadata: dict[str, object]


def _surface_contract(result: SurfaceSimulationResult) -> dict[str, object]:
    """Build the explicit quantity/unit/file contract for one surface run."""
    return {
        "schema_version": SURFACE_IO_SCHEMA_VERSION,
        "dataset_type": "physical_surface_radiation",
        "scenario_type": "surface_only",
        "background_included": False,
        "grid": result.grid.to_metadata(),
        "files": {
            "activity_density_bq_m2": ACTIVITY_DENSITY_FILENAME,
            "cell_activity_bq": CELL_ACTIVITY_FILENAME,
            "photon_fluence_rate_ph_m2_s": PHOTON_FLUENCE_FILENAME,
            "collision_air_kerma_rate_uGy_h": COLLISION_AIR_KERMA_FILENAME,
        },
        "quantities": {
            "activity_density_bq_m2": {
                "quantity": "surface_activity_density",
                "unit": SURFACE_ACTIVITY_DENSITY_UNIT,
            },
            "cell_activity_bq": {
                "quantity": "cell_activity",
                "unit": CELL_ACTIVITY_UNIT,
            },
            "photon_fluence_rate_ph_m2_s": {
                "quantity": "photon_fluence_rate",
                "unit": PHOTON_FLUENCE_RATE_UNIT,
            },
            "collision_air_kerma_rate_uGy_h": {
                "quantity": "collision_air_kerma_rate",
                "unit": COLLISION_AIR_KERMA_RATE_UNIT,
            },
        },
        "activity_metadata": result.activity_metadata,
        "photon_fluence_metadata": result.photon_fluence_metadata,
        "collision_air_kerma_metadata": result.collision_air_kerma_metadata,
        "minimum_raw_fft_value_before_clipping": (
            result.minimum_raw_fft_value_before_clipping
        ),
    }


def save_surface_simulation(
    result: SurfaceSimulationResult,
    output_directory: str | Path,
) -> SavedSurfacePaths:
    """Save current physical surface fields with one authoritative contract."""
    if not isinstance(result, SurfaceSimulationResult):
        raise TypeError("result must be a SurfaceSimulationResult.")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = SavedSurfacePaths(
        activity_density=directory / ACTIVITY_DENSITY_FILENAME,
        cell_activity=directory / CELL_ACTIVITY_FILENAME,
        photon_fluence_rate=directory / PHOTON_FLUENCE_FILENAME,
        collision_air_kerma_rate=directory / COLLISION_AIR_KERMA_FILENAME,
        metadata=directory / SURFACE_METADATA_FILENAME,
    )
    np.save(paths.activity_density, result.activity_density_bq_m2)
    np.save(paths.cell_activity, result.cell_activity_bq)
    np.save(
        paths.photon_fluence_rate,
        result.photon_fluence_rate_patch.photon_fluence_rate,
    )
    np.save(
        paths.collision_air_kerma_rate,
        result.collision_air_kerma_rate_patch.collision_air_kerma_rate_uGy_h,
    )
    write_metadata(paths.metadata, _surface_contract(result))
    return paths


def _require_quantity_contract(
    metadata: dict[str, object],
    key: str,
    expected_quantity: str,
    expected_unit: str,
) -> None:
    """Reject missing or semantically incompatible saved quantities."""
    quantities = metadata.get("quantities")
    if not isinstance(quantities, dict) or not isinstance(
        quantities.get(key), dict
    ):
        raise ValueError(f"Surface metadata is missing quantity contract {key!r}.")
    contract = quantities[key]
    if contract.get("quantity") != expected_quantity:
        raise ValueError(
            f"Saved {key} quantity is not {expected_quantity!r}."
        )
    if contract.get("unit") != expected_unit:
        raise ValueError(
            f"Saved {key} unit is not {expected_unit!r}."
        )


def _load_surface_array(directory: Path, filename: str, grid: GridSpec) -> np.ndarray:
    """Load one physical surface array and validate its common grid."""
    array = np.asarray(
        np.load(directory / filename, allow_pickle=False),
        dtype=np.float64,
    )
    if array.shape != grid.shape:
        raise ValueError(f"Saved array {filename!r} does not match GridSpec shape.")
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"Saved array {filename!r} must be finite and non-negative.")
    return array


def load_surface_simulation(
    input_directory: str | Path,
) -> LoadedSurfaceSimulation:
    """Load current physical surface fields without silent legacy conversion."""
    directory = Path(input_directory)
    metadata_path = directory / SURFACE_METADATA_FILENAME
    if not metadata_path.exists():
        raise ValueError(
            "surface_metadata.json is required for the current physical "
            "surface format; legacy nominal/dose files are not interpreted "
            "as activity, photon fluence, or collision air kerma."
        )
    metadata = read_metadata(metadata_path)
    if metadata.get("schema_version") != SURFACE_IO_SCHEMA_VERSION:
        raise ValueError("Unsupported physical surface I/O schema version.")
    if metadata.get("dataset_type") != "physical_surface_radiation":
        raise ValueError("Surface metadata dataset_type is incompatible.")
    if metadata.get("scenario_type") != "surface_only":
        raise ValueError("Surface metadata scenario_type must be 'surface_only'.")
    if metadata.get("background_included") is not False:
        raise ValueError("Physical surface files must exclude background.")
    grid_value = metadata.get("grid")
    if not isinstance(grid_value, dict):
        raise ValueError("Surface metadata is missing its GridSpec definition.")
    grid = GridSpec.from_metadata(grid_value)

    expected_contracts = {
        "activity_density_bq_m2": (
            "surface_activity_density",
            SURFACE_ACTIVITY_DENSITY_UNIT,
            ACTIVITY_DENSITY_FILENAME,
        ),
        "cell_activity_bq": (
            "cell_activity",
            CELL_ACTIVITY_UNIT,
            CELL_ACTIVITY_FILENAME,
        ),
        "photon_fluence_rate_ph_m2_s": (
            "photon_fluence_rate",
            PHOTON_FLUENCE_RATE_UNIT,
            PHOTON_FLUENCE_FILENAME,
        ),
        "collision_air_kerma_rate_uGy_h": (
            "collision_air_kerma_rate",
            COLLISION_AIR_KERMA_RATE_UNIT,
            COLLISION_AIR_KERMA_FILENAME,
        ),
    }
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError("Surface metadata is missing its files contract.")
    for key, (quantity, unit, filename) in expected_contracts.items():
        _require_quantity_contract(metadata, key, quantity, unit)
        if files.get(key) != filename:
            raise ValueError(f"Surface metadata filename for {key!r} is invalid.")

    activity_density = _load_surface_array(
        directory, ACTIVITY_DENSITY_FILENAME, grid
    )
    cell_activity = _load_surface_array(directory, CELL_ACTIVITY_FILENAME, grid)
    photon_fluence = _load_surface_array(
        directory, PHOTON_FLUENCE_FILENAME, grid
    )
    collision_air_kerma = _load_surface_array(
        directory, COLLISION_AIR_KERMA_FILENAME, grid
    )

    activity_metadata = metadata.get("activity_metadata")
    photon_metadata = metadata.get("photon_fluence_metadata")
    kerma_metadata = metadata.get("collision_air_kerma_metadata")
    if not all(
        isinstance(value, dict)
        for value in (activity_metadata, photon_metadata, kerma_metadata)
    ):
        raise ValueError("Surface metadata is missing physical model metadata.")
    if activity_metadata.get("quantity") != "surface_activity_density" or (
        activity_metadata.get("activity_density_unit")
        != SURFACE_ACTIVITY_DENSITY_UNIT
    ):
        raise ValueError("Activity-density model metadata is incompatible.")
    if photon_metadata.get("quantity") != "photon_fluence_rate" or (
        photon_metadata.get("output_unit") != PHOTON_FLUENCE_RATE_UNIT
    ):
        raise ValueError("Photon-fluence model metadata is incompatible.")
    if kerma_metadata.get("quantity") != "collision_air_kerma_rate" or (
        kerma_metadata.get("output_unit")
        != COLLISION_AIR_KERMA_RATE_UNIT
    ):
        raise ValueError("Collision-air-kerma model metadata is incompatible.")
    photon_patch = PhotonFluenceRatePatch(
        photon_fluence_rate=photon_fluence,
        grid=grid,
        metadata=photon_metadata,
    )
    kerma_patch = CollisionAirKermaRatePatch(
        collision_air_kerma_rate_uGy_h=collision_air_kerma,
        grid=grid,
        metadata=kerma_metadata,
    )
    return LoadedSurfaceSimulation(
        activity_density_bq_m2=activity_density,
        cell_activity_bq=cell_activity,
        grid=grid,
        photon_fluence_rate_patch=photon_patch,
        collision_air_kerma_rate_patch=kerma_patch,
        activity_metadata=activity_metadata,
        photon_fluence_metadata=photon_metadata,
        collision_air_kerma_metadata=kerma_metadata,
        metadata=metadata,
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
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported point-source metadata schema version.")
    if metadata.get("model_type") != "analytic_point_source":
        raise ValueError("Point-source model_type is incompatible.")
    if metadata.get("scenario_type") != "point_only":
        raise ValueError("Point-source scenario_type must be 'point_only'.")
    if metadata.get("quantity") != "synthetic_excess_gamma_dose_rate":
        raise ValueError("Point-source quantity metadata is incompatible.")
    if metadata.get("background_included") is not False:
        raise ValueError("Point-source data must exclude background.")
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
    "ACTIVITY_DENSITY_FILENAME",
    "CELL_ACTIVITY_FILENAME",
    "COLLISION_AIR_KERMA_FILENAME",
    "LoadedSurfaceSimulation",
    "PHOTON_FLUENCE_FILENAME",
    "POINT_METADATA_FILENAME",
    "SavedSurfacePaths",
    "SURFACE_IO_SCHEMA_VERSION",
    "SURFACE_METADATA_FILENAME",
    "load_point_source",
    "load_surface_simulation",
    "save_point_source",
    "save_surface_simulation",
]
