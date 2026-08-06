"""Deterministic polygon rasterisation and full FFT convolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS
from scipy.signal import fftconvolve

from ..common.grid import GridSpec
from ..common.metadata import (
    DOSE_RATE_UNIT,
    EXCESS_DOSE_RATE_QUANTITY,
    NOMINAL_SURFACE_UNIT,
    RADIATION_GENERATOR_VERSION,
)
from .dose_patch import DoseRatePatch
from .response_kernel import BenchmarkSurfaceResponseKernel
from .uniform_polygon import UniformPolygonSource
from .zoned_polygon import ZonedPolygonSource


SurfaceSource = UniformPolygonSource | ZonedPolygonSource


@dataclass(frozen=True)
class SurfaceSimulationResult:
    """Nominal input, response kernel, excess patch, and separate metadata."""

    nominal_surface_field: np.ndarray
    nominal_grid: GridSpec
    kernel: BenchmarkSurfaceResponseKernel
    dose_rate_patch: DoseRatePatch
    nominal_metadata: dict[str, object]
    dose_rate_metadata: dict[str, object]


def _output_grid_for_full_convolution(
    input_grid: GridSpec,
    kernel: BenchmarkSurfaceResponseKernel,
    output_shape: tuple[int, int],
) -> GridSpec:
    padding_m = kernel.radius_cells * input_grid.resolution_m
    bounds = (
        input_grid.minx - padding_m,
        input_grid.miny - padding_m,
        input_grid.maxx + padding_m,
        input_grid.maxy + padding_m,
    )
    return GridSpec(
        bounds=bounds,
        resolution_m=input_grid.resolution_m,
        crs=input_grid.crs,
        width=output_shape[1],
        height=output_shape[0],
    )


def simulate_surface_source(
    source: SurfaceSource,
    kernel: BenchmarkSurfaceResponseKernel,
    *,
    nominal_grid: GridSpec | None = None,
) -> SurfaceSimulationResult:
    """
    Rasterize a polygon source and compute an untruncated full FFT convolution.

    The result is a local excess-only patch. No background, randomness,
    terrain interaction, detector response, or nuclide conversion is applied.
    """
    if not isinstance(source, (UniformPolygonSource, ZonedPolygonSource)):
        raise TypeError("source must be UniformPolygonSource or ZonedPolygonSource.")
    if not isinstance(kernel, BenchmarkSurfaceResponseKernel):
        raise TypeError("kernel must be BenchmarkSurfaceResponseKernel.")
    if nominal_grid is None:
        nominal_grid = GridSpec.from_bounds(source.bounds, source.crs)
    if not CRS.from_user_input(source.crs).equals(
        CRS.from_user_input(nominal_grid.crs)
    ):
        raise ValueError("Source and nominal grid CRS do not match.")
    if not np.isclose(
        nominal_grid.resolution_m, kernel.config.resolution_m
    ):
        raise ValueError("Nominal grid and kernel resolutions do not match.")

    nominal_field = source.rasterize(nominal_grid)
    if nominal_field.shape != nominal_grid.shape:
        raise ValueError("Rasterized nominal field shape is invalid.")
    if not np.isfinite(nominal_field).all() or (nominal_field < 0).any():
        raise ValueError("Nominal field must be finite and non-negative.")

    excess = fftconvolve(nominal_field, kernel.values, mode="full")
    numerical_tolerance = max(1.0, float(nominal_field.max())) * 1e-12
    if float(excess.min()) < -numerical_tolerance:
        raise ValueError("FFT convolution produced material negative values.")
    excess = np.maximum(excess, 0.0)
    output_grid = _output_grid_for_full_convolution(
        nominal_grid, kernel, excess.shape
    )

    source_metadata = source.source_metadata()
    nominal_metadata = {
        "schema_version": 1,
        "model_type": "synthetic_nominal_surface_field",
        "surface_type": source.surface_type,
        "quantity": "synthetic_nominal_surface_field",
        "nominal_intensity_unit": NOMINAL_SURFACE_UNIT,
        "background_included": False,
        "generator_version": RADIATION_GENERATOR_VERSION,
        "grid": nominal_grid.to_metadata(),
        "input_bounds": list(nominal_grid.bounds),
        "input_shape": list(nominal_grid.shape),
        **source_metadata,
    }
    dose_metadata = {
        "schema_version": 1,
        "model_type": "benchmark_surface_response",
        "surface_type": source.surface_type,
        "quantity": EXCESS_DOSE_RATE_QUANTITY,
        "dose_output_unit": DOSE_RATE_UNIT,
        "background_included": False,
        "data_content": "source_excess_only",
        "generator_version": RADIATION_GENERATOR_VERSION,
        "crs": nominal_grid.crs,
        "resolution_m": nominal_grid.resolution_m,
        "origin": nominal_grid.origin,
        "axis_order": nominal_grid.axis_order,
        "input_bounds": list(nominal_grid.bounds),
        "output_bounds": list(output_grid.bounds),
        "input_shape": list(nominal_grid.shape),
        "output_shape": list(output_grid.shape),
        "output_grid": output_grid.to_metadata(),
        "convolution_mode": "full",
        "convolution_implementation": "scipy.signal.fftconvolve",
        **kernel.to_metadata(),
        "source": source_metadata,
    }
    patch = DoseRatePatch(
        excess_uSv_h=excess,
        grid=output_grid,
        metadata=dose_metadata,
    )
    return SurfaceSimulationResult(
        nominal_surface_field=nominal_field,
        nominal_grid=nominal_grid,
        kernel=kernel,
        dose_rate_patch=patch,
        nominal_metadata=nominal_metadata,
        dose_rate_metadata=dose_metadata,
    )


__all__ = ["SurfaceSimulationResult", "simulate_surface_source"]
