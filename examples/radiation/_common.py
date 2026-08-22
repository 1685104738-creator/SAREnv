"""Shared plotting utilities for independent surface-radiation examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from sarenv.radiation import (
    GridSpec,
    SurfacePhotonResponseConfig,
    SurfacePhotonResponseKernel,
    save_surface_simulation,
    simulate_surface_source,
)


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = EXAMPLE_DIRECTORY / "outputs"


def _local_imshow_extent(bounds):
    """Plot projected bounds as readable local x/y offsets in metres."""
    minx, miny, maxx, maxy = bounds
    return 0.0, maxx - minx, 0.0, maxy - miny


def _plot_source_geometry(axis, source, grid) -> None:
    if source.surface_type == "uniform_polygon":
        geometries = [source.config.geometry]
        intensities = [source.config.activity_density_bq_m2]
    else:
        geometries = [zone.geometry for zone in source.config.zones]
        intensities = [
            zone.activity_density_bq_m2 for zone in source.config.zones
        ]
    colours = plt.cm.viridis(
        np.linspace(0.25, 0.9, max(1, len(geometries)))
    )
    origin_x, origin_y = grid.minx, grid.miny
    for geometry, intensity, colour in zip(
        geometries, intensities, colours, strict=True
    ):
        polygons = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
        for polygon in polygons:
            x, y = polygon.exterior.xy
            axis.fill(
                np.asarray(x) - origin_x,
                np.asarray(y) - origin_y,
                color=colour,
                alpha=0.6,
                label=f"{intensity:g} Bq/m^2",
            )
    axis.set_aspect("equal")
    axis.set_title("Input polygon / zones")
    axis.legend(loc="best", fontsize=8)


def run_surface_example(
    name,
    source,
    *,
    padding_m=25.0,
):
    """Simulate and visualise one current Cs-137 surface source offline."""
    grid = GridSpec.from_bounds(source.bounds, source.crs, padding_m=padding_m)
    kernel = SurfacePhotonResponseKernel.create(
        SurfacePhotonResponseConfig(), grid
    )
    result = simulate_surface_source(source, kernel, nominal_grid=grid)
    output_directory = OUTPUT_DIRECTORY / name
    paths = save_surface_simulation(result, output_directory)

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    _plot_source_geometry(axes[0, 0], source, grid)

    activity_image = axes[0, 1].imshow(
        result.activity_density_bq_m2,
        origin="lower",
        extent=_local_imshow_extent(result.grid.bounds),
        cmap="viridis",
    )
    axes[0, 1].set_title("1 m Cs-137 activity-density raster")
    figure.colorbar(activity_image, ax=axes[0, 1], label="Activity density (Bq/m^2)")

    centre_row, centre_col = kernel.center_index
    x_radius = kernel.values.shape[1] // 2
    y_radius = kernel.values.shape[0] // 2
    kernel_image = axes[0, 2].imshow(
        kernel.values,
        origin="lower",
        extent=(-x_radius - 0.5, x_radius + 0.5, -y_radius - 0.5, y_radius + 0.5),
        cmap="magma",
        norm=LogNorm(
            vmin=max(float(kernel.values.min()), 1e-30),
            vmax=float(kernel.values[centre_row, centre_col]),
        ),
    )
    axes[0, 2].set_title("Cs-137 primary-photon kernel")
    axes[0, 2].set_xlabel("x offset (m)")
    axes[0, 2].set_ylabel("y offset (m)")
    figure.colorbar(
        kernel_image,
        ax=axes[0, 2],
        label="photons / m^2 / decay",
    )

    fluence = result.photon_fluence_rate_patch.photon_fluence_rate
    fluence_image = axes[1, 0].imshow(
        fluence,
        origin="lower",
        extent=_local_imshow_extent(result.grid.bounds),
        cmap="inferno",
    )
    axes[1, 0].set_title("Photon fluence rate")
    figure.colorbar(
        fluence_image,
        ax=axes[1, 0],
        label="photons / m^2 / s",
    )

    kerma = (
        result.collision_air_kerma_rate_patch.collision_air_kerma_rate_uGy_h
    )
    kerma_image = axes[1, 1].imshow(
        kerma,
        origin="lower",
        extent=_local_imshow_extent(result.grid.bounds),
        cmap="inferno",
    )
    axes[1, 1].set_title("Collision air kerma rate")
    figure.colorbar(kerma_image, ax=axes[1, 1], label="Collision air kerma (uGy/h)")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.98,
        "Independent radiation product\n\n"
        "Radionuclide: Cs-137\n"
        f"Grid shape: {result.grid.shape}\n"
        f"Resolution: {result.grid.resolution_m:g} m\n"
        "Observation height: 1 m AGL\n"
        "Kernel normalised: NO\n"
        "Background included: NO\n"
        f"Max kerma: {kerma.max():.6f} uGy/h\n\n"
        "No OSM, heatmap, terrain, detector,\n"
        "noise, shielding, or planner",
        va="top",
        fontsize=11,
    )
    for axis in axes.flat:
        if axis.axison and axis is not axes[0, 2]:
            axis.set_xlabel("Local x (m)")
            axis.set_ylabel("Local y (m)")

    figure.suptitle(name.replace("_", " ").title(), fontsize=16, y=1.02)
    figure_path = output_directory / "overview.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {name}: {figure_path}")
    print(f"grid={result.grid.shape}, max fluence={fluence.max():.6f} photons/m^2/s")
    print(f"max collision air kerma={kerma.max():.6f} uGy/h")
    return result, paths, figure_path


__all__ = ["OUTPUT_DIRECTORY", "run_surface_example"]
