"""Shared plotting utilities for independent surface-radiation examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sarenv.radiation import save_surface_simulation, simulate_surface_source


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = EXAMPLE_DIRECTORY / "outputs"


def _imshow_extent(bounds):
    """Convert (minx, miny, maxx, maxy) to Matplotlib extent order."""
    minx, miny, maxx, maxy = bounds
    return minx, maxx, miny, maxy


def _plot_source_geometry(axis, source) -> None:
    if source.surface_type == "uniform_polygon":
        geometries = [source.config.geometry]
        intensities = [source.config.nominal_surface_field_uSv_h]
    else:
        geometries = [zone.geometry for zone in source.config.zones]
        intensities = [
            zone.nominal_surface_field_uSv_h for zone in source.config.zones
        ]
    colours = plt.cm.viridis(
        np.linspace(0.25, 0.9, max(1, len(geometries)))
    )
    for geometry, intensity, colour in zip(
        geometries, intensities, colours, strict=True
    ):
        polygons = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
        for polygon in polygons:
            x, y = polygon.exterior.xy
            axis.fill(x, y, color=colour, alpha=0.6, label=f"{intensity:g} uSv/h")
    axis.set_aspect("equal")
    axis.set_title("Input polygon / zones")
    axis.legend(loc="best", fontsize=8)


def run_surface_example(
    name,
    source,
    kernel,
    *,
    background_uSv_h=0.20,
):
    """Simulate, save, and visualise one deterministic local surface patch."""
    result = simulate_surface_source(source, kernel)
    output_directory = OUTPUT_DIRECTORY / name
    paths = save_surface_simulation(result, output_directory)

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    _plot_source_geometry(axes[0, 0], source)

    nominal_image = axes[0, 1].imshow(
        result.nominal_surface_field,
        origin="lower",
        extent=_imshow_extent(result.nominal_grid.bounds),
        cmap="viridis",
    )
    axes[0, 1].set_title("1 m nominal surface field")
    figure.colorbar(nominal_image, ax=axes[0, 1], label="Nominal uSv/h")

    cutoff = kernel.config.cutoff_radius_m
    kernel_image = axes[0, 2].imshow(
        kernel.values,
        origin="lower",
        extent=(-cutoff - 0.5, cutoff + 0.5, -cutoff - 0.5, cutoff + 0.5),
        cmap="magma",
    )
    axes[0, 2].set_title("Normalised benchmark kernel")
    axes[0, 2].set_xlabel("x offset (m)")
    axes[0, 2].set_ylabel("y offset (m)")
    figure.colorbar(kernel_image, ax=axes[0, 2], label="Kernel weight")

    patch = result.dose_rate_patch
    excess_image = axes[1, 0].imshow(
        patch.excess_uSv_h,
        origin="lower",
        extent=_imshow_extent(patch.grid.bounds),
        cmap="inferno",
    )
    axes[1, 0].set_title("Full-convolution excess patch")
    figure.colorbar(excess_image, ax=axes[1, 0], label="Excess dose rate (uSv/h)")

    total_image = axes[1, 1].imshow(
        patch.excess_uSv_h + background_uSv_h,
        origin="lower",
        extent=_imshow_extent(patch.grid.bounds),
        cmap="inferno",
    )
    axes[1, 1].set_title(f"Total in patch (+ {background_uSv_h:g} uSv/h background)")
    figure.colorbar(total_image, ax=axes[1, 1], label="Total dose rate (uSv/h)")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.98,
        "Independent radiation product\n\n"
        f"Input shape: {result.nominal_grid.shape}\n"
        f"Output shape: {patch.grid.shape}\n"
        f"Resolution: {patch.grid.resolution_m:g} m\n"
        f"Kernel sum: {kernel.values.sum():.12f}\n"
        "dose_rate_patch.npy: excess only\n"
        "No noise, sensor, nuclide, or terrain model",
        va="top",
        fontsize=11,
    )
    for axis in axes.flat:
        if axis.axison and axis is not axes[0, 2]:
            axis.set_xlabel("Easting (m)")
            axis.set_ylabel("Northing (m)")

    figure.suptitle(name.replace("_", " ").title(), fontsize=16, y=1.02)
    figure_path = output_directory / "overview.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {name}: {figure_path}")
    print(
        f"nominal={result.nominal_grid.shape}, patch={patch.grid.shape}, "
        f"max excess={patch.excess_uSv_h.max():.6f} uSv/h"
    )
    return result, paths, figure_path


__all__ = ["OUTPUT_DIRECTORY", "run_surface_example"]
