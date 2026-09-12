"""Display SAR and radiation in one CRS without matching raster resolution."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

from sarenv import DatasetLoader
from sarenv.radiation import (
    BenchmarkSurfaceResponseConfig,
    BenchmarkSurfaceResponseKernel,
    TerrainContext,
    UniformPolygonConfig,
    UniformPolygonSource,
    simulate_surface_source,
)


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
DATASET_DIRECTORY = (
    EXAMPLE_DIRECTORY.parent
    / "sarenv_dataset"
    / "sarenv_outputs"
    / "radiation_area_01"
)
OUTPUT_PATH = EXAMPLE_DIRECTORY / "outputs" / "04_combined_alignment.png"


def _imshow_extent(bounds):
    minx, miny, maxx, maxy = bounds
    return minx, maxx, miny, maxy


def main():
    context = TerrainContext.from_dataset(DATASET_DIRECTORY)
    item = DatasetLoader(str(DATASET_DIRECTORY)).load_environment("medium")
    center = gpd.GeoSeries(
        [Point(context.center_point_wgs84)], crs="EPSG:4326"
    ).to_crs(context.projected_crs).iloc[0]
    polygon = Point(center.x + 350.0, center.y + 150.0).buffer(45.0)
    source = UniformPolygonSource(
        UniformPolygonConfig(
            "combined_surface",
            polygon,
            5.0,
            context.projected_crs,
        )
    )
    kernel = BenchmarkSurfaceResponseKernel.create(
        BenchmarkSurfaceResponseConfig(2.0, 30.0)
    )
    patch = simulate_surface_source(source, kernel).dose_rate_patch
    context.validate_grid(patch.grid)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].imshow(
        item.heatmap,
        origin="lower",
        extent=_imshow_extent(item.bounds),
        cmap="viridis",
    )
    axes[0].set_title(f"SAR probability ({item.heatmap.shape}, 30 m)")
    axes[1].imshow(
        patch.excess_uSv_h,
        origin="lower",
        extent=_imshow_extent(patch.grid.bounds),
        cmap="inferno",
    )
    axes[1].set_title(f"Radiation patch ({patch.grid.shape}, 1 m)")
    axes[2].imshow(
        item.heatmap,
        origin="lower",
        extent=_imshow_extent(item.bounds),
        cmap="Greys",
        alpha=0.45,
    )
    if not item.features.empty:
        item.features.plot(ax=axes[2], color="steelblue", linewidth=0.5, alpha=0.4)
    axes[2].imshow(
        patch.excess_uSv_h,
        origin="lower",
        extent=_imshow_extent(patch.grid.bounds),
        cmap="inferno",
        alpha=0.75,
    )
    axes[2].set_xlim(item.bounds[0], item.bounds[2])
    axes[2].set_ylim(item.bounds[1], item.bounds[3])
    axes[2].set_title("Shared CRS; no raster resampling")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=160)
    plt.close(figure)
    print(f"Saved combined alignment view: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
