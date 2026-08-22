"""Generate and visualise one fully offline analytic point source."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from sarenv.radiation import GridSpec, PointSource, PointSourceConfig
from sarenv.radiation import save_point_source


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = EXAMPLE_DIRECTORY / "outputs" / "00_point_source"
FIGURE_PATH = OUTPUT_DIRECTORY / "overview.png"
CRS = "EPSG:32630"
SOURCE_X_M = 500_000.0
SOURCE_Y_M = 5_700_000.0
REFERENCE_EXCESS_USV_H = 20_000.0


def run_point_example():
    """Save point metadata and a local 1 m visualisation; no map is loaded."""
    source = PointSource(
        PointSourceConfig(
            source_id="offline_point_source",
            x_m=SOURCE_X_M,
            y_m=SOURCE_Y_M,
            reference_excess_uSv_h=REFERENCE_EXCESS_USV_H,
            crs=CRS,
        )
    )
    grid = GridSpec.from_bounds(
        (
            SOURCE_X_M - 50.0,
            SOURCE_Y_M - 50.0,
            SOURCE_X_M + 50.0,
            SOURCE_Y_M + 50.0,
        ),
        CRS,
    )
    field = source.rasterize(grid)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    metadata_path = save_point_source(
        source, OUTPUT_DIRECTORY / "point_source_meta.json"
    )
    np.save(OUTPUT_DIRECTORY / "point_source_field_uSv_h.npy", field)

    extent = (grid.minx, grid.maxx, grid.miny, grid.maxy)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    image = axes[0].imshow(
        field,
        origin="lower",
        extent=extent,
        cmap="inferno",
        norm=LogNorm(vmin=max(float(field.min()), 1e-12), vmax=float(field.max())),
    )
    axes[0].scatter(
        [SOURCE_X_M],
        [SOURCE_Y_M],
        marker="*",
        s=130,
        c="cyan",
        edgecolors="black",
        label="Point source",
    )
    axes[0].set_title("Analytic point-source excess field")
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper right")
    figure.colorbar(image, ax=axes[0], label="Excess dose rate (uSv/h)")

    row = grid.world_to_row_col(SOURCE_X_M, SOURCE_Y_M)[0]
    x_centres = grid.cell_center_mesh()[0][row]
    axes[1].semilogy(
        x_centres - SOURCE_X_M,
        field[row],
        color="darkred",
        linewidth=2,
    )
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("East-west centreline")
    axes[1].set_xlabel("Horizontal offset from source (m)")
    axes[1].set_ylabel("Excess dose rate (uSv/h)")
    axes[1].grid(True, which="both", alpha=0.25)

    figure.suptitle(
        "Point-only radiation example (fully offline)", fontsize=15
    )
    figure.savefig(FIGURE_PATH, dpi=170, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved point image: {FIGURE_PATH}")
    print(f"Saved point metadata: {metadata_path}")
    print(
        f"grid={grid.shape}, min={field.min():.6f} uSv/h, "
        f"max raster value={field.max():.6f} uSv/h"
    )
    return source, field, FIGURE_PATH


if __name__ == "__main__":
    run_point_example()
