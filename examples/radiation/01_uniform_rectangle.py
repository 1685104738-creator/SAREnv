"""Generate a deterministic 1 m Uniform Rectangle radiation patch."""

from shapely.geometry import box

from sarenv.radiation import (
    UniformPolygonConfig,
    UniformPolygonSource,
)
from sarenv.radiation.surface import MEDIUM_ACTIVITY_DENSITY_BQ_M2

from _common import run_surface_example


source = UniformPolygonSource(
    UniformPolygonConfig(
        source_id="uniform_rectangle",
        geometry=box(500_000, 5_700_000, 500_080, 5_700_050),
        activity_density_bq_m2=MEDIUM_ACTIVITY_DENSITY_BQ_M2,
        crs="EPSG:32630",
    )
)


if __name__ == "__main__":
    run_surface_example("01_uniform_rectangle_cs137", source)
