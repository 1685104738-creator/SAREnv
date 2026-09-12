"""Generate a deterministic irregular Uniform Polygon radiation patch."""

from shapely.geometry import Polygon

from sarenv.radiation import (
    UniformPolygonConfig,
    UniformPolygonSource,
)
from sarenv.radiation.surface import HIGH_ACTIVITY_DENSITY_BQ_M2

from _common import run_surface_example


geometry = Polygon(
    [
        (500_000, 5_700_000),
        (500_065, 5_700_006),
        (500_088, 5_700_035),
        (500_051, 5_700_072),
        (500_015, 5_700_058),
        (499_990, 5_700_026),
    ]
)
source = UniformPolygonSource(
    UniformPolygonConfig(
        source_id="irregular_uniform",
        geometry=geometry,
        activity_density_bq_m2=HIGH_ACTIVITY_DENSITY_BQ_M2,
        crs="EPSG:32630",
    )
)


if __name__ == "__main__":
    run_surface_example("02_irregular_uniform_polygon_cs137", source)
