"""Generate a deterministic irregular Uniform Polygon radiation patch."""

from shapely.geometry import Polygon

from sarenv.radiation import (
    BenchmarkSurfaceResponseConfig,
    BenchmarkSurfaceResponseKernel,
    UniformPolygonConfig,
    UniformPolygonSource,
)

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
        nominal_surface_field_uSv_h=8.0,
        crs="EPSG:32630",
    )
)
kernel = BenchmarkSurfaceResponseKernel.create(
    BenchmarkSurfaceResponseConfig(core_radius_m=2.0, cutoff_radius_m=25.0)
)


if __name__ == "__main__":
    run_surface_example("02_irregular_uniform_polygon", source, kernel)
