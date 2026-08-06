"""Generate a deterministic 1 m Uniform Rectangle radiation patch."""

from shapely.geometry import box

from sarenv.radiation import (
    BenchmarkSurfaceResponseConfig,
    BenchmarkSurfaceResponseKernel,
    UniformPolygonConfig,
    UniformPolygonSource,
)

from _common import run_surface_example


source = UniformPolygonSource(
    UniformPolygonConfig(
        source_id="uniform_rectangle",
        geometry=box(500_000, 5_700_000, 500_080, 5_700_050),
        nominal_surface_field_uSv_h=5.0,
        crs="EPSG:32630",
    )
)
kernel = BenchmarkSurfaceResponseKernel.create(
    BenchmarkSurfaceResponseConfig(core_radius_m=2.0, cutoff_radius_m=20.0)
)


if __name__ == "__main__":
    run_surface_example("01_uniform_rectangle", source, kernel)
