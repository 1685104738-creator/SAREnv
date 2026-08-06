"""Generate a deterministic last-defined-wins Zoned Polygon radiation patch."""

from shapely.geometry import Polygon, box

from sarenv.radiation import (
    BenchmarkSurfaceResponseConfig,
    BenchmarkSurfaceResponseKernel,
    SurfaceZone,
    ZonedPolygonConfig,
    ZonedPolygonSource,
)

from _common import run_surface_example


outer = Polygon(
    [
        (500_000, 5_700_000),
        (500_100, 5_700_000),
        (500_110, 5_700_070),
        (500_030, 5_700_090),
        (499_985, 5_700_045),
    ]
)
zones = (
    SurfaceZone("outer", outer, 2.0),
    SurfaceZone("middle", box(500_020, 5_700_020, 500_085, 5_700_067), 6.0),
    SurfaceZone("core", box(500_043, 5_700_032, 500_068, 5_700_055), 12.0),
)
source = ZonedPolygonSource(
    ZonedPolygonConfig(source_id="zoned_polygon", zones=zones, crs="EPSG:32630")
)
kernel = BenchmarkSurfaceResponseKernel.create(
    BenchmarkSurfaceResponseConfig(core_radius_m=2.0, cutoff_radius_m=25.0)
)


if __name__ == "__main__":
    run_surface_example("03_zoned_polygon", source, kernel)
