# Radiation API migration

The former `RadiationConfig` / `generate_radiation_layer()` workflow generated
one raster on the SAR master grid, read `heatmap.npy`, and loaded/cropped that
raster through `DatasetLoader`. It is deprecated and disabled.

## New model boundaries

- SAR retains its existing probability heatmap, resolution, loader, paths, and
  evaluation behaviour.
- Radiation uses independent deterministic 1 m local grids.
- `PointSourceConfig` and `PointSource` provide analytic, excess-only point
  queries. A point raster is optional and local.
- `UniformPolygonConfig` or `ZonedPolygonConfig` define physical surface
  activity rasters. `simulate_surface_source()` converts activity density to
  cell activity and applies the unnormalised, inverse-square, air-attenuated
  Cs-137 photon kernel using linear FFT convolution.
- `DoseRatePatch` contains source excess only. `CompositeRadiationField` owns
  the background and adds it exactly once.
- `TerrainContext` reads `features.geojson` and public spatial metadata for CRS,
  bounds checks, and visualisation. It does not read `heatmap.npy` and does not
  alter propagation.

## Surface example

```python
from shapely.geometry import box
from sarenv.radiation import (
    BenchmarkSurfaceResponseConfig,
    BenchmarkSurfaceResponseKernel,
    UniformPolygonConfig,
    UniformPolygonSource,
    save_surface_simulation,
    simulate_surface_source,
)

source = UniformPolygonSource(UniformPolygonConfig(
    source_id="area-a",
    geometry=box(500000, 5700000, 500050, 5700040),
    nominal_surface_field_uSv_h=5.0,
    crs="EPSG:32630",
))
kernel = BenchmarkSurfaceResponseKernel.create(
    BenchmarkSurfaceResponseConfig(core_radius_m=2.0, cutoff_radius_m=20.0)
)
result = simulate_surface_source(source, kernel)
save_surface_simulation(result, "scenario_a")
```

There is intentionally no implicit placement preset, Gaussian surface model,
seed, random texture, heatmap-shape check, or small/medium/large SAR crop.
Existing `radiation.npy` files are ignored by `DatasetLoader`; migrate them by
regenerating an explicit point source or polygon scenario.

## Formal UAV height contract

The formal experiment independently evaluates the same hidden source at 1 m
and 50 m AGL. Survivor exposure uses 1 m truth. The UAV sensor, online
estimator, radiation trigger/planner and UAV exposure use 50 m truth. The
noise-free sensor has no 50 m-to-1 m correction or ground-equivalent mode.
