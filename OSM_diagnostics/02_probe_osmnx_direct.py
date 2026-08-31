"""Probe 2: direct tiny OSMnx/Overpass query with no SAREnv wrapper."""

from __future__ import annotations

import time
import traceback

import osmnx as ox

from diagnostic_common import (
    DIAGNOSTIC_DIR,
    build_tiny_polygon,
    configure_isolated_osmnx,
    run_logged,
    summarize_geometries,
)


def main() -> None:
    print("PROBE 2 — direct OSMnx tiny highway query")
    settings = configure_isolated_osmnx()
    polygon, details = build_tiny_polygon(size_m=150.0)
    print(f"OSMnx version: {ox.__version__}")
    print(f"Overpass URL: {ox.settings.overpass_url}")
    print(f"timeout: {ox.settings.requests_timeout}")
    print(f"cache isolation: {settings}")
    print(f"metadata source: {details['metadata_source']}")
    print(f"centre WGS84: {details['center_wgs84']}")
    print(f"projected bounds: {details['projected_bounds']}")
    print(f"polygon WGS84 bounds: {details['wgs84_bounds']}")
    print("tags: {'highway': True}")

    start = time.perf_counter()
    try:
        features = ox.features_from_polygon(polygon, tags={"highway": True})
        elapsed = time.perf_counter() - start
        print(f"elapsed_seconds: {elapsed:.3f}")
        print(f"rows: {len(features)}")
        print(f"CRS: {features.crs}")
        print(f"geometry summary: {summarize_geometries(features.geometry)}")
        basic_columns = [
            column
            for column in ("highway", "name", "ref", "geometry")
            if column in features.columns
        ]
        preview = features[basic_columns].head(5)
        print("first features:")
        print(preview.to_string())
        output_path = DIAGNOSTIC_DIR / "02_direct_highway.geojson"
        output_path.write_text(features.to_json(drop_id=False), encoding="utf-8")
        print(f"saved diagnostic GeoJSON: {output_path}")
        print("RESULT: PASS")
    except Exception as exception:
        elapsed = time.perf_counter() - start
        print(f"elapsed_seconds: {elapsed:.3f}")
        print(f"exception_type: {type(exception).__module__}.{type(exception).__name__}")
        print(f"exception: {exception}")
        print(f"exception_repr: {exception!r}")
        print("full traceback:")
        traceback.print_exc()
        print("RESULT: FAIL")


if __name__ == "__main__":
    run_logged("02_probe_osmnx_direct.log", main)

