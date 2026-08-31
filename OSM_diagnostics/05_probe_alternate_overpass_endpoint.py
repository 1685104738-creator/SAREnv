"""Probe 5: repeat the tiny direct query on a process-local alternate endpoint."""

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


ALTERNATE_BASE_URL = "https://overpass.private.coffee/api"
SOURCE_URL = "https://wiki.openstreetmap.org/wiki/Overpass_API"


def main() -> None:
    print("PROBE 5 — process-local alternate Overpass endpoint")
    print(
        "default endpoint result (from completed Probe 2): FAIL — "
        "requests.exceptions.ConnectTimeout after 102.211 seconds at "
        "https://overpass-api.de/api/interpreter"
    )
    print(f"alternate endpoint source: {SOURCE_URL}")
    settings = configure_isolated_osmnx()
    polygon, details = build_tiny_polygon(size_m=150.0)
    default_base_url = ox.settings.overpass_url
    print(f"default base URL before temporary override: {default_base_url}")
    print(f"alternate base URL: {ALTERNATE_BASE_URL}")
    print(f"timeout: {ox.settings.requests_timeout}")
    print(f"cache isolation: {settings}")
    print(f"polygon WGS84 bounds: {details['wgs84_bounds']}")
    print("tags: {'highway': True}")

    start = time.perf_counter()
    try:
        ox.settings.overpass_url = ALTERNATE_BASE_URL
        features = ox.features_from_polygon(polygon, tags={"highway": True})
        elapsed = time.perf_counter() - start
        print(f"alternate elapsed_seconds: {elapsed:.3f}")
        print(f"alternate rows: {len(features)}")
        print(f"alternate CRS: {features.crs}")
        print(
            "alternate geometry summary: "
            f"{summarize_geometries(features.geometry)}"
        )
        output_path = DIAGNOSTIC_DIR / "05_alternate_highway.geojson"
        output_path.write_text(features.to_json(drop_id=False), encoding="utf-8")
        print(f"saved diagnostic GeoJSON: {output_path}")
        print("alternate endpoint result: PASS")
        print("RESULT: DEFAULT_FAIL_ALTERNATE_PASS")
    except Exception as exception:
        elapsed = time.perf_counter() - start
        print(f"alternate elapsed_seconds: {elapsed:.3f}")
        print(
            "alternate exception_type: "
            f"{type(exception).__module__}.{type(exception).__name__}"
        )
        print(f"alternate exception: {exception}")
        print(f"alternate exception_repr: {exception!r}")
        print("alternate full traceback:")
        traceback.print_exc()
        print("alternate endpoint result: FAIL")
        print("RESULT: DEFAULT_FAIL_ALTERNATE_FAIL")
    finally:
        ox.settings.overpass_url = default_base_url
        print(
            "process-local setting restored before exit: "
            f"{ox.settings.overpass_url}"
        )


if __name__ == "__main__":
    run_logged("05_probe_alternate_overpass_endpoint.log", main)

