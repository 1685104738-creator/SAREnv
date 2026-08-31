"""Probe 3: tiny query through SAREnv GeoPolygon and query_features."""

from __future__ import annotations

import subprocess
import time
import traceback
import types

import osmnx as ox

from diagnostic_common import (
    build_tiny_polygon,
    configure_isolated_osmnx,
    run_logged,
    summarize_geometries,
)
from sarenv.core.geometries import GeoPolygon


INITIAL_MAIN_COMMIT = "0f2d344d17c0b9c18cec2176dd9b171beba727d6"
DEFAULT_OSMNX_USER_AGENT = "OSMnx Python package (https://github.com/gboeing/osmnx)"


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def load_initial_main_query_features():
    """Load the exact task-start main wrapper without changing the worktree."""

    source = git("show", f"{INITIAL_MAIN_COMMIT}:sarenv/io/osm_query.py")
    module = types.ModuleType("sarenv.io.osm_query_initial_main_snapshot")
    module.__package__ = "sarenv.io"
    module.__file__ = (
        f"git:{INITIAL_MAIN_COMMIT}:sarenv/io/osm_query.py"
    )
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.query_features, module.__file__


def main() -> None:
    print("PROBE 3 — SAREnv native wrapper tiny highway query")
    settings = configure_isolated_osmnx()
    # Importing the externally switched branch can mutate this process-global
    # value. Restore the OSMnx default used by task-start main before querying.
    ox.settings.http_user_agent = DEFAULT_OSMNX_USER_AGENT
    query_features, wrapper_source = load_initial_main_query_features()
    polygon, details = build_tiny_polygon(size_m=150.0)
    geo_polygon = GeoPolygon(polygon, crs="EPSG:4326")
    tags = {"highway": True}
    print(f"worktree branch at execution: {git('branch', '--show-current')}")
    print(f"worktree HEAD at execution: {git('rev-parse', 'HEAD')}")
    print(f"wrapper source locked to task-start main: {wrapper_source}")
    print(f"HTTP user agent restored to main behavior: {ox.settings.http_user_agent}")
    print(f"cache isolation: {settings}")
    print(f"metadata source: {details['metadata_source']}")
    print(f"centre WGS84: {details['center_wgs84']}")
    print(f"GeoPolygon CRS: {geo_polygon.crs}")
    print(f"GeoPolygon bounds: {list(geo_polygon.get_geometry().bounds)}")
    print(f"tags: {tags}")

    start = time.perf_counter()
    try:
        result = query_features(geo_polygon, tags)
        elapsed = time.perf_counter() - start
        print(f"elapsed_seconds: {elapsed:.3f}")
        if result is None:
            print("returned: None")
            print("RESULT: FAIL")
            print(
                "FINDING: query_features returned None; inspect the SAREnv warning above "
                "to distinguish a swallowed request exception from empty post-processing."
            )
            return
        print(f"returned keys: {list(result.keys())}")
        for key, geometry in result.items():
            if geometry is None:
                print(f"key={key}: geometry=None")
                continue
            print(
                f"key={key}: geom_type={geometry.geom_type}, "
                f"is_empty={geometry.is_empty}, is_valid={geometry.is_valid}"
            )
            print(
                f"key={key}: recursive summary="
                f"{summarize_geometries([geometry])}"
            )
        nonempty = [
            geometry
            for geometry in result.values()
            if geometry is not None and not geometry.is_empty
        ]
        print(f"RESULT: {'PASS' if nonempty else 'FAIL'}")
    except Exception as exception:
        elapsed = time.perf_counter() - start
        print(f"elapsed_seconds: {elapsed:.3f}")
        print(f"exception_type: {type(exception).__module__}.{type(exception).__name__}")
        print(f"exception: {exception}")
        print("full traceback:")
        traceback.print_exc()
        print("RESULT: FAIL")


if __name__ == "__main__":
    run_logged("03_probe_sarenv_wrapper.log", main)
