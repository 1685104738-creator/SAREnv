"""Probe 1: offline API and dependency compatibility check."""

from __future__ import annotations

import ast
import inspect
import platform
import sys
import time

import geopandas as gpd
import osmnx as ox
import pyproj
import shapely
import sarenv

from diagnostic_common import run_logged
from sarenv.io import osm_query


def main() -> None:
    probe_start = time.perf_counter()
    print("PROBE 1 — API / dependency compatibility (OFFLINE; NO NETWORK)")
    print(f"Python: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"sarenv module path: {sarenv.__file__}")
    print(f"osmnx: {ox.__version__}")
    print(f"geopandas: {gpd.__version__}")
    print(f"shapely: {shapely.__version__}")
    print(f"pyproj: {pyproj.__version__}")
    print()

    api_exists = hasattr(ox, "features_from_polygon")
    print(f"hasattr(ox, 'features_from_polygon'): {api_exists}")
    if api_exists:
        print(
            "ox.features_from_polygon signature: "
            f"{inspect.signature(ox.features_from_polygon)}"
        )
    else:
        print("ox.features_from_polygon signature: UNAVAILABLE")
    print(
        "sarenv.io.osm_query.query_features signature: "
        f"{inspect.signature(osm_query.query_features)}"
    )

    source = inspect.getsource(osm_query.query_features)
    tree = ast.parse(source)
    production_calls = []
    uses_tags_keyword = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "features_from_polygon"
        ):
            keyword_names = [keyword.arg for keyword in node.keywords]
            production_calls.append(
                {
                    "function": ast.unparse(function),
                    "keyword_names": keyword_names,
                    "source": ast.unparse(node),
                }
            )
            uses_tags_keyword = "tags" in keyword_names
    print(f"production features_from_polygon calls: {production_calls}")
    print(f"production call uses tags= keyword: {uses_tags_keyword}")

    passed = api_exists and uses_tags_keyword
    print()
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    if passed:
        print(
            "FINDING: installed OSMnx exposes the API shape used by SAREnv. "
            "This offline result says nothing about network/Overpass health."
        )
    else:
        print("FINDING: installed OSMnx API is incompatible with the current wrapper.")
    print(f"elapsed_seconds: {time.perf_counter() - probe_start:.6f}")


if __name__ == "__main__":
    run_logged("01_check_api_versions.log", main)
