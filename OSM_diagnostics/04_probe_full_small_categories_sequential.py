"""Probe 4: query the exact Direct Small polygon one category at a time."""

from __future__ import annotations

import ast
import datetime as dt
import json
import subprocess
import time
import traceback
from collections import Counter
from typing import Any

import osmnx as ox

from diagnostic_common import (
    DIAGNOSTIC_DIR,
    FEATURES_DIR,
    build_full_small_polygon,
    configure_isolated_osmnx,
    run_logged,
    save_json,
    summarize_geometries,
)
from sarenv.core.generation import DataGenerator
from sarenv.core.geometries import GeoPolygon
from sarenv.io import osm_query as osm_query_module


SUMMARY_PATH = DIAGNOSTIC_DIR / "04_full_small_categories_summary.json"
INITIAL_MAIN_COMMIT = "0f2d344d17c0b9c18cec2176dd9b171beba727d6"
DEFAULT_OSMNX_USER_AGENT = "OSMnx Python package (https://github.com/gboeing/osmnx)"
EMPTY_RESPONSE_MARKERS = (
    "No matching features",
    "Found no features",
    "No data elements",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def load_initial_main_tags_mapping() -> dict[str, Any]:
    """Read DataGenerator.tags_mapping literally from task-start main source."""

    source = git("show", f"{INITIAL_MAIN_COMMIT}:sarenv/core/generation.py")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "DataGenerator":
            continue
        for child in node.body:
            if not isinstance(child, ast.FunctionDef) or child.name != "__init__":
                continue
            for statement in ast.walk(child):
                if not isinstance(statement, ast.Assign):
                    continue
                if any(
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "tags_mapping"
                    for target in statement.targets
                ):
                    return ast.literal_eval(statement.value)
    raise RuntimeError("Could not locate DataGenerator.tags_mapping in initial main")


def exception_payload(exception: Exception) -> dict[str, str]:
    return {
        "exception_class": f"{type(exception).__module__}.{type(exception).__name__}",
        "exception_text": str(exception),
        "exception_repr": repr(exception),
    }


def is_genuine_empty_response(exception: Exception) -> bool:
    return (
        type(exception).__name__ == "InsufficientResponseError"
        and any(marker in str(exception) for marker in EMPTY_RESPONSE_MARKERS)
    )


def run_wrapper_on_fetched_gdf(
    polygon: Any, tags: dict[str, Any], raw_gdf: Any
) -> dict[str, Any]:
    """Run the real wrapper post-processing without issuing another request."""

    original_function = osm_query_module.ox.features_from_polygon

    def return_fetched_gdf(*_args: Any, **_kwargs: Any) -> Any:
        return raw_gdf.copy()

    osm_query_module.ox.features_from_polygon = return_fetched_gdf
    try:
        wrapped = osm_query_module.query_features(
            GeoPolygon(polygon, crs="EPSG:4326"), tags
        )
    finally:
        osm_query_module.ox.features_from_polygon = original_function

    if wrapped is None:
        return {"returned_none": True, "returned_keys": [], "geometries": {}}
    geometry_results = {}
    for key, geometry in wrapped.items():
        geometry_results[key] = (
            None if geometry is None else summarize_geometries([geometry])
        )
    return {
        "returned_none": False,
        "returned_keys": list(wrapped.keys()),
        "geometries": geometry_results,
    }


def persist_summary(summary: dict[str, Any]) -> None:
    status_counts = Counter(
        result["status"] for result in summary["category_results"]
    )
    summary["status_counts"] = dict(status_counts)
    raw_multi_geometry_categories = {
        geometry_type: [
            result["category"]
            for result in summary["category_results"]
            if result.get("geometry_summary", {}).get(f"has_{geometry_type}", False)
        ]
        for geometry_type in (
            "MultiPolygon",
            "MultiLineString",
            "GeometryCollection",
        )
    }
    wrapper_multi_geometry_categories = {}
    for geometry_type in (
        "MultiPolygon",
        "MultiLineString",
        "GeometryCollection",
    ):
        categories = []
        for result in summary["category_results"]:
            wrapped = result.get("sarenv_wrapper_on_same_raw_response", {})
            wrapper_geometries = wrapped.get("geometries", {})
            if any(
                geometry_summary
                and geometry_summary.get(f"has_{geometry_type}", False)
                for geometry_summary in wrapper_geometries.values()
            ):
                categories.append(result["category"])
        wrapper_multi_geometry_categories[geometry_type] = categories
    summary["multi_geometry_categories"] = {
        "raw_osmnx_rows": raw_multi_geometry_categories,
        "sarenv_wrapper_outputs": wrapper_multi_geometry_categories,
    }
    save_json(SUMMARY_PATH, summary)


def save_category_geojson(category: str, raw_gdf: Any) -> str:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FEATURES_DIR / f"{category}.geojson"
    output_path.write_text(raw_gdf.to_json(drop_id=False), encoding="utf-8")
    return str(output_path)


def query_category(
    category: str, tags: dict[str, Any], polygon: Any
) -> dict[str, Any]:
    category_start = utc_now()
    total_start = time.perf_counter()
    attempts = []
    raw_gdf = None
    status = "FAIL_QUERY"
    final_error = None

    for attempt_number in (1, 2):
        attempt_start_utc = utc_now()
        attempt_start = time.perf_counter()
        print(
            f"ATTEMPT category={category} number={attempt_number} "
            f"start_utc={attempt_start_utc} tags={json.dumps(tags, ensure_ascii=False)}"
        )
        try:
            raw_gdf = ox.features_from_polygon(polygon, tags=tags)
            attempt_elapsed = time.perf_counter() - attempt_start
            if raw_gdf.empty:
                status = "PASS_EMPTY"
                attempt_status = "PASS_EMPTY"
            else:
                status = "PASS_WITH_DATA"
                attempt_status = "PASS_WITH_DATA"
            attempts.append(
                {
                    "attempt": attempt_number,
                    "start_time_utc": attempt_start_utc,
                    "elapsed_seconds": attempt_elapsed,
                    "status": attempt_status,
                    "row_count": len(raw_gdf),
                }
            )
            print(
                f"ATTEMPT_RESULT category={category} number={attempt_number} "
                f"status={attempt_status} rows={len(raw_gdf)} "
                f"elapsed_seconds={attempt_elapsed:.3f}"
            )
            break
        except Exception as exception:
            attempt_elapsed = time.perf_counter() - attempt_start
            empty_response = is_genuine_empty_response(exception)
            attempt_status = "PASS_EMPTY" if empty_response else "FAIL_QUERY"
            payload = {
                "attempt": attempt_number,
                "start_time_utc": attempt_start_utc,
                "elapsed_seconds": attempt_elapsed,
                "status": attempt_status,
                **exception_payload(exception),
            }
            attempts.append(payload)
            print(
                f"ATTEMPT_RESULT category={category} number={attempt_number} "
                f"status={attempt_status} elapsed_seconds={attempt_elapsed:.3f} "
                f"exception_class={payload['exception_class']} "
                f"exception_text={payload['exception_text']}"
            )
            traceback.print_exc()
            if empty_response:
                status = "PASS_EMPTY"
                final_error = None
                break
            final_error = exception_payload(exception)
            if attempt_number == 1:
                print(f"RETRY category={category}: one recorded sequential retry")

    result: dict[str, Any] = {
        "category": category,
        "tags": tags,
        "start_time_utc": category_start,
        "elapsed_seconds": time.perf_counter() - total_start,
        "status": status,
        "attempts": attempts,
        "error": final_error if status == "FAIL_QUERY" else None,
    }

    if status == "PASS_WITH_DATA" and raw_gdf is not None:
        geometry_summary = summarize_geometries(raw_gdf.geometry)
        result["row_count"] = len(raw_gdf)
        result["crs"] = str(raw_gdf.crs)
        result["geometry_summary"] = geometry_summary
        try:
            result["diagnostic_geojson"] = save_category_geojson(category, raw_gdf)
        except Exception as exception:
            result["diagnostic_geojson_error"] = exception_payload(exception)
            print(
                f"GEOJSON_SAVE_WARNING category={category} "
                f"exception={exception!r}"
            )
        try:
            result["sarenv_wrapper_on_same_raw_response"] = run_wrapper_on_fetched_gdf(
                polygon, tags, raw_gdf
            )
        except Exception as exception:
            result["sarenv_wrapper_on_same_raw_response"] = {
                "processing_exception": exception_payload(exception)
            }
            print(
                f"WRAPPER_POSTPROCESS_WARNING category={category} "
                f"exception={exception!r}"
            )
            traceback.print_exc()
        print(f"GEOMETRY_SUMMARY category={category}: {geometry_summary}")
    elif status == "PASS_EMPTY":
        result["row_count"] = 0
        result["geometry_summary"] = summarize_geometries([])

    print(
        f"CATEGORY_RESULT category={category} status={status} "
        f"elapsed_seconds={result['elapsed_seconds']:.3f}"
    )
    return result


def main() -> None:
    print("PROBE 4 — exact Direct Small categories, sequential, no concurrency")
    settings = configure_isolated_osmnx()
    ox.settings.http_user_agent = DEFAULT_OSMNX_USER_AGENT
    polygon, polygon_details = build_full_small_polygon()
    current_worktree_mapping = DataGenerator().tags_mapping
    tags_mapping = load_initial_main_tags_mapping()
    worktree_branch = git("branch", "--show-current")
    worktree_head = git("rev-parse", "HEAD")
    print(f"OSMnx version: {ox.__version__}")
    print(f"Overpass URL: {ox.settings.overpass_url}")
    print(f"timeout: {ox.settings.requests_timeout}")
    print(f"cache isolation: {settings}")
    print(f"worktree branch at execution: {worktree_branch}")
    print(f"worktree HEAD at execution: {worktree_head}")
    print(f"mapping source locked to initial main commit: {INITIAL_MAIN_COMMIT}")
    print(
        "current worktree mapping matches initial main: "
        f"{current_worktree_mapping == tags_mapping}"
    )
    print(f"HTTP user agent restored to main behavior: {ox.settings.http_user_agent}")
    print(f"metadata source: {polygon_details['metadata_source']}")
    print(f"centre WGS84: {polygon_details['center_wgs84']}")
    print(f"projected CRS: {polygon_details['projected_crs']}")
    print(f"projected bounds: {polygon_details['projected_bounds']}")
    print(f"WGS84 bounds: {polygon_details['wgs84_bounds']}")
    print(
        f"physical size: {polygon_details['width_m']} m x "
        f"{polygon_details['height_m']} m"
    )
    print(f"category count: {len(tags_mapping)}")
    print("production tags_mapping:")
    print(json.dumps(tags_mapping, indent=2, ensure_ascii=False))

    if not (
        abs(polygon_details["width_m"] - 1200.0) < 1e-6
        and abs(polygon_details["height_m"] - 1200.0) < 1e-6
    ):
        raise ValueError("Authoritative Small bounds are not 1200 m x 1200 m")

    summary: dict[str, Any] = {
        "probe": "04 full Direct Small categories sequential",
        "started_utc": utc_now(),
        "completed_utc": None,
        "osmnx_version": ox.__version__,
        "overpass_settings": settings,
        "polygon": polygon_details,
        "category_count": len(tags_mapping),
        "tags_mapping": tags_mapping,
        "execution_mode": "strictly sequential; no process/thread executor",
        "retry_policy": "one recorded retry only for FAIL_QUERY; no retry for PASS_EMPTY",
        "source_lock": {
            "initial_main_commit": INITIAL_MAIN_COMMIT,
            "worktree_branch_at_execution": worktree_branch,
            "worktree_head_at_execution": worktree_head,
            "current_worktree_mapping_matches_initial_main": (
                current_worktree_mapping == tags_mapping
            ),
            "http_user_agent": ox.settings.http_user_agent,
        },
        "category_results": [],
    }
    persist_summary(summary)

    for category, tags in tags_mapping.items():
        print()
        print("=" * 80)
        print(f"CATEGORY_START {category}")
        result = query_category(category, tags, polygon)
        summary["category_results"].append(result)
        persist_summary(summary)

    summary["completed_utc"] = utc_now()
    persist_summary(summary)
    print()
    print(f"FINAL_STATUS_COUNTS: {summary['status_counts']}")
    print(
        "FINAL_MULTI_GEOMETRY_CATEGORIES: "
        f"{summary['multi_geometry_categories']}"
    )
    print(f"summary JSON: {SUMMARY_PATH}")
    print("RESULT: COMPLETE")


if __name__ == "__main__":
    run_logged("04_full_small_categories.log", main)
