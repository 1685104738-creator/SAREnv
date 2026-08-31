"""Enrich machine-readable results and build the final diagnostic report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diagnostic_common import DIAGNOSTIC_DIR, save_json


SUMMARY_PATH = DIAGNOSTIC_DIR / "04_full_small_categories_summary.json"
COMPARISON_PATH = DIAGNOSTIC_DIR / "06_existing_features_comparison.json"
SOURCE_ANALYSIS_PATH = DIAGNOSTIC_DIR / "07_source_control_flow_analysis.json"
INITIAL_OUTPUT_PATH = DIAGNOSTIC_DIR / "07_initial_main_output_observation.json"
REPORT_PATH = DIAGNOSTIC_DIR / "07_diagnostic_report.md"
INITIAL_MAIN_COMMIT = "0f2d344d17c0b9c18cec2176dd9b171beba727d6"


def wrapper_multi_categories(summary: dict[str, Any]) -> dict[str, list[str]]:
    output = {}
    for geometry_type in (
        "MultiPolygon",
        "MultiLineString",
        "GeometryCollection",
    ):
        categories = []
        for result in summary["category_results"]:
            wrapped = result.get("sarenv_wrapper_on_same_raw_response", {})
            geometry_summaries = wrapped.get("geometries", {})
            if any(
                item and item.get(f"has_{geometry_type}", False)
                for item in geometry_summaries.values()
            ):
                categories.append(result["category"])
        output[geometry_type] = categories
    return output


def raw_multi_categories(summary: dict[str, Any]) -> dict[str, list[str]]:
    return {
        geometry_type: [
            result["category"]
            for result in summary["category_results"]
            if result.get("geometry_summary", {}).get(
                f"has_{geometry_type}", False
            )
        ]
        for geometry_type in (
            "MultiPolygon",
            "MultiLineString",
            "GeometryCollection",
        )
    }


def status_error(result: dict[str, Any]) -> str:
    error = result.get("error")
    if not error:
        return ""
    return f"{error['exception_class']}: {error['exception_text']}"


def compact_geometry(result: dict[str, Any]) -> str:
    geometry = result.get("geometry_summary")
    if not geometry:
        return "—"
    counts = geometry["top_level_geometry_type_counts"]
    return ", ".join(f"{key}={value}" for key, value in counts.items()) or "0"


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    raw_multi = raw_multi_categories(summary)
    wrapper_multi = wrapper_multi_categories(summary)
    summary["multi_geometry_categories"] = {
        "raw_osmnx_rows": raw_multi,
        "sarenv_wrapper_outputs": wrapper_multi,
    }
    save_json(SUMMARY_PATH, summary)

    source_analysis = {
        "source_commit": INITIAL_MAIN_COMMIT,
        "classification": "fail-soft / partial export is allowed",
        "evidence": [
            {
                "file": "sarenv/io/osm_query.py",
                "function": "query_features",
                "lines": "38-43",
                "behavior": (
                    "catches every Exception from ox.features_from_polygon, logs a "
                    "warning, and returns None"
                ),
            },
            {
                "file": "sarenv/core/generation.py",
                "function": "process_feature_osm",
                "lines": "47-51, 63-67",
                "behavior": (
                    "converts wrapper None or no usable geometries to (category, None)"
                ),
            },
            {
                "file": "sarenv/core/generation.py",
                "function": "Environment._load_features",
                "lines": "335-360",
                "behavior": (
                    "submits all categories through an unconstrained ProcessPoolExecutor; "
                    "stores None for failed categories and continues collecting others"
                ),
            },
            {
                "file": "sarenv/core/generation.py",
                "function": "Environment.generate_heatmaps / get_combined_heatmap",
                "lines": "380-451",
                "behavior": (
                    "only creates tasks for non-None/non-empty categories and skips None "
                    "heatmaps when combining"
                ),
            },
            {
                "file": "sarenv/core/generation.py",
                "function": "DataGenerator.export_dataset",
                "lines": "725-741, 855-877",
                "behavior": (
                    "concatenates only non-None/non-empty feature GeoDataFrames and exports "
                    "the partial collection if at least one category succeeded"
                ),
            },
        ],
        "answer": (
            "B: main does not fail fast per category. A failed query becomes None, and "
            "remaining successful categories can still produce heatmap.npy and "
            "features.geojson."
        ),
    }
    save_json(SOURCE_ANALYSIS_PATH, source_analysis)

    initial_output = {
        "observation_time_context": "task start, before external branch switch",
        "path": str(
            DIAGNOSTIC_DIR.parent / "examples" / "sarenv_dataset" / "features.geojson"
        ),
        "file_size_bytes": 16246835,
        "last_write_time_local": "2026-08-25 15:42:53",
        "feature_count": 33683,
        "feature_type_counts": {"road": 15358, "structure": 18325},
        "present_categories": ["structure", "road"],
        "missing_categories": [
            "linear",
            "drainage",
            "water",
            "brush",
            "scrub",
            "woodland",
            "field",
            "rock",
        ],
        "bbox_wgs84": [
            10.134978815338195,
            55.05866125257006,
            10.44474559818122,
            55.23395669999999,
        ],
        "important_scope_note": (
            "This observed two-category file is in Denmark, not the Bristol Direct "
            "Small bounds, so it cannot be category-by-category correlated with Probe 4. "
            "The external branch switch later removed it from the working tree."
        ),
    }
    save_json(INITIAL_OUTPUT_PATH, initial_output)

    probe4_total = sum(
        float(result["elapsed_seconds"]) for result in summary["category_results"]
    )
    category_rows = []
    for result in summary["category_results"]:
        tags = json.dumps(result["tags"], ensure_ascii=False, separators=(",", ":"))
        error = status_error(result).replace("|", "\\|")
        category_rows.append(
            f"| {result['category']} | `{tags}` | {result['status']} | "
            f"{result.get('row_count', '—')}; {compact_geometry(result)} | "
            f"{result['elapsed_seconds']:.3f}s | {error or '—'} |"
        )

    present = comparison["present_categories"]
    missing = comparison["missing_categories"]
    report = f"""# Executive Summary

1. OSMnx 2.1.0 API is compatible with main's `features_from_polygon(..., tags=...)` call.
2. The first tiny direct query failed with `ConnectTimeout` after 102.211s.
3. Main's real wrapper also failed on the same tiny query after 102.232s, swallowed the request exception, and returned `None`.
4. Full Bristol Small: 9 `PASS_WITH_DATA`, 0 `PASS_EMPTY`, 1 `FAIL_QUERY` (rock). Structure needed a retry; rock failed both attempts.
5. Primary layer: intermittent network/Overpass availability. Fail-soft SAREnv control flow and concurrent production requests make partial exports possible. Geometry is downstream and is not the cause of missing acquisition categories.

# Environment

- Python 3.12.10
- OSMnx 2.1.0
- GeoPandas 1.1.3
- Shapely 2.1.2
- PyProj 3.7.2
- Task-start SAREnv state: `main@{INITIAL_MAIN_COMMIT}`
- Worktree externally switched during diagnostics to `radiation-evaluation@c3e200464f54b13be90f206e10e212eeff1d32b3`; probes that inspect main are locked to the task-start git object.

# Probe Results

| Probe | Result | Time | Key finding |
|---|---|---:|---|
| Probe 1 | PASS | 0.007s | offline API shape compatible; no network claim |
| Probe 2 | FAIL | 102.211s | direct tiny query hit `ConnectTimeout` |
| Probe 3 | FAIL | 102.232s | locked main wrapper returned `None` after same timeout |
| Probe 4 | COMPLETE | {probe4_total:.3f}s | 9 data, 0 empty, 1 query failure |
| Probe 5 | FAIL | 594.659s | alternate reached host but ended HTTP 500 |

# Full Small Category Results

| Category | Tags | Status | Count/geometry | Time | Error |
|---|---|---|---|---:|---|
{chr(10).join(category_rows)}

# Geometry Findings

- Raw OSMnx rows: no MultiPolygon, MultiLineString, or GeometryCollection in any successful category.
- SAREnv wrapper MultiPolygon: {', '.join(wrapper_multi['MultiPolygon'])}.
- SAREnv wrapper MultiLineString: {', '.join(wrapper_multi['MultiLineString'])}.
- SAREnv wrapper GeometryCollection: {', '.join(wrapper_multi['GeometryCollection'])}.
- No `GeometryCollection -> Multi* -> leaf` nesting occurred in this run. Observed paths were one container level followed by leaf geometries.
- Existing Bristol Small export contains one valid `linear` MultiLineString (2 LineStrings) and one valid `woodland` MultiPolygon (2 Polygons).

# Why the previous map was incomplete

The task-start two-category file had only structure/road but its bbox is Denmark, not Bristol, so it cannot be directly correlated with this Small probe. The existing Bristol Small file has {comparison['feature_count']} features across {len(present)} categories ({', '.join(present)}), missing {', '.join(missing)}. Probe 4 now found real drainage and water data, so their absence in that existing export was not a genuine regional empty result. Rock remains unknown geographically because both current attempts failed.

Main is fail-soft: `query_features` catches every query exception and returns `None`; `process_feature_osm` converts it to `(category, None)`; `_load_features` stores `None`; heatmap generation and feature export skip None categories and continue. Thus any subset that succeeds can yield a completed but partial dataset.

# Root Cause Assessment

1. Intermittent Overpass/network request-path failures — HIGH. Tiny direct failed; structure failed then succeeded; rock failed twice; alternate returned HTTP 500.
2. SAREnv fail-soft category handling plus unconstrained production `ProcessPoolExecutor` — HIGH as the mechanism that turns transient failures into a silently partial export.
3. Query/server load and category timing variability — MEDIUM as an amplifier. Successful categories ranged from 2.864s to 59.982s, but tiny queries also failed, so size is not the primary necessary cause.
4. MultiGeometry/rasterization limitations — LOW for missing categories. Multi* are created downstream by wrapper consolidation; they may explain individual warnings/features, not acquisition loss.

# Recommended Next Action

In the next implementation round only: first verify host egress/proxy and a reachable stable endpoint, then use a controlled raw-response cache, serialize/limit category requests, and make export fail loudly when required categories have `FAIL_QUERY`. Separately add recursive flattening before rasterization. None of these changes was applied here.

# Repository Integrity

No production source, production config, existing Small data, heatmap, or final-experiment result was written by these probes. All created scripts/logs/JSON/GeoJSON are under `OSM_diagnostics/`. Final `git status` and `git diff --check` are captured separately because the user/external branch switch changed repository state during the run.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"updated: {SUMMARY_PATH}")
    print(f"saved: {SOURCE_ANALYSIS_PATH}")
    print(f"saved: {INITIAL_OUTPUT_PATH}")
    print(f"saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
