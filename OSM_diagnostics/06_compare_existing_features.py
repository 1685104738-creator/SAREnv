"""Read-only comparison of the existing Bristol Small GeoJSON and mapping."""

from __future__ import annotations

import collections
import json

from shapely.geometry import shape

from diagnostic_common import (
    DIAGNOSTIC_DIR,
    SMALL_FEATURES_PATH,
    run_logged,
    save_json,
)
from sarenv.core.generation import DataGenerator


OUTPUT_PATH = DIAGNOSTIC_DIR / "06_existing_features_comparison.json"


def main() -> None:
    print("EXISTING BRISTOL SMALL FEATURES — READ-ONLY COMPARISON")
    mapping = DataGenerator().tags_mapping
    expected = list(mapping.keys())
    result = {
        "source_path": str(SMALL_FEATURES_PATH),
        "source_exists": SMALL_FEATURES_PATH.exists(),
        "expected_categories": expected,
        "tags_mapping": mapping,
    }
    if not SMALL_FEATURES_PATH.exists():
        result.update(
            {
                "present_categories": [],
                "missing_categories": expected,
                "feature_type_counts": {},
                "note": "Existing Bristol Small features.geojson was not found.",
            }
        )
    else:
        document = json.loads(SMALL_FEATURES_PATH.read_text(encoding="utf-8"))
        features = document.get("features", [])
        counts = collections.Counter(
            feature.get("properties", {}).get("feature_type")
            for feature in features
        )
        present = [category for category in expected if counts.get(category, 0) > 0]
        missing = [category for category in expected if counts.get(category, 0) == 0]
        geometry_counts = collections.Counter(
            feature.get("geometry", {}).get("type")
            for feature in features
            if feature.get("geometry")
        )
        geometry_counts_by_category = collections.defaultdict(collections.Counter)
        multi_geometry_features = []
        for feature in features:
            category = feature.get("properties", {}).get("feature_type")
            geometry_document = feature.get("geometry")
            if not geometry_document:
                continue
            geometry_type = geometry_document.get("type")
            geometry_counts_by_category[category][geometry_type] += 1
            if geometry_type in {
                "MultiPolygon",
                "MultiLineString",
                "GeometryCollection",
            }:
                geometry = shape(geometry_document)
                multi_geometry_features.append(
                    {
                        "feature_id": feature.get("id"),
                        "feature_type": category,
                        "geometry_type": geometry_type,
                        "child_count": len(geometry.geoms),
                        "child_types": [child.geom_type for child in geometry.geoms],
                        "is_valid": geometry.is_valid,
                        "is_empty": geometry.is_empty,
                    }
                )
        result.update(
            {
                "source_size_bytes": SMALL_FEATURES_PATH.stat().st_size,
                "source_modified_time": SMALL_FEATURES_PATH.stat().st_mtime,
                "top_level_keys": list(document.keys()),
                "bbox": document.get("bbox"),
                "bounds": document.get("bounds"),
                "center_point": document.get("center_point"),
                "feature_count": len(features),
                "feature_type_counts": dict(counts),
                "geojson_geometry_type_counts": dict(geometry_counts),
                "geometry_type_counts_by_category": {
                    category: dict(counts)
                    for category, counts in geometry_counts_by_category.items()
                },
                "multi_geometry_features": multi_geometry_features,
                "present_categories": present,
                "missing_categories": missing,
                "note": (
                    "Missing in an exported file does not by itself distinguish "
                    "PASS_EMPTY from FAIL_QUERY; correlate with Probe 4."
                ),
            }
        )
    save_json(OUTPUT_PATH, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_logged("06_existing_features_comparison.log", main)
