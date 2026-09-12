# Executive Summary

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
- Task-start SAREnv state: `main@0f2d344d17c0b9c18cec2176dd9b171beba727d6`
- Worktree externally switched during diagnostics to `radiation-evaluation@c3e200464f54b13be90f206e10e212eeff1d32b3`; probes that inspect main are locked to the task-start git object.

# Probe Results

| Probe | Result | Time | Key finding |
|---|---|---:|---|
| Probe 1 | PASS | 0.007s | offline API shape compatible; no network claim |
| Probe 2 | FAIL | 102.211s | direct tiny query hit `ConnectTimeout` |
| Probe 3 | FAIL | 102.232s | locked main wrapper returned `None` after same timeout |
| Probe 4 | COMPLETE | 623.783s | 9 data, 0 empty, 1 query failure |
| Probe 5 | FAIL | 594.659s | alternate reached host but ended HTTP 500 |

# Full Small Category Results

| Category | Tags | Status | Count/geometry | Time | Error |
|---|---|---|---|---:|---|
| structure | `{"building":true,"man_made":true,"bridge":true,"tunnel":true}` | PASS_WITH_DATA | 1146; LineString=6, Point=2, Polygon=1138 | 107.718s | — |
| road | `{"highway":true,"tracktype":true}` | PASS_WITH_DATA | 265; LineString=215, Point=50 | 2.864s | — |
| linear | `{"railway":true,"barrier":true,"fence":true,"wall":true,"pipeline":true}` | PASS_WITH_DATA | 84; LineString=62, Point=21, Polygon=1 | 59.982s | — |
| drainage | `{"waterway":["drain","ditch","culvert","canal"]}` | PASS_WITH_DATA | 1; LineString=1 | 8.277s | — |
| water | `{"natural":["water","wetland"],"water":true,"wetland":true,"reservoir":true}` | PASS_WITH_DATA | 2; Polygon=2 | 54.137s | — |
| brush | `{"landuse":["grass"]}` | PASS_WITH_DATA | 17; Polygon=17 | 10.998s | — |
| scrub | `{"natural":"scrub"}` | PASS_WITH_DATA | 10; Polygon=10 | 34.656s | — |
| woodland | `{"landuse":["forest","wood"],"natural":"wood"}` | PASS_WITH_DATA | 15; Polygon=15 | 16.212s | — |
| field | `{"landuse":["farmland","farm","meadow"]}` | PASS_WITH_DATA | 35; Polygon=35 | 45.685s | — |
| rock | `{"natural":["rock","bare_rock","scree","cliff"]}` | FAIL_QUERY | —; — | 283.255s | requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='overpass-api.de', port=443): Max retries exceeded with url: /api/interpreter (Caused by ConnectTimeoutError(<HTTPSConnection(host='overpass-api.de', port=443) at 0x17e8d830ef0>, 'Connection to overpass-api.de timed out. (connect timeout=180)')) |

# Geometry Findings

- Raw OSMnx rows: no MultiPolygon, MultiLineString, or GeometryCollection in any successful category.
- SAREnv wrapper MultiPolygon: structure, water, brush, scrub, woodland, field.
- SAREnv wrapper MultiLineString: structure, linear.
- SAREnv wrapper GeometryCollection: structure, road, linear.
- No `GeometryCollection -> Multi* -> leaf` nesting occurred in this run. Observed paths were one container level followed by leaf geometries.
- Existing Bristol Small export contains one valid `linear` MultiLineString (2 LineStrings) and one valid `woodland` MultiPolygon (2 Polygons).

# Why the previous map was incomplete

The task-start two-category file had only structure/road but its bbox is Denmark, not Bristol, so it cannot be directly correlated with this Small probe. The existing Bristol Small file has 913 features across 7 categories (structure, road, linear, brush, scrub, woodland, field), missing drainage, water, rock. Probe 4 now found real drainage and water data, so their absence in that existing export was not a genuine regional empty result. Rock remains unknown geographically because both current attempts failed.

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
