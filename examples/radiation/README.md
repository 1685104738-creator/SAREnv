# Independent radiation examples

Run from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe examples\radiation\00_point_source.py
.\.venv\Scripts\python.exe examples\radiation\01_uniform_rectangle.py
.\.venv\Scripts\python.exe examples\radiation\02_irregular_uniform_polygon.py
.\.venv\Scripts\python.exe examples\radiation\03_zoned_polygon.py
```

All commands above are radiation-only and fully offline. They do not load a
SAR dataset, `features.geojson`, or `heatmap.npy`, and they never call OSM or
Overpass. Images are written below `examples/radiation/outputs/`.

`04_combined_sar_radiation_alignment.py` is a historical combined-display
example and is intentionally not part of this radiation-only workflow.
