# Independent radiation examples

Run from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe examples\radiation\01_uniform_rectangle.py
.\.venv\Scripts\python.exe examples\radiation\02_irregular_uniform_polygon.py
.\.venv\Scripts\python.exe examples\radiation\03_zoned_polygon.py
.\.venv\Scripts\python.exe examples\radiation\04_combined_sar_radiation_alignment.py
```

The first three examples are radiation-only and never load `heatmap.npy`. The
combined example is display-only: it overlays independent 30 m SAR and 1 m
radiation rasters using world-coordinate extents and does not resample either.
