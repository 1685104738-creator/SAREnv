"""Capture the pre-probe repository and Python/OSM environment."""

from __future__ import annotations

import ast
import inspect
import platform
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pyproj
import shapely
import sarenv

from diagnostic_common import DIAGNOSTIC_DIR, REPO_ROOT
from sarenv.io import osm_query


OUTPUT_PATH = DIAGNOSTIC_DIR / "environment.txt"
TASK_START_BRANCH = "main"
TASK_START_HEAD = "0f2d344d17c0b9c18cec2176dd9b171beba727d6"
TASK_START_STATUS = """?? examples/radiation/
?? examples/sarenv_dataset/
?? results/evaluation/
?? results/final_experiment/
?? results/radiation_aware_adaptive_entry/
?? results/radiation_aware_full/"""


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout
    if completed.stderr:
        output += completed.stderr
    return output.rstrip()


def query_api_calls() -> list[str]:
    source = inspect.getsource(osm_query.query_features)
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            calls.append(
                f"{ast.unparse(function)} keyword_args="
                f"{[keyword.arg for keyword in node.keywords]}"
            )
    return calls


def main() -> None:
    raw_status = git("status", "--short")
    baseline_status = "\n".join(
        line for line in raw_status.splitlines() if "OSM_diagnostics" not in line
    )
    lines = [
        "SAREnv OSM diagnostics environment",
        "",
        "Task-start observation (captured by the first read-only shell command,",
        "before OSM_diagnostics existed and before the external branch switch):",
        f"git branch: {TASK_START_BRANCH}",
        f"git HEAD: {TASK_START_HEAD}",
        "git status --short:",
        TASK_START_STATUS,
        "",
        "State when this environment file was generated (after external switch):",
        f"git branch: {git('branch', '--show-current')}",
        f"git HEAD: {git('rev-parse', 'HEAD')}",
        "git status --short, excluding this new OSM_diagnostics directory:",
        baseline_status or "(clean)",
        "git status --short, raw at capture time:",
        raw_status or "(clean)",
        "",
        f"Python version: {platform.python_version()} ({sys.version})",
        f"Python executable: {sys.executable}",
        f"sarenv module path: {Path(sarenv.__file__).resolve()}",
        f"sarenv query module path: {Path(osm_query.__file__).resolve()}",
        f"osmnx version: {ox.__version__}",
        f"geopandas version: {gpd.__version__}",
        f"shapely version: {shapely.__version__}",
        f"pyproj version: {pyproj.__version__}",
        "",
        f"ox.settings.overpass_url: {ox.settings.overpass_url}",
        f"ox.settings.requests_timeout: {ox.settings.requests_timeout}",
        f"ox.settings.use_cache: {ox.settings.use_cache}",
        f"ox.settings.cache_folder: {ox.settings.cache_folder}",
        f"ox.settings.overpass_rate_limit: {ox.settings.overpass_rate_limit}",
        f"ox.settings.overpass_settings: {ox.settings.overpass_settings}",
        "",
        "query_features signature: " + str(inspect.signature(osm_query.query_features)),
        "query API calls found in sarenv/io/osm_query.py:",
        *[f"- {call}" for call in query_api_calls()],
        "expected production call: ox.features_from_polygon(..., tags=tags_to_query)",
    ]
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
