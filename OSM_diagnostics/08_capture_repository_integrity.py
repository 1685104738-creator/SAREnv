"""Capture final read-only Git and protected-path integrity evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from diagnostic_common import DIAGNOSTIC_DIR, REPO_ROOT


OUTPUT_PATH = DIAGNOSTIC_DIR / "08_repository_integrity.txt"


def git(*arguments: str) -> tuple[int, str]:
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
    return completed.returncode, output.rstrip()


def file_fingerprint(relative_path: str) -> dict[str, object]:
    path = REPO_ROOT / relative_path
    if not path.exists():
        return {"path": str(path), "exists": False}
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_time": stat.st_mtime,
        "sha256": hasher.hexdigest(),
    }


def main() -> None:
    status_code, status = git("status", "--short")
    diff_check_code, diff_check = git("diff", "--check")
    branch_code, branch = git("branch", "--show-current")
    head_code, head = git("rev-parse", "HEAD")
    protected = [
        file_fingerprint(
            "examples/sarenv_dataset/sarenv_outputs/"
            "radiation_area_01_small_20m/features.geojson"
        ),
        file_fingerprint(
            "examples/sarenv_dataset/sarenv_outputs/"
            "radiation_area_01_small_20m/heatmap.npy"
        ),
        file_fingerprint(
            "examples/sarenv_dataset/sarenv_outputs/"
            "radiation_area_01_small_20m/metadata.json"
        ),
    ]
    lines = [
        "FINAL REPOSITORY INTEGRITY CAPTURE",
        "",
        f"branch (exit {branch_code}): {branch}",
        f"HEAD (exit {head_code}): {head}",
        "",
        f"git status --short (exit {status_code}):",
        status or "(clean)",
        "",
        f"git diff --check (exit {diff_check_code}):",
        diff_check or "(no whitespace errors)",
        "",
        "protected Bristol Small file fingerprints (read-only final state):",
        json.dumps(protected, indent=2),
        "",
        "Scope note: task began on main@0f2d344d17c0b9c18cec2176dd9b171beba727d6.",
        "The worktree externally switched branches during Probe 3; no diagnostic command performed checkout.",
        "All diagnostic writes are under OSM_diagnostics/.",
    ]
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

