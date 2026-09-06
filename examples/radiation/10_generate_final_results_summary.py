"""Generate dissertation-ready figures and tables from completed experiments.

This is a read-only post-processing entry point.  It never runs a mission and
never writes inside an existing experiment directory.  All generated artefacts
are written to ``results/final_results_summary``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SENSITIVITY_ROOT = (
    REPOSITORY_ROOT / "results" / "50m_intensity_sensitivity_20_new_seeds"
)
REPRESENTATIVE_ROOT = REPOSITORY_ROOT / "results" / "final_experiment"
DATASET_ROOT = (
    REPOSITORY_ROOT
    / "examples"
    / "sarenv_dataset"
    / "sarenv_outputs"
    / "radiation_area_01_small_20m"
)
OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "final_results_summary"

INTENSITIES = (0.25, 0.5, 1.0, 2.0, 4.0)
INTENSITY_LABELS = ("0.25×", "0.5×", "1×", "2×", "4×")
MORPHOLOGIES = ("Single", "Multi", "Surface")
MORPHOLOGY_COLOURS = {
    "Single": "#0072B2",
    "Multi": "#D55E00",
    "Surface": "#6A3D9A",
}
MORPHOLOGY_MARKERS = {"Single": "o", "Multi": "s", "Surface": "^"}
T_CRITICAL_95_DF19 = 2.093024054408263

METRICS = {
    "sar_likelihood": "delta_percent_sar_likelihood_score",
    "sar_time": "delta_percent_sar_time_discounted_score",
    "found": "delta_survivors_found",
    "survivor_total": "delta_percent_survivor_total_pre_discovery_dose",
    "survivor_p95": "delta_percent_survivor_p95_pre_discovery_dose",
    "survivor_max": "delta_percent_survivor_max_pre_discovery_dose",
    "uav_excess": "delta_percent_uav_cumulative_excess_dose",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.titlesize": 12.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(OUTPUT_ROOT / f"{stem}.png", dpi=320, bbox_inches="tight")
    figure.savefig(OUTPUT_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    intensity = pd.read_csv(SENSITIVITY_ROOT / "intensity_summary.csv")
    status = pd.read_csv(SENSITIVITY_ROOT / "run_status.csv")
    config = json.loads(
        (SENSITIVITY_ROOT / "experiment_config.json").read_text(encoding="utf-8")
    )
    return intensity, status, config


def _validate_inputs(
    intensity: pd.DataFrame,
    status: pd.DataFrame,
    config: dict[str, object],
) -> None:
    required_columns = {
        "seed",
        "intensity_multiplier",
        "independent_final_validation",
        "Y",
        "sigma_m",
        "influence_radius_m",
        "scenario",
        "radiation_positions_m",
        "radiation_episodes",
        "radiation_steps",
        *METRICS.values(),
    }
    missing = required_columns.difference(intensity.columns)
    if missing:
        raise ValueError(f"intensity_summary.csv is missing columns: {sorted(missing)}")
    if len(intensity) != 300:
        raise ValueError(f"Expected 300 scenario rows, found {len(intensity)}.")
    if intensity.duplicated(["seed", "intensity_multiplier", "scenario"]).any():
        raise ValueError("Duplicate seed × intensity × morphology rows were found.")
    if set(intensity["seed"]) != set(config["seeds"]):
        raise ValueError("CSV seeds do not match experiment_config.json.")
    if tuple(sorted(intensity["intensity_multiplier"].unique())) != INTENSITIES:
        raise ValueError("Unexpected intensity levels in intensity_summary.csv.")
    if set(intensity["scenario"]) != set(MORPHOLOGIES):
        raise ValueError("Unexpected morphology labels in intensity_summary.csv.")
    if len(status) != 100 or set(status["status"]) != {"SUCCESS"}:
        raise ValueError("All 100 completed seed × intensity runs are required.")
    if status["error"].notna().any():
        raise ValueError("run_status.csv contains errors.")
    fixed = {
        "Y": 0.05,
        "sigma_m": 40.0,
        "influence_radius_m": 120.0,
    }
    for column, expected in fixed.items():
        values = intensity[column].unique()
        if len(values) != 1 or not math.isclose(float(values[0]), expected):
            raise ValueError(f"Frozen {column} does not equal {expected}.")
    validation_flags = intensity.groupby("intensity_multiplier")[
        "independent_final_validation"
    ].unique()
    for multiplier in INTENSITIES:
        expected = multiplier == 1.0
        if validation_flags[multiplier].tolist() != [expected]:
            raise ValueError("Only the 1× rows may be final-validation rows.")
    geometry_variants = intensity.groupby(["seed", "scenario"])[
        "radiation_positions_m"
    ].nunique()
    if not (geometry_variants == 1).all():
        raise ValueError("Intensity comparisons do not preserve paired geometry.")
    if intensity[list(METRICS.values())].isna().any().any():
        raise ValueError("A required final metric is missing.")
    _validate_report_summary(intensity)


def _validate_report_summary(intensity: pd.DataFrame) -> None:
    """Reconcile the rounded summary table already published in the run report."""
    report_lines = (SENSITIVITY_ROOT / "intensity_report.md").read_text(
        encoding="utf-8"
    ).splitlines()
    parsed: dict[tuple[float, str], list[str]] = {}
    for line in report_lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 16 or cells[1] not in MORPHOLOGIES:
            continue
        label = cells[0]
        if not label.endswith("x"):
            continue
        multiplier = float(label[:-1])
        parsed.setdefault((multiplier, cells[1]), cells)
    if len(parsed) != 15:
        raise ValueError("Could not reconcile all 15 report summary rows.")

    columns = (
        METRICS["sar_likelihood"],
        METRICS["sar_time"],
        METRICS["found"],
        METRICS["survivor_total"],
        METRICS["survivor_p95"],
        METRICS["survivor_max"],
        METRICS["uav_excess"],
    )
    for multiplier in INTENSITIES:
        for morphology in MORPHOLOGIES:
            group = intensity[
                intensity["intensity_multiplier"].eq(multiplier)
                & intensity["scenario"].eq(morphology)
            ]
            cells = parsed[(multiplier, morphology)]
            reported_means = [float(value) for value in cells[2:9]]
            computed_means = [round(float(group[column].mean()), 2) for column in columns]
            if any(
                not math.isclose(reported, computed, abs_tol=0.0051)
                for reported, computed in zip(reported_means, computed_means, strict=True)
            ):
                raise ValueError(
                    f"Report means disagree for {morphology} at {multiplier}×."
                )
            reported_counts = [int(value) for value in cells[9:12]]
            computed_counts = [
                int((group[column] < 0.0).sum())
                for column in (
                    METRICS["survivor_total"],
                    METRICS["survivor_p95"],
                    METRICS["survivor_max"],
                )
            ]
            if reported_counts != computed_counts:
                raise ValueError(
                    f"Report improvement counts disagree for {morphology} "
                    f"at {multiplier}×."
                )


def _group_summary(intensity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for morphology in MORPHOLOGIES:
        for multiplier in INTENSITIES:
            group = intensity[
                intensity["scenario"].eq(morphology)
                & intensity["intensity_multiplier"].eq(multiplier)
            ]
            row: dict[str, object] = {
                "morphology": morphology,
                "intensity": multiplier,
                "n": len(group),
            }
            for metric_name, column in METRICS.items():
                values = group[column].astype(float)
                row[f"{metric_name}_mean"] = float(values.mean())
                row[f"{metric_name}_median"] = float(values.median())
                row[f"{metric_name}_sd"] = float(values.std(ddof=1))
                row[f"{metric_name}_min"] = float(values.min())
                row[f"{metric_name}_max"] = float(values.max())
                row[f"{metric_name}_improved"] = int((values < 0.0).sum())
            for name, column in (
                ("radiation_steps", "radiation_steps"),
                ("radiation_episodes", "radiation_episodes"),
            ):
                values = group[column].astype(float)
                row[f"{name}_mean"] = float(values.mean())
                row[f"{name}_sd"] = float(values.std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def _load_route(path: Path) -> pd.DataFrame:
    route = pd.read_csv(path)
    required = {"x_m", "y_m"}
    if not required.issubset(route.columns):
        raise ValueError(f"Route is missing coordinates: {path}")
    return route


def _representative_context() -> tuple[
    tuple[float, float, float, float], np.ndarray
]:
    metadata = json.loads((DATASET_ROOT / "metadata.json").read_text(encoding="utf-8"))
    survivors = json.loads(
        (DATASET_ROOT / "lost_persons.json").read_text(encoding="utf-8")
    )
    bounds = tuple(float(value) for value in metadata["bounds_projected"])
    points = np.asarray(survivors["coordinates_projected"], dtype=float)
    return bounds, points


def _point_truth_grid(
    config: dict[str, object],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    field = np.full_like(x_grid, float(config["background_rate"]), dtype=float)
    reference_z = float(config["value_reference_height_m"])
    for source in config["sources"]:
        dx = x_grid - float(source["x_m"])
        dy = y_grid - float(source["y_m"])
        dz = reference_z - float(source["source_height_m"])
        distance_squared = dx * dx + dy * dy + dz * dz
        core_squared = float(source["core_radius_m"]) ** 2
        reference_squared = float(source["reference_distance_m"]) ** 2
        field += float(source["reference_excess_uSv_h"]) * (
            reference_squared + core_squared
        ) / (distance_squared + core_squared)
    return field


def _surface_truth_grid(
    config: dict[str, object],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    metadata = json.loads(
        (
            REPRESENTATIVE_ROOT
            / "03_uniform_surface"
            / "truth"
            / "ground_1m"
            / "surface_metadata.json"
        ).read_text(encoding="utf-8")
    )
    raster = np.load(
        REPRESENTATIVE_ROOT
        / "03_uniform_surface"
        / "truth"
        / "ground_1m"
        / "collision_air_kerma_rate_uGy_h.npy"
    )
    grid = metadata["grid"]
    min_x, min_y, max_x, max_y = (float(value) for value in grid["bounds"])
    resolution = float(grid["resolution_m"])
    rows = np.floor((y_grid - min_y) / resolution).astype(int)
    cols = np.floor((x_grid - min_x) / resolution).astype(int)
    inside = (
        (x_grid >= min_x)
        & (x_grid < max_x)
        & (y_grid >= min_y)
        & (y_grid < max_y)
        & (rows >= 0)
        & (rows < raster.shape[0])
        & (cols >= 0)
        & (cols < raster.shape[1])
    )
    field = np.full_like(x_grid, float(config["background_rate"]), dtype=float)
    field[inside] += raster[rows[inside], cols[inside]]
    return field


def _relative_xy(
    x_values: np.ndarray | pd.Series,
    y_values: np.ndarray | pd.Series,
    centre: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(x_values, dtype=float) - centre[0],
        np.asarray(y_values, dtype=float) - centre[1],
    )


def figure_01_route_comparison() -> None:
    bounds, survivors = _representative_context()
    centre = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
    resolution = 4.0
    x_values = np.arange(bounds[0] + resolution / 2.0, bounds[2], resolution)
    y_values = np.arange(bounds[1] + resolution / 2.0, bounds[3], resolution)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    relative_extent = (
        bounds[0] - centre[0],
        bounds[2] - centre[0],
        bounds[1] - centre[1],
        bounds[3] - centre[1],
    )
    survivor_x, survivor_y = _relative_xy(survivors[:, 0], survivors[:, 1], centre)
    original = _load_route(
        REPRESENTATIVE_ROOT / "baseline_original" / "simulation" / "mission_steps.csv"
    )

    scenario_directories = (
        ("01_single_point", "Single point"),
        ("02_multi_point", "Multi point"),
        ("03_uniform_surface", "Surface"),
    )
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(10.5, 13.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row_index, (directory, row_label) in enumerate(scenario_directories):
        scenario_root = REPRESENTATIVE_ROOT / directory
        config = json.loads(
            (scenario_root / "scenario_config.json").read_text(encoding="utf-8")
        )
        if directory == "03_uniform_surface":
            truth = _surface_truth_grid(config, x_grid, y_grid)
        else:
            truth = _point_truth_grid(config, x_grid, y_grid)
        positive = truth[truth > 0.0]
        norm = LogNorm(
            vmin=max(float(np.percentile(positive, 2.0)), 1e-8),
            vmax=float(positive.max()),
        )
        aware = _load_route(
            scenario_root / "radiation_aware" / "simulation" / "mission_steps.csv"
        )
        image = None
        for column_index, (route, column_title) in enumerate(
            ((original, "Original Greedy"), (aware, "Radiation-Aware Greedy"))
        ):
            axis = axes[row_index, column_index]
            axis.set_facecolor("#07070A")
            image = axis.imshow(
                truth,
                origin="lower",
                extent=relative_extent,
                cmap="magma",
                norm=norm,
                interpolation="nearest",
                rasterized=True,
            )
            route_x, route_y = _relative_xy(route["x_m"], route["y_m"], centre)
            axis.plot(route_x, route_y, color="white", linewidth=0.65, alpha=0.92)
            axis.scatter(
                survivor_x,
                survivor_y,
                s=8,
                facecolors="none",
                edgecolors="#D9EAF7",
                linewidths=0.45,
                alpha=0.8,
            )
            axis.scatter(
                route_x[0],
                route_y[0],
                marker="D",
                s=34,
                color="#56B4E9",
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
            )
            if directory != "03_uniform_surface":
                source_x, source_y = _relative_xy(
                    [source["x_m"] for source in config["sources"]],
                    [source["y_m"] for source in config["sources"]],
                    centre,
                )
                axis.scatter(
                    source_x,
                    source_y,
                    marker="*",
                    s=82,
                    color="#F0E442",
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=7,
                )
            else:
                polygon = np.asarray(config["polygon_exterior_coordinates"], dtype=float)
                polygon_x, polygon_y = _relative_xy(
                    polygon[:, 0], polygon[:, 1], centre
                )
                axis.plot(
                    polygon_x,
                    polygon_y,
                    color="#56B4E9",
                    linestyle="--",
                    linewidth=1.2,
                )
            axis.set_aspect("equal")
            axis.set_title(column_title if row_index == 0 else "")
            if column_index == 0:
                axis.set_ylabel(f"{row_label}\nNorthing offset (m)")
            if row_index == 2:
                axis.set_xlabel("Easting offset (m)")
        assert image is not None
        unit = str(config["radiation_rate_unit"])
        colour_label = (
            f"Ground-equivalent total rate ({unit})"
            if directory != "03_uniform_surface"
            else f"Ground excess collision air kerma rate ({unit})"
        )
        colourbar = figure.colorbar(image, ax=axes[row_index, :], shrink=0.82, pad=0.015)
        colourbar.set_label(colour_label)

    handles = (
        Line2D([0], [0], color="white", linewidth=1.6, label="Executed route"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#D9EAF7",
            label="Survivor",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="#56B4E9",
            markeredgecolor="black",
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#F0E442",
            markeredgecolor="black",
            markersize=10,
            label="Point source / surface boundary",
        ),
    )
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        columnspacing=1.2,
    )
    figure.suptitle("Representative route changes across radiation morphologies")
    _save_figure(figure, "figure_01_route_comparison")


def _draw_distribution_panel(
    axis: plt.Axes,
    validation: pd.DataFrame,
    column: str,
    title: str,
) -> None:
    positions = np.arange(len(MORPHOLOGIES), dtype=float)
    groups = [
        validation[validation["scenario"].eq(morphology)]
        .sort_values("seed")[column]
        .to_numpy(dtype=float)
        for morphology in MORPHOLOGIES
    ]
    boxes = axis.boxplot(
        groups,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.25},
        whiskerprops={"color": "#4D4D4D", "linewidth": 0.9},
        capprops={"color": "#4D4D4D", "linewidth": 0.9},
    )
    for patch, morphology in zip(boxes["boxes"], MORPHOLOGIES, strict=True):
        patch.set_facecolor(MORPHOLOGY_COLOURS[morphology])
        patch.set_alpha(0.28)
        patch.set_edgecolor(MORPHOLOGY_COLOURS[morphology])
    for index, (morphology, values) in enumerate(zip(MORPHOLOGIES, groups, strict=True)):
        jitter = np.linspace(-0.16, 0.16, values.size)
        axis.scatter(
            index + jitter,
            values,
            s=22,
            marker=MORPHOLOGY_MARKERS[morphology],
            facecolor=MORPHOLOGY_COLOURS[morphology],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.85,
            zorder=3,
        )
        axis.scatter(
            index,
            values.mean(),
            marker="D",
            s=34,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xticks(positions, MORPHOLOGIES)
    axis.set_title(title)
    axis.set_ylabel("Radiation-Aware change relative to Original (%)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)


def figure_02_validation_distribution(intensity: pd.DataFrame) -> None:
    validation = intensity[
        intensity["independent_final_validation"].astype(bool)
    ].copy()
    if len(validation) != 60:
        raise ValueError("The 1× validation must contain 60 morphology rows.")
    panels = (
        (METRICS["survivor_total"], "A  Survivor total exposure"),
        (METRICS["survivor_p95"], "B  Survivor P95 exposure"),
        (METRICS["survivor_max"], "C  Survivor maximum exposure"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.5), constrained_layout=True)
    for axis, (column, title) in zip(axes, panels, strict=True):
        _draw_distribution_panel(axis, validation, column, title)
    axes[0].set_ylim(-115, 230)
    axes[1].set_ylim(-115, 260)
    axes[2].set_ylim(-120, 550)
    inset = axes[2].inset_axes([0.49, 0.55, 0.48, 0.38])
    _draw_distribution_panel(
        inset,
        validation,
        METRICS["survivor_max"],
        "Central range",
    )
    inset.set_ylim(-110, 110)
    inset.set_ylabel("")
    inset.tick_params(labelsize=6.5)
    inset.title.set_fontsize(7.5)
    axes[2].indicate_inset_zoom(inset, edgecolor="#4D4D4D", alpha=0.7)
    handles = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            label="Individual seed",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Mean",
        ),
    )
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Independent 1× validation across 20 unseen seeds")
    _save_figure(figure, "figure_02_validation_distribution")


def _mean_and_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=float)
    mean = float(array.mean())
    ci = T_CRITICAL_95_DF19 * float(array.std(ddof=1)) / math.sqrt(array.size)
    return mean, ci


def figure_03_intensity_sensitivity(intensity: pd.DataFrame) -> None:
    panels = (
        (METRICS["survivor_total"], "A  Survivor total exposure"),
        (METRICS["survivor_p95"], "B  Survivor P95 exposure"),
    )
    x_positions = np.arange(len(INTENSITIES), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for axis, (column, title) in zip(axes, panels, strict=True):
        for morphology in MORPHOLOGIES:
            means = []
            cis = []
            for multiplier in INTENSITIES:
                values = intensity[
                    intensity["scenario"].eq(morphology)
                    & intensity["intensity_multiplier"].eq(multiplier)
                ][column]
                mean, ci = _mean_and_ci(values)
                means.append(mean)
                cis.append(ci)
            axis.errorbar(
                x_positions,
                means,
                yerr=cis,
                color=MORPHOLOGY_COLOURS[morphology],
                marker=MORPHOLOGY_MARKERS[morphology],
                markersize=5.5,
                linewidth=1.5,
                capsize=2.5,
                label=morphology,
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.axvline(2.0, color="#666666", linestyle=":", linewidth=1.0)
        axis.set_xticks(x_positions, INTENSITY_LABELS)
        axis.set_xlabel("Radiation intensity multiplier")
        axis.set_ylabel("Mean Radiation-Aware change (%)")
        axis.set_title(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    axes[0].legend(frameon=False, loc="lower left")
    figure.suptitle(
        "Survivor-risk sensitivity to radiation intensity "
        "(mean and 95% confidence interval)"
    )
    _save_figure(figure, "figure_03_intensity_sensitivity")


def figure_04_improvement_consistency(intensity: pd.DataFrame) -> None:
    panels = (
        (METRICS["survivor_total"], "A  Survivor Total"),
        (METRICS["survivor_p95"], "B  Survivor P95"),
        (METRICS["survivor_max"], "C  Survivor Maximum"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.3, 3.7), constrained_layout=True)
    image = None
    for axis, (column, title) in zip(axes, panels, strict=True):
        counts = np.empty((len(MORPHOLOGIES), len(INTENSITIES)), dtype=int)
        for row_index, morphology in enumerate(MORPHOLOGIES):
            for col_index, multiplier in enumerate(INTENSITIES):
                values = intensity[
                    intensity["scenario"].eq(morphology)
                    & intensity["intensity_multiplier"].eq(multiplier)
                ][column]
                counts[row_index, col_index] = int((values < 0.0).sum())
        image = axis.imshow(counts, cmap="Blues", vmin=0, vmax=20, aspect="auto")
        for row_index in range(counts.shape[0]):
            for col_index in range(counts.shape[1]):
                count = counts[row_index, col_index]
                axis.text(
                    col_index,
                    row_index,
                    f"{count}/20",
                    ha="center",
                    va="center",
                    color="white" if count >= 12 else "black",
                    fontweight="bold",
                    fontsize=9.5,
                )
        axis.add_patch(
            Rectangle(
                (1.5, -0.5),
                1.0,
                len(MORPHOLOGIES),
                fill=False,
                edgecolor="black",
                linewidth=2.0,
            )
        )
        axis.set_xticks(np.arange(len(INTENSITIES)), INTENSITY_LABELS)
        axis.set_yticks(np.arange(len(MORPHOLOGIES)), MORPHOLOGIES)
        axis.set_xlabel("Radiation intensity multiplier")
        axis.set_title(title)
    assert image is not None
    colourbar = figure.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    colourbar.set_label("Seeds with lower exposure")
    figure.suptitle("Consistency of survivor-exposure improvement")
    figure.text(
        0.5,
        -0.025,
        "A black border marks the independent 1× validation condition. No outliers are excluded.",
        ha="center",
        fontsize=8.5,
    )
    _save_figure(figure, "figure_04_improvement_consistency")


def figure_05_sar_radiation_tradeoff(summary: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7.7, 5.8), constrained_layout=True)
    colour_map = plt.get_cmap("viridis")
    intensity_colours = {
        multiplier: colour_map(index / (len(INTENSITIES) - 1))
        for index, multiplier in enumerate(INTENSITIES)
    }
    for _, row in summary.iterrows():
        axis.scatter(
            row["sar_time_mean"],
            row["survivor_p95_mean"],
            s=74,
            marker=MORPHOLOGY_MARKERS[str(row["morphology"])],
            facecolor=intensity_colours[float(row["intensity"])],
            edgecolor="black",
            linewidth=0.65,
            zorder=3,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Time-discounted SAR change (%)")
    axis.set_ylabel("Survivor P95 exposure change (%)")
    axis.set_title("Mean SAR–radiation trade-off across 20 paired seeds")
    axis.grid(color="#D9D9D9", linewidth=0.55, alpha=0.75)
    morphology_handles = [
        Line2D(
            [0],
            [0],
            marker=MORPHOLOGY_MARKERS[morphology],
            color="none",
            markerfacecolor="#BBBBBB",
            markeredgecolor="black",
            markersize=7,
            label=morphology,
        )
        for morphology in MORPHOLOGIES
    ]
    intensity_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=intensity_colours[multiplier],
            markeredgecolor="black",
            markersize=6.5,
            label=label,
        )
        for multiplier, label in zip(INTENSITIES, INTENSITY_LABELS, strict=True)
    ]
    first_legend = axis.legend(
        handles=morphology_handles,
        title="Morphology",
        frameon=False,
        loc="upper left",
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=intensity_handles,
        title="Intensity",
        frameon=False,
        loc="lower right",
        ncol=1,
    )
    axis.text(
        0.01,
        0.01,
        "Lower-left: lower survivor exposure with lower SAR performance",
        transform=axis.transAxes,
        fontsize=8,
        color="#444444",
        va="bottom",
    )
    _save_figure(figure, "figure_05_sar_radiation_tradeoff")


def figure_06_radiation_mode_behaviour(intensity: pd.DataFrame) -> None:
    x_positions = np.arange(len(INTENSITIES), dtype=float)
    display_offsets = {"Single": -0.045, "Multi": 0.045, "Surface": 0.0}
    panels = (
        ("radiation_steps", "A  Radiation-mode steps", "Mean radiation-mode steps"),
        ("radiation_episodes", "B  Radiation episodes", "Mean episode count"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for axis, (column, title, ylabel) in zip(axes, panels, strict=True):
        for morphology in MORPHOLOGIES:
            means = []
            cis = []
            for multiplier in INTENSITIES:
                values = intensity[
                    intensity["scenario"].eq(morphology)
                    & intensity["intensity_multiplier"].eq(multiplier)
                ][column]
                mean, ci = _mean_and_ci(values)
                means.append(mean)
                cis.append(ci)
            axis.errorbar(
                x_positions + display_offsets[morphology],
                means,
                yerr=cis,
                color=MORPHOLOGY_COLOURS[morphology],
                marker=MORPHOLOGY_MARKERS[morphology],
                markersize=5.5,
                linewidth=1.5,
                capsize=2.5,
                label=morphology,
            )
        axis.axvline(2.0, color="#666666", linestyle=":", linewidth=1.0)
        axis.set_xticks(x_positions, INTENSITY_LABELS)
        axis.set_xlabel("Radiation intensity multiplier")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    axes[0].axhline(
        3600.0,
        color="#555555",
        linestyle="--",
        linewidth=0.9,
        label="Mission move ceiling",
    )
    axes[0].set_ylim(0.0, 3800.0)
    axes[0].legend(frameon=False, loc="center right")
    figure.suptitle(
        "Radiation-mode engagement across intensity levels "
        "(mean and 95% confidence interval)"
    )
    _save_figure(figure, "figure_06_radiation_mode_behaviour")


def figure_07_representative_estimated_map() -> None:
    simulation = (
        REPRESENTATIVE_ROOT
        / "01_single_point"
        / "radiation_aware"
        / "simulation"
    )
    npz_path = simulation / "estimated_radiation_grid.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        estimate = np.asarray(data["mean"], dtype=float)
        observed = np.asarray(data["observed_mask"], dtype=bool)
        bounds = tuple(float(value) for value in data["bounds"])
        resolution = float(data["resolution_m"])
        reference_height = float(data["value_reference_height_m"])
        quantity = str(data["quantity"].item())
        unit = str(data["unit"].item())
    if estimate.shape != observed.shape or not math.isclose(resolution, 1.0):
        raise ValueError("Unexpected representative estimate-grid contract.")
    resolved = json.loads(
        (simulation / "resolved_parameters.json").read_text(encoding="utf-8")
    )
    if (
        resolved["measurement_quantity"] != quantity
        or float(resolved["value_reference_height_m"]) != reference_height
    ):
        raise ValueError("NPZ metadata conflicts with frozen resolved parameters.")
    route = _load_route(simulation / "mission_steps.csv")
    measurements = pd.read_csv(simulation / "radiation_measurements.csv")
    positive = estimate[observed & (estimate > 0.0)]
    masked = np.ma.masked_where(~observed, estimate)
    centre = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
    relative_extent = (
        bounds[0] - centre[0],
        bounds[2] - centre[0],
        bounds[1] - centre[1],
        bounds[3] - centre[1],
    )
    route_x, route_y = _relative_xy(route["x_m"], route["y_m"], centre)
    measurement_x, measurement_y = _relative_xy(
        measurements["x_m"], measurements["y_m"], centre
    )
    figure, axis = plt.subplots(figsize=(7.5, 6.7), constrained_layout=True)
    axis.set_facecolor("#E6E6E6")
    image = axis.imshow(
        masked,
        origin="lower",
        extent=relative_extent,
        cmap="magma",
        norm=LogNorm(
            vmin=max(float(np.percentile(positive, 1.0)), 1e-8),
            vmax=float(np.percentile(positive, 99.9)),
        ),
        interpolation="nearest",
        rasterized=True,
    )
    axis.plot(
        route_x,
        route_y,
        color="#56B4E9",
        linewidth=0.75,
        alpha=0.9,
        label="Executed UAV trajectory",
    )
    display_stride = 60
    axis.scatter(
        measurement_x[::display_stride],
        measurement_y[::display_stride],
        s=13,
        facecolor="#F0E442",
        edgecolor="black",
        linewidth=0.35,
        label="Measurement locations (every 60th shown)",
        zorder=4,
    )
    axis.scatter(
        route_x[0],
        route_y[0],
        marker="D",
        s=38,
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        label="Start",
        zorder=5,
    )
    colourbar = figure.colorbar(image, ax=axis, shrink=0.88)
    colourbar.set_label(f"Estimated ground-equivalent excess rate ({unit})")
    axis.set_xlabel("Easting offset from estimate-grid centre (m)")
    axis.set_ylabel("Northing offset from estimate-grid centre (m)")
    axis.set_aspect("equal")
    axis.legend(frameon=True, facecolor="white", framealpha=0.88, loc="upper right")
    axis.set_title("Representative online radiation estimate: Single point")
    figure.text(
        0.5,
        -0.015,
        (
            f"Frozen legacy contract: {quantity.replace('_', ' ')} at "
            f"{reference_height:g} m. Qualitative capability example only; "
            "not map-reconstruction validation."
        ),
        ha="center",
        fontsize=8.2,
    )
    _save_figure(figure, "figure_07_representative_estimated_radiation_map")


def _signed(value: float, decimals: int = 2, suffix: str = "") -> str:
    return f"{value:+.{decimals}f}{suffix}"


def _table_rows(summary: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for morphology in MORPHOLOGIES:
        for multiplier, label in zip(INTENSITIES, INTENSITY_LABELS, strict=True):
            record = summary[
                summary["morphology"].eq(morphology)
                & summary["intensity"].eq(multiplier)
            ].iloc[0]
            intensity_label = (
                f"{label} (Independent Final Validation)"
                if multiplier == 1.0
                else label
            )
            rows.append(
                {
                    "Morphology": morphology,
                    "Intensity": intensity_label,
                    "SAR likelihood Δ": _signed(record["sar_likelihood_mean"], suffix="%"),
                    "Time-discounted SAR Δ": _signed(record["sar_time_mean"], suffix="%"),
                    "Found Δ": _signed(record["found_mean"]),
                    "Survivor Total Δ": _signed(record["survivor_total_mean"], suffix="%"),
                    "Survivor P95 Δ": _signed(record["survivor_p95_mean"], suffix="%"),
                    "Survivor Maximum Δ": _signed(record["survivor_max_mean"], suffix="%"),
                    "UAV excess dose Δ": _signed(record["uav_excess_mean"], suffix="%"),
                    "Total improved seeds": f"{int(record['survivor_total_improved'])}/20",
                    "P95 improved seeds": f"{int(record['survivor_p95_improved'])}/20",
                    "Maximum improved seeds": f"{int(record['survivor_max_improved'])}/20",
                }
            )
    return rows


def _markdown_table(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0])
    lines = [
        "# Table 1. Final radiation-aware results",
        "",
        (
            "Values are means across 20 paired seeds. Negative radiation-exposure "
            "values indicate reductions; negative SAR values indicate performance "
            "loss; negative Found values indicate fewer survivors found. Positive "
            "UAV values indicate greater UAV excess dose. The bold 1× rows are the "
            "independent final validation; other intensities are paired sensitivity tests."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [row[header] for header in headers]
        if "Independent Final Validation" in row["Intensity"]:
            values = [f"**{value}**" for value in values]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    replacements = {
        "%": r"\%",
        "×": r"$\times$",
        "Δ": r"$\Delta$",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _latex_table(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0])
    header_labels = (
        "Morphology",
        "Intensity",
        r"SAR likelihood $\Delta$",
        r"Time SAR $\Delta$",
        r"Found $\Delta$",
        r"Total $\Delta$",
        r"P95 $\Delta$",
        r"Maximum $\Delta$",
        r"UAV excess $\Delta$",
        "Total improved",
        "P95 improved",
        "Max improved",
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Mean paired changes for Radiation-Aware relative to Original "
            r"Greedy across 20 seeds. The bold 1$\times$ rows are the independent "
            r"final validation; other intensities are sensitivity tests.}"
        ),
        r"\label{tab:final-radiation-results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"\toprule",
        " & ".join(header_labels) + r" \\",
        r"\midrule",
    ]
    previous_morphology = None
    for row in rows:
        if previous_morphology is not None and row["Morphology"] != previous_morphology:
            lines.append(r"\midrule")
        values = [_latex_escape(row[header]) for header in headers]
        if "Independent Final Validation" in row["Intensity"]:
            values[1] = r"1$\times$ (validation)"
            values = [rf"\textbf{{{value}}}" for value in values]
        lines.append(" & ".join(values) + r" \\")
        previous_morphology = row["Morphology"]
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{0.4em}",
            r"\begin{minipage}{0.98\textwidth}\footnotesize",
            (
                r"\textit{Note:} All $\Delta$ columns except Found are percentages. "
                r"Negative radiation-exposure values indicate reductions; negative SAR "
                r"values indicate performance loss; negative Found values indicate fewer "
                r"survivors found. Positive UAV values indicate greater cumulative UAV "
                r"excess dose."
            ),
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_table(summary: pd.DataFrame) -> None:
    rows = _table_rows(summary)
    csv_path = OUTPUT_ROOT / "table_01_final_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_ROOT / "table_01_final_results.md").write_text(
        _markdown_table(rows), encoding="utf-8"
    )
    (OUTPUT_ROOT / "table_01_final_results.tex").write_text(
        _latex_table(rows), encoding="utf-8"
    )


def _stat_line(
    summary: pd.DataFrame,
    morphology: str,
    metric: str,
    multiplier: float = 1.0,
) -> str:
    row = summary[
        summary["morphology"].eq(morphology)
        & summary["intensity"].eq(multiplier)
    ].iloc[0]
    return (
        f"mean {_signed(row[f'{metric}_mean'], suffix='%')}, "
        f"median {_signed(row[f'{metric}_median'], suffix='%')}, "
        f"SD {row[f'{metric}_sd']:.2f}%, range "
        f"[{_signed(row[f'{metric}_min'], suffix='%')}, "
        f"{_signed(row[f'{metric}_max'], suffix='%')}], "
        f"{int(row[f'{metric}_improved'])}/20 improved"
    )


def write_readme(summary: pd.DataFrame) -> None:
    multi_max = _stat_line(summary, "Multi", "survivor_max")
    single_total = _stat_line(summary, "Single", "survivor_total")
    surface_total = _stat_line(summary, "Surface", "survivor_total")
    single_p95 = _stat_line(summary, "Single", "survivor_p95")
    multi_p95 = _stat_line(summary, "Multi", "survivor_p95")
    surface_p95 = _stat_line(summary, "Surface", "survivor_p95")
    readme = f"""# Final Results Summary

## Data basis and checks

The statistical figures and Table 1 use the completed `intensity_summary.csv`: 20 independent unseen seeds at 1× and paired sensitivity comparisons at 0.25×, 0.5×, 2× and 4×. Each seed retains the same radiation geometry across intensity levels. This is 20 independent worlds, not 100 independent worlds. All 100 seed × intensity runs succeeded and all 300 morphology rows were present, with no duplicate keys or missing required metrics.

The analysis independently confirmed the frozen validation parameters: Y = 0.05, sigma = 40 m and influence radius = 120 m. The rounded means and improvement counts reconcile with `intensity_report.md`. That report's abbreviated “UAV mean Δ%” values correspond to the CSV's cumulative UAV **excess-dose** percentage change; Table 1 uses the explicit label “UAV excess dose Δ”.

## Figure 1 — Representative route comparison

`figure_01_route_comparison` shows Original and Radiation-Aware routes for Single-point, Multi-point and Surface radiation in the completed representative experiment. The map extent, survivors, start and radiation truth are held constant within each row. Radiation information visibly changes the executed route in all three morphologies.

This figure supports a qualitative behavioural comparison only. It is not a significance test and the representative run predates the 20-seed 50 m validation configuration.

## Figure 2 — Independent 1× validation distributions

`figure_02_validation_distribution` shows every unseen seed, boxplots and means for survivor Total, P95 and Maximum exposure changes. Negative values indicate improvement; no outlier is removed.

- Single Total: {single_total}.
- Surface Total: {surface_total}.
- P95 improved in 19/20 Single, 17/20 Multi and 19/20 Surface seeds. Single: {single_p95}. Multi: {multi_p95}. Surface: {surface_p95}.
- Multi Maximum has severe tail instability: {multi_max}. Its mean is adverse despite a beneficial median, with a largest observed change of +508.19%.

The figure establishes distributional stability and tail risk. It does not identify the causal path mechanism behind individual outliers.

## Figure 3 — Intensity sensitivity

`figure_03_intensity_sensitivity` plots mean survivor Total and P95 changes with 95% t confidence intervals across the same 20 paired seeds. Single Total and P95 reductions strengthen with intensity. Surface shows a large benefit from 0.25× onward, although its mean is not monotonic at every level. Multi P95 remains beneficial on average at every intensity, while Multi Total remains close to neutral or adverse until 4×.

The paired curves support intensity robustness with strong morphology dependence. They do not represent 100 independent spatial worlds.

## Figure 4 — Improvement consistency

`figure_04_improvement_consistency` reports improved seeds out of 20 for Total, P95 and Maximum exposure. The outlined 1× column is the independent validation. Single reaches 20/20 for Total and P95 at 2× and 4×. Multi has consistently lower Total and Maximum improvement counts, while its P95 counts remain 17–18/20. Surface is usually beneficial but retains a small number of adverse seeds.

Counts describe direction, not the size or practical importance of a change.

## Figure 5 — SAR–radiation trade-off

`figure_05_sar_radiation_tradeoff` places each morphology × intensity mean according to time-discounted SAR change and survivor P95 exposure change. Surface combines the largest radiation benefit with the smallest average time-discounted SAR cost (about 7%). Single provides clear radiation benefit with an approximately 16–17% time-discounted SAR cost. Multi has the largest SAR cost (about 25%) and a smaller mean P95 benefit (about 16–21%).

This is an empirical trade-off summary. It does not assign a preferred operating point or causal weight to either objective.

## Figure 6 — Radiation-mode behaviour

`figure_06_radiation_mode_behaviour` shows mean radiation-mode steps and episode counts. Single and Multi enter one episode lasting all 3,600 recorded radiation-mode steps at every tested intensity, including 0.25×. Surface instead shows repeated local entry and exit: mean steps rise from 410.15 at 0.25× to 773.25 at 4×, while mean episodes rise from 17.50 to 28.65.

The data directly support Point/Multi mode saturation and local Surface engagement. A plausible limitation is that low-intensity Point/Multi cases do not recover Original-like search behaviour. The retained sensitivity output does not include route-level event logs, so it cannot prove that any specific movement mechanism caused the Multi outliers.

## Figure 7 — Representative online radiation estimate

`figure_07_representative_estimated_radiation_map` shows the saved estimated field, executed route and a subset of measurement locations for the representative Single-point experiment. The plot reads the NPZ's frozen legacy metadata: a 1 m-resolution, ground-equivalent excess gamma dose-rate estimate referenced to 1 m and expressed in uSv/h. It is not reinterpreted under the newer 50 m contract.

The figure demonstrates that route measurements populated an online estimate used by the system. It is not a quantitative map-reconstruction validation: no ground-truth comparison, RMSE, MAE, correlation or accuracy score was calculated.

## Table 1 — Final quantitative summary

`table_01_final_results` gives one row for each of the 15 morphology × intensity combinations. The bold 1× rows are the independent final validation. Values are seed-level paired means; distributional detail remains in Figures 2–4.

## Key findings

1. Single and Surface survivor-exposure reductions were stable in the independent 1× validation, with Total/P95 improvements in 18/20 and 19/20 seeds for both morphologies.
2. P95 was the most directionally consistent radiation-benefit metric across morphologies and intensities.
3. Multi retained a typical P95 benefit but showed severe worst-case tail instability. At 1× its Maximum mean was +55.63% while its median was -27.18%.
4. Survivor-risk reduction generally strengthened with intensity, especially for Single, while SAR penalties did not worsen monotonically with intensity. Morphology was the stronger separator of SAR cost.
5. Radiation-Aware search produced a clear SAR–radiation trade-off: lower survivor pre-discovery exposure commonly coincided with lower time-discounted SAR performance and, in some conditions, fewer survivors found.
6. UAV excess dose did not decrease consistently. The supported claim is survivor-risk prioritisation, not universal radiation avoidance or guaranteed worst-case reduction.
"""
    (OUTPUT_ROOT / "README_results.md").write_text(readme, encoding="utf-8")


def main() -> None:
    _configure_style()
    intensity, status, config = _read_inputs()
    _validate_inputs(intensity, status, config)
    summary = _group_summary(intensity)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure_01_route_comparison()
    figure_02_validation_distribution(intensity)
    figure_03_intensity_sensitivity(intensity)
    figure_04_improvement_consistency(intensity)
    figure_05_sar_radiation_tradeoff(summary)
    figure_06_radiation_mode_behaviour(intensity)
    figure_07_representative_estimated_map()
    write_table(summary)
    write_readme(summary)
    print(f"Generated final results summary in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
