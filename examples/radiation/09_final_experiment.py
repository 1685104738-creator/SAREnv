"""Run the frozen MSc dissertation radiation-aware final experiment.

This runner creates one radiation-blind Original Greedy baseline, three
deterministic radiation scenarios, one complete radiation-aware mission per
scenario, unit-aware post-run evaluations, paired survivor comparisons and
the final cross-scenario outputs. It never queries OSM or regenerates SAR data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from shapely.geometry import box

from sarenv import DatasetLoader, load_lost_person_locations
from sarenv.analytics.mission_mode import MissionMode, MissionModeController
from sarenv.analytics.paths import (
    generate_greedy_path,
    greedy_world_to_grid_position,
    native_greedy_step_allowance,
)
from sarenv.analytics.radiation_priority import RadiationGreedyStepPlanner
from sarenv.analytics.route_execution import (
    ExecutedNodeRadiationObserver,
    ExecutedSARState,
    PrecomputedRouteController,
)
from sarenv.evaluation import (
    EvaluationConfig,
    PostRunEvaluator,
    RadiationQuantityContract,
    SurfaceKermaTruthField,
    load_executed_trajectory,
)
from sarenv.radiation import (
    GridSpec,
    PointSource,
    PointSourceConfig,
    save_surface_simulation,
)
from sarenv.radiation.composite.field import CompositeRadiationField
from sarenv.radiation.online.estimator import (
    IncrementalRadiationGrid,
    RADIATION_KERNEL_SIGMA_M,
    RADIATION_UPDATE_RADIUS_M,
)
from sarenv.radiation.online.sensor import NoiseFreeRadiationSensor
from sarenv.radiation.online.trigger import (
    AdaptiveHysteresisRadiationTrigger,
    NOMINAL_HYSTERESIS_MARGIN,
    NOISE_FREE_CONFIRMATION_SAMPLES,
)
from sarenv.radiation.surface import (
    MEDIUM_ACTIVITY_DENSITY_BQ_M2,
    SurfacePhotonResponseConfig,
    SurfacePhotonResponseKernel,
    UniformPolygonConfig,
    UniformPolygonSource,
    simulate_surface_source,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = (
    REPOSITORY_ROOT
    / "examples"
    / "sarenv_dataset"
    / "sarenv_outputs"
    / "radiation_area_01_small_20m"
)
FINAL_ROOT = REPOSITORY_ROOT / "results" / "final_experiment"

PLANNER_SEED = 42
FOV_DEG = 45.0
PLATFORM_ALTITUDE_M = 50.0
VALUE_REFERENCE_HEIGHT_M = 1.0
RADIATION_RESOLUTION_M = 1.0
INITIAL_ENTRY_MULTIPLIER = 0.01
BASE_HAZARD_REFERENCE = 10.0
HYSTERESIS_MARGIN = NOMINAL_HYSTERESIS_MARGIN
REFERENCE_RADIUS_FRACTION = 0.5
EVALUATION_SPEED_M_S = 10.0
EVALUATION_SPEED_STATUS = "provisional"
RADIATION_INTEGRATION_STEP_M = 1.0
POINT_BACKGROUND_USV_H = 0.20
POINT_REFERENCE_EXCESS_USV_H = 20_000.0
SINGLE_POINT_OFFSET_M = (-130.0, -190.0)
MULTI_POINT_OFFSETS_M = (
    (-130.0, -190.0),
    (220.0, 160.0),
    (-270.0, 230.0),
)
SURFACE_HALF_WIDTH_M = 100.0
SURFACE_HALF_HEIGHT_M = 70.0
SURFACE_PATCH_PADDING_M = 200.0
SURFACE_BACKGROUND_UGY_H = 0.0

POINT_QUANTITY = "ground_equivalent_excess_gamma_dose_rate"
POINT_RATE_UNIT = "uSv/h"
POINT_DOSE_UNIT = "uSv"
SURFACE_QUANTITY = "ground_equivalent_excess_collision_air_kerma_rate"
SURFACE_RATE_UNIT = "uGy/h"
SURFACE_DOSE_UNIT = "uGy"
SAMPLING_CONTRACT = "one_noise_free_measurement_per_executed_route_node"


TruthQuery = Callable[[float, float, float], float]


@dataclass(frozen=True)
class FrozenContext:
    item: object
    survivors: object
    probability_map: np.ndarray
    bounds: tuple[float, float, float, float]
    search_center: tuple[float, float]
    max_radius_m: float
    detection_radius_m: float
    mission_step_allowance: int
    heatmap_sha256: str
    survivor_sha256: str


@dataclass(frozen=True)
class FrozenRadiationScenario:
    directory_name: str
    scenario_id: str
    scenario_type: str
    measurement_quantity: str
    contract: RadiationQuantityContract
    background_rate: float
    truth_query_excess: TruthQuery
    truth_query_total: TruthQuery
    source_positions_m: tuple[tuple[float, float], ...]
    config: dict[str, object]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError(f"Cannot write empty final output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def load_frozen_context() -> FrozenContext:
    item = DatasetLoader(str(DATASET_DIRECTORY)).load_environment()
    survivors_path = DATASET_DIRECTORY / "lost_persons.json"
    survivors = load_lost_person_locations(survivors_path)
    probability_map = np.asarray(item.heatmap, dtype=float)
    bounds = tuple(float(value) for value in item.bounds)
    if item.size != "small":
        raise ValueError("FINAL experiment requires authoritative Direct Small.")
    if not math.isclose(float(item.meter_per_bin), 20.0, abs_tol=1e-12):
        raise ValueError("FINAL experiment requires 20 m SAR resolution.")
    if probability_map.shape != (60, 60):
        raise ValueError("FINAL experiment requires the frozen 60x60 SAR grid.")
    if not math.isclose(float(probability_map.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("FINAL Small probability map must sum to one.")
    if survivors.environment_size != item.size:
        raise ValueError("Frozen survivors do not match the dataset size.")
    if tuple(survivors.bounds) != bounds:
        raise ValueError("Frozen survivors do not match the dataset bounds.")
    if len(survivors.points) != 100:
        raise ValueError("FINAL experiment requires the saved 100 survivors.")
    search_center = (
        (bounds[0] + bounds[2]) / 2.0,
        (bounds[1] + bounds[3]) / 2.0,
    )
    detection_radius_m = PLATFORM_ALTITUDE_M * np.tan(
        np.radians(FOV_DEG / 2.0)
    )
    return FrozenContext(
        item=item,
        survivors=survivors,
        probability_map=probability_map,
        bounds=bounds,
        search_center=search_center,
        max_radius_m=float(item.radius_km) * 1_000.0,
        detection_radius_m=float(detection_radius_m),
        mission_step_allowance=native_greedy_step_allowance(probability_map.shape),
        heatmap_sha256=_sha256(DATASET_DIRECTORY / "heatmap.npy"),
        survivor_sha256=_sha256(survivors_path),
    )


def snapshot_code_state(context: FrozenContext) -> None:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    code_state = FINAL_ROOT / "code_state"
    code_state.mkdir(parents=True, exist_ok=True)
    branch = _git("branch", "--show-current").strip()
    head = _git("rev-parse", "HEAD").strip()
    status = _git("status", "--short").splitlines()
    diff_stat = _git("diff", "--stat")
    diff_check = _git("diff", "--check", check=False)
    (code_state / "final_experiment_code.patch").write_text(
        _git("diff", "--binary"),
        encoding="utf-8",
    )
    (code_state / "git_status.txt").write_text(
        "\n".join(status) + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(
        code_state / "reproducibility_python_sources.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for root in (
            REPOSITORY_ROOT / "sarenv",
            REPOSITORY_ROOT / "examples" / "radiation",
            REPOSITORY_ROOT / "tests",
        ):
            for path in root.rglob("*.py"):
                archive.write(path, path.relative_to(REPOSITORY_ROOT))
    manifest = {
        "experiment_name": "MSc dissertation FINAL experiment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "head": head,
        "git_status_short": status,
        "git_diff_stat": diff_stat.splitlines(),
        "git_diff_check_output": diff_check.splitlines(),
        "dataset_directory": str(DATASET_DIRECTORY),
        "dataset_files_sha256": {
            name: _sha256(DATASET_DIRECTORY / name)
            for name in ("heatmap.npy", "features.geojson", "metadata.json")
        },
        "survivors_sha256": context.survivor_sha256,
        "planner_seed": PLANNER_SEED,
        "frozen_parameters": _common_parameters(context),
        "code_snapshot": {
            "tracked_diff": "code_state/final_experiment_code.patch",
            "python_source_archive": (
                "code_state/reproducibility_python_sources.zip"
            ),
        },
    }
    _json(FINAL_ROOT / "experiment_manifest.json", manifest)


def _common_parameters(context: FrozenContext) -> dict[str, object]:
    return {
        "dataset_size": "small",
        "dataset_bounds_projected": list(context.bounds),
        "dataset_crs": str(context.item.projected_crs),
        "sar_resolution_m": float(context.item.meter_per_bin),
        "sar_shape": list(context.probability_map.shape),
        "sar_probability_sum": float(context.probability_map.sum()),
        "radiation_resolution_m": RADIATION_RESOLUTION_M,
        "fov_deg": FOV_DEG,
        "camera_detection_radius_m": context.detection_radius_m,
        "platform_altitude_m": PLATFORM_ALTITUDE_M,
        "value_reference_height_m": VALUE_REFERENCE_HEIGHT_M,
        "altitude_correction_assumption": "perfect_ground_equivalent_scalar",
        "planner_seed": PLANNER_SEED,
        "base_hazard_reference": BASE_HAZARD_REFERENCE,
        "initial_entry_multiplier": INITIAL_ENTRY_MULTIPLIER,
        "initial_entry_threshold": (
            BASE_HAZARD_REFERENCE * INITIAL_ENTRY_MULTIPLIER
        ),
        "hysteresis_margin": HYSTERESIS_MARGIN,
        "confirmation_samples": NOISE_FREE_CONFIRMATION_SAMPLES,
        "gaussian_sigma_m": RADIATION_KERNEL_SIGMA_M,
        "gaussian_update_radius_m": RADIATION_UPDATE_RADIUS_M,
        "candidate_radiation_aggregation": (
            "strict_max_over_overlapping_radiation_grid_cells"
        ),
        "mission_completion_rule": (
            "all_searchable_cells_covered_or_native_greedy_move_ceiling"
        ),
        "native_mission_move_allowance": context.mission_step_allowance,
        "distance_budget_m": None,
        "evaluation_speed_m_s": EVALUATION_SPEED_M_S,
        "evaluation_speed_status": EVALUATION_SPEED_STATUS,
        "radiation_integration_step_m": RADIATION_INTEGRATION_STEP_M,
        "heatmap_sha256": context.heatmap_sha256,
        "survivors_sha256": context.survivor_sha256,
    }


def _plot_route(
    path: Path,
    context: FrozenContext,
    records: list[dict[str, object]],
    *,
    title: str,
    source_positions: tuple[tuple[float, float], ...] = (),
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(
        context.probability_map,
        origin="lower",
        extent=(
            context.bounds[0],
            context.bounds[2],
            context.bounds[1],
            context.bounds[3],
        ),
        cmap="YlOrRd",
        interpolation="nearest",
    )
    figure.colorbar(image, ax=axis, label="SAR probability per cell")
    for mode, colour in (("NORMAL", "limegreen"), ("RADIATION", "deepskyblue")):
        labelled = False
        for previous, current in zip(records, records[1:]):
            if str(current["mode_before"]) != mode:
                continue
            axis.plot(
                [previous["x_m"], current["x_m"]],
                [previous["y_m"], current["y_m"]],
                color=colour,
                linewidth=0.9,
                alpha=0.85,
                label=mode if not labelled else None,
            )
            labelled = True
    axis.scatter(
        [point.x for point in context.survivors.points],
        [point.y for point in context.survivors.points],
        c="purple",
        s=13,
        alpha=0.55,
        label="Frozen survivors",
    )
    axis.scatter(
        [records[0]["x_m"]],
        [records[0]["y_m"]],
        c="white",
        edgecolors="black",
        s=70,
        label="Start",
        zorder=7,
    )
    if source_positions:
        axis.scatter(
            [position[0] for position in source_positions],
            [position[1] for position in source_positions],
            marker="*",
            c="yellow",
            edgecolors="black",
            s=145,
            label="Source truth (evaluation only)",
            zorder=8,
        )
    axis.set_title(title)
    axis.set_xlim(context.bounds[0], context.bounds[2])
    axis.set_ylim(context.bounds[1], context.bounds[3])
    axis.set_aspect("equal")
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.legend(loc="upper right", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_original_baseline(context: FrozenContext) -> Path:
    simulation_directory = FINAL_ROOT / "baseline_original" / "simulation"
    simulation_directory.mkdir(parents=True, exist_ok=True)
    route = generate_greedy_path(
        center_x=context.search_center[0],
        center_y=context.search_center[1],
        num_drones=1,
        probability_map=context.probability_map,
        bounds=context.bounds,
        max_radius=context.max_radius_m,
        fov_deg=FOV_DEG,
        altitude=PLATFORM_ALTITUDE_M,
        max_steps=context.mission_step_allowance,
        rng=np.random.default_rng(PLANNER_SEED),
    )[0]
    coordinates = list(route.coords)
    if len(coordinates) < 2:
        raise RuntimeError("Original Greedy produced an empty FINAL route.")
    records: list[dict[str, object]] = []
    cumulative = 0.0
    previous: tuple[float, float] | None = None
    for index, coordinate in enumerate(coordinates):
        point = (float(coordinate[0]), float(coordinate[1]))
        segment = 0.0 if previous is None else math.dist(previous, point)
        cumulative += segment
        row, col = greedy_world_to_grid_position(
            *point,
            map_shape=context.probability_map.shape,
            bounds=context.bounds,
        )
        records.append(
            {
                "step_index": index,
                "mode_before": MissionMode.NORMAL.value,
                "mode_after": MissionMode.NORMAL.value,
                "x_m": point[0],
                "y_m": point[1],
                "sar_row": row,
                "sar_col": col,
                "step_distance_m": segment,
                "total_distance_m": cumulative,
            }
        )
        previous = point
    trajectory_path = simulation_directory / "mission_steps.csv"
    _csv(trajectory_path, records)
    resolved = {
        **_common_parameters(context),
        "scenario_id": "radiation_blind_original_greedy_baseline",
        "planner": "Original Greedy",
        "radiation_truth_access": False,
        "sensor_used": False,
        "estimator_used": False,
        "trigger_used": False,
        "executed_nodes": len(records),
        "executed_moves": len(records) - 1,
        "route_distance_m": cumulative,
        "completion_reason": (
            "native_original_route_generation_complete_or_move_ceiling"
        ),
    }
    resolved_path = simulation_directory / "resolved_parameters.json"
    _json(resolved_path, resolved)
    _plot_route(
        simulation_directory / "trajectory.png",
        context,
        records,
        title="Frozen radiation-blind Original Greedy baseline",
    )
    summary = {
        "status": "COMPLETE",
        "planner": "Original Greedy",
        "total_executed_nodes": len(records),
        "total_executed_moves": len(records) - 1,
        "total_route_distance_m": cumulative,
        "route_sha256": _sha256(trajectory_path),
        "heatmap_sha256": context.heatmap_sha256,
        "survivors_sha256": context.survivor_sha256,
        "radiation_blind": True,
        "outputs": {
            "mission_steps_csv": str(trajectory_path),
            "resolved_parameters_json": str(resolved_path),
            "trajectory_png": str(simulation_directory / "trajectory.png"),
        },
    }
    _json(simulation_directory / "summary.json", summary)
    return trajectory_path


def _point_source(
    context: FrozenContext,
    *,
    source_id: str,
    offset_m: tuple[float, float],
) -> PointSource:
    return PointSource(
        PointSourceConfig(
            source_id=source_id,
            x_m=context.search_center[0] + offset_m[0],
            y_m=context.search_center[1] + offset_m[1],
            reference_excess_uSv_h=POINT_REFERENCE_EXCESS_USV_H,
            crs=str(context.item.projected_crs),
        )
    )


def freeze_point_scenario(
    context: FrozenContext,
    *,
    multi: bool,
) -> FrozenRadiationScenario:
    offsets = MULTI_POINT_OFFSETS_M if multi else (SINGLE_POINT_OFFSET_M,)
    sources = tuple(
        _point_source(
            context,
            source_id=(
                f"final_multi_point_{index + 1}"
                if multi
                else "single_point_formal_first_run"
            ),
            offset_m=offset,
        )
        for index, offset in enumerate(offsets)
    )
    directory_name = "02_multi_point" if multi else "01_single_point"
    scenario_id = (
        "final_three_point_stress_scenario"
        if multi
        else "single_point_formal_first_run"
    )
    field = CompositeRadiationField(
        crs=str(context.item.projected_crs),
        background_uSv_h=POINT_BACKGROUND_USV_H,
        point_sources=sources,
    )
    config = {
        **_common_parameters(context),
        "scenario_id": scenario_id,
        "scenario_type": "multi_point" if multi else "point_only",
        "measurement_quantity": POINT_QUANTITY,
        "radiation_quantity": "synthetic_gamma_dose_rate",
        "radiation_rate_unit": POINT_RATE_UNIT,
        "radiation_dose_unit": POINT_DOSE_UNIT,
        "background_rate": POINT_BACKGROUND_USV_H,
        "background_unit": POINT_RATE_UNIT,
        "sources": [source.to_metadata() for source in sources],
        "source_configuration_rule": (
            "three deterministic centre-relative offsets fixed before planning; "
            "independent of survivors and routes"
            if multi
            else "reused validated single_point_formal_first_run position"
        ),
        "per_source_reference_excess": POINT_REFERENCE_EXCESS_USV_H,
        "per_source_reference_unit": POINT_RATE_UNIT,
        "total_source_strength_relation_to_single": (
            "three sources at the same per-source strength; higher total burden"
            if multi
            else "canonical single source"
        ),
    }
    scenario_path = FINAL_ROOT / directory_name / "scenario_config.json"
    _json(scenario_path, config)
    return FrozenRadiationScenario(
        directory_name=directory_name,
        scenario_id=scenario_id,
        scenario_type=str(config["scenario_type"]),
        measurement_quantity=POINT_QUANTITY,
        contract=RadiationQuantityContract(
            quantity="synthetic_gamma_dose_rate",
            rate_unit=POINT_RATE_UNIT,
            dose_unit=POINT_DOSE_UNIT,
        ),
        background_rate=POINT_BACKGROUND_USV_H,
        truth_query_excess=field.query_excess_dose_rate,
        truth_query_total=field.query_total_dose_rate,
        source_positions_m=tuple(
            (source.config.x_m, source.config.y_m) for source in sources
        ),
        config=config,
    )


def freeze_surface_scenario(context: FrozenContext) -> FrozenRadiationScenario:
    center_x, center_y = context.search_center
    geometry = box(
        center_x - SURFACE_HALF_WIDTH_M,
        center_y - SURFACE_HALF_HEIGHT_M,
        center_x + SURFACE_HALF_WIDTH_M,
        center_y + SURFACE_HALF_HEIGHT_M,
    )
    source = UniformPolygonSource(
        UniformPolygonConfig(
            source_id="final_uniform_cs137_surface",
            geometry=geometry,
            activity_density_bq_m2=MEDIUM_ACTIVITY_DENSITY_BQ_M2,
            crs=str(context.item.projected_crs),
        )
    )
    grid = GridSpec.from_bounds(
        source.bounds,
        source.crs,
        padding_m=SURFACE_PATCH_PADDING_M,
    )
    kernel = SurfacePhotonResponseKernel.create(
        SurfacePhotonResponseConfig(),
        grid,
    )
    result = simulate_surface_source(source, kernel, nominal_grid=grid)
    truth = SurfaceKermaTruthField.from_simulation(
        result,
        platform_altitude_m=PLATFORM_ALTITUDE_M,
    )
    scenario_directory = FINAL_ROOT / "03_uniform_surface"
    truth_directory = scenario_directory / "truth"
    save_surface_simulation(result, truth_directory / "ground_1m")
    np.save(
        truth_directory / "platform_collision_air_kerma_rate_50m.npy",
        truth.platform_patch.collision_air_kerma_rate_uGy_h,
    )
    _json(
        truth_directory / "platform_collision_air_kerma_rate_50m_meta.json",
        truth.platform_patch.metadata,
    )
    config = {
        **_common_parameters(context),
        "scenario_id": "final_uniform_cs137_surface",
        "scenario_type": "surface_only",
        "measurement_quantity": SURFACE_QUANTITY,
        "radiation_quantity": truth.quantity,
        "radiation_rate_unit": truth.unit,
        "radiation_dose_unit": SURFACE_DOSE_UNIT,
        "background_rate": SURFACE_BACKGROUND_UGY_H,
        "background_unit": SURFACE_RATE_UNIT,
        "radionuclide": "Cs-137",
        "surface_type": "uniform_polygon",
        "activity_density_bq_m2": MEDIUM_ACTIVITY_DENSITY_BQ_M2,
        "polygon_bounds": list(geometry.bounds),
        "polygon_exterior_coordinates": [
            [float(x), float(y)] for x, y in geometry.exterior.coords
        ],
        "patch_bounds": list(grid.bounds),
        "patch_shape": list(grid.shape),
        "patch_padding_m": SURFACE_PATCH_PADDING_M,
        "ground_truth_plane_height_m": VALUE_REFERENCE_HEIGHT_M,
        "platform_evaluation_plane_height_m": PLATFORM_ALTITUDE_M,
        "platform_plane_method": (
            "same production activity raster, Cs-137 gamma yield, inverse-square "
            "air-attenuated kernel and fluence-to-air-kerma conversion; only "
            "observation height changed in R"
        ),
        "uGy_to_uSv_conversion_applied": False,
        "surface_hazard_reference_native_value": BASE_HAZARD_REFERENCE,
        "surface_hazard_reference_native_unit": SURFACE_RATE_UNIT,
        "minimum_raw_ground_fft_before_clipping": (
            result.minimum_raw_fft_value_before_clipping
        ),
        "minimum_raw_platform_fft_before_clipping": (
            truth.minimum_raw_platform_fft_value
        ),
        "ground_max_excess_rate": float(
            result.collision_air_kerma_rate_patch.collision_air_kerma_rate_uGy_h.max()
        ),
        "platform_max_excess_rate": float(
            truth.platform_patch.collision_air_kerma_rate_uGy_h.max()
        ),
        "configuration_rule": (
            "deterministic rectangle centred on the frozen SAR environment; "
            "fixed before planning and independent of survivors/routes"
        ),
    }
    _json(scenario_directory / "scenario_config.json", config)

    def total_query(x_m: float, y_m: float, z_m: float) -> float:
        return SURFACE_BACKGROUND_UGY_H + truth.query_excess(x_m, y_m, z_m)

    return FrozenRadiationScenario(
        directory_name="03_uniform_surface",
        scenario_id=str(config["scenario_id"]),
        scenario_type="surface_only",
        measurement_quantity=SURFACE_QUANTITY,
        contract=RadiationQuantityContract(
            quantity=truth.quantity,
            rate_unit=SURFACE_RATE_UNIT,
            dose_unit=SURFACE_DOSE_UNIT,
        ),
        background_rate=SURFACE_BACKGROUND_UGY_H,
        truth_query_excess=truth.query_excess,
        truth_query_total=total_query,
        source_positions_m=((center_x, center_y),),
        config=config,
    )


def run_radiation_aware_mission(
    context: FrozenContext,
    scenario: FrozenRadiationScenario,
) -> Path:
    started = time.perf_counter()
    output_directory = FINAL_ROOT / scenario.directory_name / "radiation_aware" / "simulation"
    output_directory.mkdir(parents=True, exist_ok=True)
    planner_rng = np.random.default_rng(PLANNER_SEED)
    initial_route = generate_greedy_path(
        center_x=context.search_center[0],
        center_y=context.search_center[1],
        num_drones=1,
        probability_map=context.probability_map,
        bounds=context.bounds,
        max_radius=context.max_radius_m,
        fov_deg=FOV_DEG,
        altitude=PLATFORM_ALTITUDE_M,
        max_steps=context.mission_step_allowance,
        rng=planner_rng,
    )[0]
    state = ExecutedSARState(
        map_shape=context.probability_map.shape,
        bounds=context.bounds,
        detection_radius_m=context.detection_radius_m,
    )
    normal_controller = PrecomputedRouteController(initial_route, state)
    sensor = NoiseFreeRadiationSensor(
        scenario.truth_query_total,
        quantity=scenario.measurement_quantity,
        unit=scenario.contract.rate_unit,
        value_reference_height_m=VALUE_REFERENCE_HEIGHT_M,
        background_value_to_subtract=scenario.background_rate,
    )
    radiation_grid = GridSpec.from_bounds(
        context.bounds,
        str(context.item.projected_crs),
    )
    estimator = IncrementalRadiationGrid(
        radiation_grid,
        quantity=scenario.measurement_quantity,
        unit=scenario.contract.rate_unit,
        update_radius_m=RADIATION_UPDATE_RADIUS_M,
        kernel_sigma_m=RADIATION_KERNEL_SIGMA_M,
        value_reference_height_m=VALUE_REFERENCE_HEIGHT_M,
    )
    trigger = AdaptiveHysteresisRadiationTrigger(
        base_hazard_reference=BASE_HAZARD_REFERENCE,
        initial_entry_multiplier=INITIAL_ENTRY_MULTIPLIER,
        quantity=scenario.measurement_quantity,
        unit=scenario.contract.rate_unit,
        hysteresis_margin=HYSTERESIS_MARGIN,
        confirmation_samples=NOISE_FREE_CONFIRMATION_SAMPLES,
    )
    observer = ExecutedNodeRadiationObserver(sensor, estimator, trigger)
    radiation_planner = RadiationGreedyStepPlanner(
        probability_map=context.probability_map,
        bounds=context.bounds,
        search_center=context.search_center,
        max_radius=context.max_radius_m,
        detection_radius_m=context.detection_radius_m,
        estimator=estimator,
        hazard_reference_excess=BASE_HAZARD_REFERENCE,
        reference_radius_fraction=REFERENCE_RADIUS_FRACTION,
        rng=planner_rng,
    )
    controller = MissionModeController(
        state=state,
        normal_route_controller=normal_controller,
        observer=observer,
        radiation_planner=radiation_planner,
        initial_budget_m=None,
        mission_step_allowance=context.mission_step_allowance,
        planner_rng=planner_rng,
        probability_map=context.probability_map,
        bounds=context.bounds,
        search_center=context.search_center,
        max_radius=context.max_radius_m,
        fov_deg=FOV_DEG,
        altitude_m=PLATFORM_ALTITUDE_M,
    )

    steps: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    measurements: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    guard = context.mission_step_allowance * 2 + 10
    while not controller.is_complete:
        if len(events) >= guard:
            raise RuntimeError("FINAL mission exceeded its derived call guard.")
        result = controller.advance(simulated_time_s=0.0)
        execution = result.route_execution
        observation = result.radiation_observation
        events.append(
            {
                "controller_call_index": result.controller_call_index,
                "executed_step_index": result.executed_step_index,
                "mode_before": result.mode_before.value,
                "mode_after": result.mode_after.value,
                "moved": result.moved,
                "x_m": execution.position_m[0] if execution else None,
                "y_m": execution.position_m[1] if execution else None,
                "event_type": (
                    observation.hazard_transition.upper()
                    if observation and observation.hazard_transition
                    else ("COMPLETE" if result.mission_complete else None)
                ),
                "episode_index": (
                    observation.hazard_episode_index if observation else None
                ),
                "entry_multiplier_initial": INITIAL_ENTRY_MULTIPLIER,
                "entry_threshold_current": (
                    observation.entry_threshold_current
                    if observation
                    else trigger.current_entry_threshold
                ),
                "exit_threshold_current": (
                    observation.exit_threshold_current
                    if observation
                    else trigger.current_exit_threshold
                ),
                "base_hazard_reference": BASE_HAZARD_REFERENCE,
                "trigger_measurement": (
                    observation.measurement.value if observation else None
                ),
                "radiation_quantity": scenario.measurement_quantity,
                "radiation_unit": scenario.contract.rate_unit,
                "transition_reason": result.mode_transition_reason,
                "normal_route_interrupted": result.normal_route_interrupted,
                "normal_route_replanned": result.normal_route_replanned,
                "mission_complete": result.mission_complete,
                "completion_reason": result.completion_reason,
            }
        )
        if result.radiation_plan is not None:
            plan = result.radiation_plan
            for assessment in plan.decision.assessments:
                estimate = assessment.radiation_estimate
                candidates.append(
                    {
                        "mission_step": result.executed_step_index,
                        "controller_call_index": result.controller_call_index,
                        "mode": result.mode_before.value,
                        "candidate_row": assessment.candidate.row,
                        "candidate_col": assessment.candidate.col,
                        "candidate_x_m": assessment.candidate.x_m,
                        "candidate_y_m": assessment.candidate.y_m,
                        "sar_score": assessment.candidate.sar_score,
                        "sar_reference_score": radiation_planner.sar_reference.score,
                        "sar_relevant": assessment.sar_relevant,
                        "sar_cell_radiation_max_excess": estimate.value,
                        "radiation_quantity": estimate.quantity,
                        "radiation_unit": estimate.unit,
                        "radiation_equivalent_priority": (
                            assessment.radiation_equivalent_priority
                        ),
                        "max_weight_sum": estimate.max_weight_sum,
                        "supported_radiation_cells": estimate.supported_cell_count,
                        "total_overlapping_radiation_cells": estimate.total_cell_count,
                        "support_fraction": estimate.support_fraction,
                        "eligible": assessment.sar_relevant,
                        "chosen": assessment.chosen,
                        "selection_reason": assessment.selection_reason,
                        "planner_decision_reason": plan.decision.reason,
                    }
                )
        if execution is not None and observation is not None:
            measurement = observation.measurement
            step = {
                "step_index": result.executed_step_index,
                "mode_before": result.mode_before.value,
                "mode_after": result.mode_after.value,
                "x_m": execution.position_m[0],
                "y_m": execution.position_m[1],
                "sar_row": execution.grid_position[0],
                "sar_col": execution.grid_position[1],
                "step_distance_m": execution.distance_from_previous_m,
                "total_distance_m": state.total_executed_distance_m,
                "measurement_excess": measurement.value,
                "measurement_quantity": measurement.quantity,
                "measurement_unit": measurement.unit,
                "platform_altitude_m": measurement.platform_altitude_m,
                "value_reference_height_m": measurement.value_reference_height_m,
                "estimator_updated_cells": observation.updated_radiation_cells,
                "estimator_update_time_s": observation.estimator_update_time_s,
                "hazard_active": observation.anomaly_confirmed,
                "hazard_transition": observation.hazard_transition,
                "hazard_episode_index": observation.hazard_episode_index,
                "entry_threshold_current": observation.entry_threshold_current,
                "exit_threshold_current": observation.exit_threshold_current,
                "normal_route_interrupted": result.normal_route_interrupted,
                "mode_transition_reason": result.mode_transition_reason,
                "radiation_planning_time_s": (
                    result.radiation_plan.planning_time_s
                    if result.radiation_plan is not None
                    else None
                ),
                "continuation_planning_time_s": result.continuation_planning_time_s,
            }
            steps.append(step)
            measurements.append(
                {
                    "step_index": step["step_index"],
                    "x_m": step["x_m"],
                    "y_m": step["y_m"],
                    "platform_altitude_m": step["platform_altitude_m"],
                    "value_reference_height_m": step["value_reference_height_m"],
                    "value": step["measurement_excess"],
                    "quantity": step["measurement_quantity"],
                    "unit": step["measurement_unit"],
                    "sampling_contract": SAMPLING_CONTRACT,
                }
            )

    if not steps:
        raise RuntimeError("FINAL radiation-aware mission executed no nodes.")
    if not candidates:
        candidates.append(
            {
                "mission_step": None,
                "controller_call_index": None,
                "mode": None,
                "candidate_row": None,
                "candidate_col": None,
                "candidate_x_m": None,
                "candidate_y_m": None,
                "sar_score": None,
                "sar_reference_score": radiation_planner.sar_reference.score,
                "sar_relevant": None,
                "sar_cell_radiation_max_excess": None,
                "radiation_quantity": scenario.measurement_quantity,
                "radiation_unit": scenario.contract.rate_unit,
                "radiation_equivalent_priority": None,
                "max_weight_sum": None,
                "supported_radiation_cells": None,
                "total_overlapping_radiation_cells": None,
                "support_fraction": None,
                "eligible": None,
                "chosen": None,
                "selection_reason": "no_radiation_mode_decisions",
                "planner_decision_reason": "radiation_mode_not_entered",
            }
        )

    mission_steps_path = output_directory / "mission_steps.csv"
    _csv(mission_steps_path, steps)
    _csv(output_directory / "mode_events.csv", events)
    _csv(output_directory / "radiation_measurements.csv", measurements)
    _csv(output_directory / "candidate_decisions.csv", candidates)
    np.savez_compressed(
        output_directory / "estimated_radiation_grid.npz",
        mean=np.asarray(estimator.mean),
        weight_sum=np.asarray(estimator.weight_sum),
        observed_mask=np.asarray(estimator.observed_mask),
        bounds=np.asarray(estimator.grid.bounds),
        resolution_m=estimator.grid.resolution_m,
        value_reference_height_m=VALUE_REFERENCE_HEIGHT_M,
        quantity=scenario.measurement_quantity,
        unit=scenario.contract.rate_unit,
    )
    _plot_route(
        output_directory / "trajectory.png",
        context,
        steps,
        title=f"FINAL Radiation-aware: {scenario.scenario_id}",
        source_positions=scenario.source_positions_m,
    )

    enter_events = [
        event for event in events if event["transition_reason"] == "hazard_enter_threshold"
    ]
    exit_events = [
        event for event in events if event["transition_reason"] == "hazard_exit_threshold"
    ]
    chosen = [record for record in candidates if record["chosen"] is True]
    resolved = {
        **scenario.config,
        "planner": "Radiation-aware Greedy",
        "measurement_quantity": scenario.measurement_quantity,
        "measurement_unit": scenario.contract.rate_unit,
        "estimated_quantity_contract": "excess_above_background_numeric_zero_prior",
        "hazard_reference_native_value": BASE_HAZARD_REFERENCE,
        "hazard_reference_native_unit": scenario.contract.rate_unit,
        "entry_increment_native_value": trigger.initial_entry_threshold,
        "initial_entry_threshold_native_value": trigger.initial_entry_threshold,
        "final_entry_threshold_native_value": trigger.current_entry_threshold,
        "final_exit_threshold_native_value": trigger.current_exit_threshold,
        "derived_sar_reference_score": radiation_planner.sar_reference.score,
        "sar_reference_derivation": {
            "method": radiation_planner.sar_reference.method,
            "reference_radius_fraction": radiation_planner.sar_reference.radius_fraction,
            "reference_radius_m": radiation_planner.sar_reference.radius_m,
            "ring_cell_count": radiation_planner.sar_reference.ring_cell_count,
        },
        "sampling_contract": SAMPLING_CONTRACT,
        "planner_truth_access": False,
        "sensor_truth_access": True,
        "old_debug_crop_used": False,
        "conditional_probability_renormalisation_used": False,
    }
    resolved_path = output_directory / "resolved_parameters.json"
    _json(resolved_path, resolved)
    summary = {
        "status": "COMPLETE" if controller.is_complete else "INCOMPLETE",
        "scenario_id": scenario.scenario_id,
        "completion_reason": controller.completion_reason,
        "total_executed_nodes": state.executed_node_count,
        "total_executed_moves": state.executed_move_count,
        "total_route_distance_m": state.total_executed_distance_m,
        "normal_steps": sum(step["mode_before"] == "NORMAL" for step in steps),
        "radiation_steps": sum(step["mode_before"] == "RADIATION" for step in steps),
        "radiation_episodes": len(enter_events),
        "enter_events": len(enter_events),
        "exit_events": len(exit_events),
        "radiation_priority_decisions": sum(
            record["planner_decision_reason"] == "radiation_priority"
            for record in chosen
        ),
        "original_greedy_fallback_decisions": sum(
            str(record["planner_decision_reason"]).endswith("_original_greedy_step")
            for record in chosen
        ),
        "mode_sequence": [mode.value for mode in controller.mode_history],
        "entry_threshold_progression": [
            event["entry_threshold_current"] for event in enter_events
        ],
        "final_entry_threshold": trigger.current_entry_threshold,
        "final_exit_threshold": trigger.current_exit_threshold,
        "threshold_unit": scenario.contract.rate_unit,
        "estimator_runtime_s": sum(
            float(step["estimator_update_time_s"]) for step in steps
        ),
        "radiation_planner_runtime_s": sum(
            float(step["radiation_planning_time_s"] or 0.0) for step in steps
        ),
        "continuation_planner_runtime_s": sum(
            float(step["continuation_planning_time_s"] or 0.0) for step in steps
        ),
        "wall_runtime_s": time.perf_counter() - started,
        "route_sha256": _sha256(mission_steps_path),
        "heatmap_sha256": context.heatmap_sha256,
        "survivors_sha256": context.survivor_sha256,
    }
    _json(output_directory / "summary.json", summary)
    return mission_steps_path


def evaluate_route(
    context: FrozenContext,
    scenario: FrozenRadiationScenario,
    *,
    planner_name: str,
    trajectory_path: Path,
) -> None:
    output_directory = (
        FINAL_ROOT / scenario.directory_name / planner_name / "evaluation"
    )
    evaluator = PostRunEvaluator(
        EvaluationConfig(
            trajectory_path=trajectory_path,
            dataset_directory=DATASET_DIRECTORY,
            survivors_path=DATASET_DIRECTORY / "lost_persons.json",
            scenario_parameters_path=(
                FINAL_ROOT / scenario.directory_name / "scenario_config.json"
            ),
            output_directory=output_directory,
            platform_altitude_m=PLATFORM_ALTITUDE_M,
            survivor_reference_height_m=VALUE_REFERENCE_HEIGHT_M,
            fov_deg=FOV_DEG,
            background_rate=scenario.background_rate,
            radiation_quantity=scenario.contract.quantity,
            radiation_rate_unit=scenario.contract.rate_unit,
            radiation_dose_unit=scenario.contract.dose_unit,
            radiation_integration_step_m=RADIATION_INTEGRATION_STEP_M,
            evaluation_speed_m_s=EVALUATION_SPEED_M_S,
            speed_status=EVALUATION_SPEED_STATUS,
        ),
        excess_truth_query=scenario.truth_query_excess,
        source_positions_m=scenario.source_positions_m,
    )
    evaluator.evaluate()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _metric_row(
    name: str,
    original: float,
    radiation_aware: float,
    *,
    unit: str,
) -> dict[str, object]:
    delta = radiation_aware - original
    relative = None if original == 0.0 else 100.0 * delta / original
    return {
        "metric": name,
        "unit": unit,
        "original": original,
        "radiation_aware": radiation_aware,
        "absolute_delta_radiation_aware_minus_original": delta,
        "relative_change_percent": relative,
    }


def _nested(payload: dict[str, object], *keys: str) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, dict):
            raise KeyError(keys)
        value = value[key]
    return value


def _comparison_metrics(
    original: dict[str, object],
    radiation_aware: dict[str, object],
    contract: RadiationQuantityContract,
) -> list[dict[str, object]]:
    definitions = (
        ("sar_likelihood_score", ("sar", "total_likelihood_score"), "probability"),
        ("sar_time_discounted_score", ("sar", "total_time_discounted_score"), "score"),
        ("survivors_found", ("survivor_radiation", "found_count"), "count"),
        ("covered_area", ("sar", "area_covered_km2"), "km^2"),
        ("route_distance", ("trajectory", "total_distance_m"), "m"),
        (
            "uav_excess_exposure_distance_integral",
            ("uav_radiation", "uav_excess_exposure_distance_integral"),
            contract.distance_integral_unit,
        ),
        (
            "uav_total_exposure_distance_integral",
            ("uav_radiation", "uav_total_exposure_distance_integral"),
            contract.distance_integral_unit,
        ),
        (
            "uav_cumulative_excess_dose",
            ("uav_radiation", "uav_cumulative_excess_dose"),
            contract.dose_unit,
        ),
        (
            "uav_cumulative_total_dose",
            ("uav_radiation", "uav_cumulative_total_dose"),
            contract.dose_unit,
        ),
        (
            "survivor_total_pre_discovery_dose",
            (
                "survivor_radiation",
                "all_survivors",
                "pre_discovery_total_dose",
                "sum",
            ),
            contract.dose_unit,
        ),
        (
            "survivor_mean_pre_discovery_dose",
            (
                "survivor_radiation",
                "all_survivors",
                "pre_discovery_total_dose",
                "mean",
            ),
            contract.dose_unit,
        ),
        (
            "survivor_median_pre_discovery_dose",
            (
                "survivor_radiation",
                "all_survivors",
                "pre_discovery_total_dose",
                "median",
            ),
            contract.dose_unit,
        ),
        (
            "survivor_p95_pre_discovery_dose",
            (
                "survivor_radiation",
                "all_survivors",
                "pre_discovery_total_dose",
                "p95",
            ),
            contract.dose_unit,
        ),
        (
            "survivor_max_pre_discovery_dose",
            (
                "survivor_radiation",
                "all_survivors",
                "pre_discovery_total_dose",
                "max",
            ),
            contract.dose_unit,
        ),
        (
            "heading_changes",
            ("trajectory", "turn_diagnostics", "number_of_heading_changes"),
            "count",
        ),
        (
            "total_absolute_heading_change",
            ("trajectory", "turn_diagnostics", "total_absolute_heading_change_deg"),
            "deg",
        ),
    )
    rows = []
    for name, keys, unit in definitions:
        rows.append(
            _metric_row(
                name,
                float(_nested(original, *keys)),
                float(_nested(radiation_aware, *keys)),
                unit=unit,
            )
        )
    return rows


def _pairwise_survivors(
    scenario: FrozenRadiationScenario,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = FINAL_ROOT / scenario.directory_name
    original = _read_csv(root / "original" / "evaluation" / "survivor_radiation.csv")
    aware = _read_csv(root / "radiation_aware" / "evaluation" / "survivor_radiation.csv")
    original_by_id = {int(row["survivor_id"]): row for row in original}
    aware_by_id = {int(row["survivor_id"]): row for row in aware}
    if original_by_id.keys() != aware_by_id.keys():
        raise ValueError("Paired survivor IDs differ between planners.")
    rows = []
    lower = higher = unchanged = 0
    original_only = aware_only = both_unfound = 0
    for survivor_id in sorted(original_by_id):
        left = original_by_id[survivor_id]
        right = aware_by_id[survivor_id]
        original_dose = float(left["pre_discovery_total_dose"])
        aware_dose = float(right["pre_discovery_total_dose"])
        reduction = original_dose - aware_dose
        tolerance = max(1e-12, 1e-6 * max(abs(original_dose), abs(aware_dose)))
        if reduction > tolerance:
            lower += 1
        elif reduction < -tolerance:
            higher += 1
        else:
            unchanged += 1
        original_found = left["found"].lower() == "true"
        aware_found = right["found"].lower() == "true"
        if not original_found and aware_found:
            aware_only += 1
        elif original_found and not aware_found:
            original_only += 1
        elif not original_found and not aware_found:
            both_unfound += 1
        rows.append(
            {
                "survivor_id": survivor_id,
                "x_m": float(left["x_m"]),
                "y_m": float(left["y_m"]),
                "original_found": original_found,
                "radiation_aware_found": aware_found,
                "original_discovery_distance_m": (
                    None
                    if not left["first_discovery_distance_m"]
                    else float(left["first_discovery_distance_m"])
                ),
                "radiation_aware_discovery_distance_m": (
                    None
                    if not right["first_discovery_distance_m"]
                    else float(right["first_discovery_distance_m"])
                ),
                "discovery_distance_delta_aware_minus_original_m": (
                    None
                    if not left["first_discovery_distance_m"]
                    or not right["first_discovery_distance_m"]
                    else float(right["first_discovery_distance_m"])
                    - float(left["first_discovery_distance_m"])
                ),
                "original_pre_discovery_total_dose": original_dose,
                "radiation_aware_pre_discovery_total_dose": aware_dose,
                "dose_unit": scenario.contract.dose_unit,
                "dose_delta_original_minus_radiation_aware": reduction,
                "dose_reduction_ratio": (
                    None if original_dose == 0.0 else reduction / original_dose
                ),
                "dose_reduction_percent": (
                    None if original_dose == 0.0 else 100.0 * reduction / original_dose
                ),
            }
        )
    summary = {
        "survivor_count": len(rows),
        "radiation_aware_dose_lower_count": lower,
        "radiation_aware_dose_higher_count": higher,
        "approximately_unchanged_count": unchanged,
        "original_unfound_radiation_aware_found_count": aware_only,
        "radiation_aware_unfound_original_found_count": original_only,
        "both_unfound_count": both_unfound,
        "unchanged_tolerance_rule": "max(1e-12, 1e-6 * max(abs(pair doses)))",
        "dose_unit": scenario.contract.dose_unit,
    }
    return rows, summary


def _truth_grid(
    context: FrozenContext,
    scenario: FrozenRadiationScenario,
    *,
    height_m: float = VALUE_REFERENCE_HEIGHT_M,
    resolution_m: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.arange(
        context.bounds[0] + resolution_m / 2.0,
        context.bounds[2],
        resolution_m,
    )
    y_values = np.arange(
        context.bounds[1] + resolution_m / 2.0,
        context.bounds[3],
        resolution_m,
    )
    field = np.empty((y_values.size, x_values.size), dtype=float)
    for row, y_m in enumerate(y_values):
        for col, x_m in enumerate(x_values):
            field[row, col] = scenario.background_rate + scenario.truth_query_excess(
                float(x_m),
                float(y_m),
                height_m,
            )
    return x_values, y_values, field


def _draw_route(axis, trajectory, *, label_prefix: str = "") -> None:
    segments = np.stack(
        (
            np.asarray(trajectory.coordinates[:-1]),
            np.asarray(trajectory.coordinates[1:]),
        ),
        axis=1,
    )
    modes = np.asarray([node.mode for node in trajectory.nodes[1:]])
    for mode, colour in (("NORMAL", "limegreen"), ("RADIATION", "deepskyblue")):
        selected = segments[modes == mode]
        if selected.size:
            axis.add_collection(
                LineCollection(
                    selected,
                    colors=colour,
                    linewidths=0.85,
                    alpha=0.9,
                    label=f"{label_prefix}{mode}",
                )
            )


def _draw_truth_sources(axis, scenario: FrozenRadiationScenario) -> None:
    if scenario.scenario_type == "surface_only":
        coordinates = np.asarray(
            scenario.config["polygon_exterior_coordinates"],
            dtype=float,
        )
        axis.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            color="cyan",
            linewidth=1.6,
            label="Surface truth boundary",
        )
    else:
        axis.scatter(
            [position[0] for position in scenario.source_positions_m],
            [position[1] for position in scenario.source_positions_m],
            marker="*",
            c="yellow",
            edgecolors="black",
            s=105,
            label="Point truth",
            zorder=8,
        )


def _format_axis(axis, context: FrozenContext, title: str) -> None:
    axis.set_title(title)
    axis.set_xlim(context.bounds[0], context.bounds[2])
    axis.set_ylim(context.bounds[1], context.bounds[3])
    axis.set_aspect("equal")
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")


def _load_survivor_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in _read_csv(path):
        records.append(
            {
                **row,
                "survivor_id": int(row["survivor_id"]),
                "x_m": float(row["x_m"]),
                "y_m": float(row["y_m"]),
                "found": row["found"].lower() == "true",
                "pre_discovery_total_dose": float(
                    row["pre_discovery_total_dose"]
                ),
            }
        )
    return records


def _comparison_figures(
    context: FrozenContext,
    scenario: FrozenRadiationScenario,
    pairwise: list[dict[str, object]],
) -> None:
    root = FINAL_ROOT / scenario.directory_name
    output = root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    aware_trajectory = load_executed_trajectory(
        root / "radiation_aware" / "simulation" / "mission_steps.csv"
    )
    original_trajectory = load_executed_trajectory(
        FINAL_ROOT / "baseline_original" / "simulation" / "mission_steps.csv"
    )
    x_values, y_values, truth = _truth_grid(context, scenario)
    positive = truth[truth > 0.0]
    contour_levels = None
    if positive.size and positive.max() > positive.min():
        contour_levels = np.geomspace(
            max(float(np.percentile(positive, 10)), 1e-12),
            float(positive.max()),
            5,
        )

    figure, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for axis, trajectory, title in (
        (axes[0], original_trajectory, "Original Greedy"),
        (axes[1], aware_trajectory, "Radiation-aware Greedy"),
    ):
        image = axis.imshow(
            context.probability_map,
            origin="lower",
            extent=(
                context.bounds[0],
                context.bounds[2],
                context.bounds[1],
                context.bounds[3],
            ),
            cmap="YlOrRd",
            vmin=0.0,
            vmax=float(context.probability_map.max()),
        )
        if contour_levels is not None:
            axis.contour(
                x_values,
                y_values,
                truth,
                levels=contour_levels,
                colors="black",
                linewidths=0.55,
                alpha=0.55,
            )
        _draw_route(axis, trajectory)
        axis.scatter(
            [point.x for point in context.survivors.points],
            [point.y for point in context.survivors.points],
            s=9,
            c="purple",
            alpha=0.55,
        )
        _draw_truth_sources(axis, scenario)
        _format_axis(axis, context, title)
    figure.colorbar(image, ax=axes, label="SAR probability per cell", shrink=0.8)
    figure.suptitle(f"{scenario.scenario_id}: frozen paired trajectories")
    figure.savefig(output / "01_trajectory_side_by_side.png", dpi=190)
    plt.close(figure)

    original_records = _load_survivor_records(
        root / "original" / "evaluation" / "survivor_radiation.csv"
    )
    aware_records = _load_survivor_records(
        root / "radiation_aware" / "evaluation" / "survivor_radiation.csv"
    )
    maximum_dose = max(
        max(float(record["pre_discovery_total_dose"]) for record in original_records),
        max(float(record["pre_discovery_total_dose"]) for record in aware_records),
    )
    dose_norm = Normalize(vmin=0.0, vmax=max(maximum_dose, 1e-12))
    figure, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    scatter = None
    for axis, records, title in (
        (axes[0], original_records, "Original Greedy"),
        (axes[1], aware_records, "Radiation-aware Greedy"),
    ):
        axis.imshow(
            np.log10(np.maximum(truth, 1e-12)),
            origin="lower",
            extent=(
                context.bounds[0], context.bounds[2],
                context.bounds[1], context.bounds[3],
            ),
            cmap="Greys",
            alpha=0.35,
        )
        for found, marker in ((True, "o"), (False, "X")):
            selected = [record for record in records if record["found"] is found]
            if selected:
                scatter = axis.scatter(
                    [record["x_m"] for record in selected],
                    [record["y_m"] for record in selected],
                    c=[record["pre_discovery_total_dose"] for record in selected],
                    cmap="plasma",
                    norm=dose_norm,
                    marker=marker,
                    edgecolors="black" if found else "white",
                    s=45 if found else 65,
                )
        _format_axis(axis, context, title)
    if scatter is not None:
        figure.colorbar(
            scatter,
            ax=axes,
            label=f"Pre-discovery total ({scenario.contract.dose_unit})",
            shrink=0.8,
        )
    figure.suptitle(f"{scenario.scenario_id}: survivor pre-discovery exposure")
    figure.savefig(output / "02_survivor_dose_side_by_side.png", dpi=190)
    plt.close(figure)

    differences = np.asarray(
        [record["dose_delta_original_minus_radiation_aware"] for record in pairwise],
        dtype=float,
    )
    limit = max(float(np.max(np.abs(differences))), 1e-12)
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    scatter = axis.scatter(
        [record["x_m"] for record in pairwise],
        [record["y_m"] for record in pairwise],
        c=differences,
        cmap="coolwarm",
        norm=Normalize(vmin=-limit, vmax=limit),
        edgecolors="black",
        s=55,
    )
    figure.colorbar(
        scatter,
        ax=axis,
        label=(
            "Original - Radiation-aware pre-discovery total "
            f"({scenario.contract.dose_unit})"
        ),
    )
    _draw_truth_sources(axis, scenario)
    _format_axis(axis, context, "Positive values indicate Radiation-aware reduction")
    figure.savefig(output / "03_survivor_dose_difference_map.png", dpi=190)
    plt.close(figure)

    original_curve = _read_csv(
        root / "original" / "evaluation" / "cumulative_metrics.csv"
    )
    aware_curve = _read_csv(
        root / "radiation_aware" / "evaluation" / "cumulative_metrics.csv"
    )
    figure, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    for records, label, colour in (
        (original_curve, "Original", "darkorange"),
        (aware_curve, "Radiation-aware", "deepskyblue"),
    ):
        distance = np.asarray([float(row["path_distance_m"]) for row in records])
        axes[0].plot(
            distance,
            [float(row["cumulative_sar_probability"]) for row in records],
            label=label,
            color=colour,
        )
        axes[1].plot(
            distance,
            [float(row["cumulative_survivors_found"]) for row in records],
            label=label,
            color=colour,
        )
        axes[2].plot(
            distance,
            [float(row["cumulative_uav_total_distance_integral"]) for row in records],
            label=label,
            color=colour,
        )
    axes[0].set_ylabel("Cumulative SAR probability")
    axes[1].set_ylabel("Survivors found")
    axes[2].set_ylabel(
        f"UAV total exposure integral\n({scenario.contract.distance_integral_unit})"
    )
    axes[2].set_xlabel("Path distance (m)")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.suptitle(f"{scenario.scenario_id}: cumulative paired results")
    figure.savefig(output / "04_cumulative_curves_overlay.png", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(
        np.log10(np.maximum(truth, 1e-12)),
        origin="lower",
        extent=(
            context.bounds[0], context.bounds[2],
            context.bounds[1], context.bounds[3],
        ),
        cmap="magma",
    )
    original_xy = np.asarray(original_trajectory.coordinates)
    aware_xy = np.asarray(aware_trajectory.coordinates)
    axis.plot(original_xy[:, 0], original_xy[:, 1], color="white", linewidth=0.8, alpha=0.8, label="Original")
    axis.plot(aware_xy[:, 0], aware_xy[:, 1], color="cyan", linewidth=0.8, alpha=0.8, label="Radiation-aware")
    _draw_truth_sources(axis, scenario)
    _format_axis(axis, context, "Both routes over radiation truth")
    axis.legend(fontsize=8)
    figure.colorbar(
        image,
        ax=axis,
        label=f"log10 total rate ({scenario.contract.rate_unit})",
    )
    figure.savefig(output / "05_radiation_truth_both_routes.png", dpi=190)
    plt.close(figure)


def generate_comparison(
    context: FrozenContext,
    scenario: FrozenRadiationScenario,
) -> list[dict[str, object]]:
    root = FINAL_ROOT / scenario.directory_name
    original = _read_json(root / "original" / "evaluation" / "evaluation_summary.json")
    aware = _read_json(
        root / "radiation_aware" / "evaluation" / "evaluation_summary.json"
    )
    metrics = _comparison_metrics(original, aware, scenario.contract)
    pairwise, pairwise_summary = _pairwise_survivors(scenario)
    comparison_directory = root / "comparison"
    _csv(comparison_directory / "comparison_summary.csv", metrics)
    _json(
        comparison_directory / "comparison_summary.json",
        {
            "scenario_id": scenario.scenario_id,
            "radiation_quantity": scenario.contract.quantity,
            "radiation_rate_unit": scenario.contract.rate_unit,
            "radiation_dose_unit": scenario.contract.dose_unit,
            "metrics": metrics,
            "paired_survivor_summary": pairwise_summary,
        },
    )
    _csv(
        comparison_directory / "survivor_pairwise_comparison.csv",
        pairwise,
    )
    _json(
        comparison_directory / "survivor_pairwise_summary.json",
        pairwise_summary,
    )
    _comparison_figures(context, scenario, pairwise)
    return metrics


def _summary_row(
    scenario: FrozenRadiationScenario,
    planner: str,
) -> dict[str, object]:
    root = FINAL_ROOT / scenario.directory_name
    evaluation = _read_json(root / planner / "evaluation" / "evaluation_summary.json")
    simulation = (
        _read_json(root / "radiation_aware" / "simulation" / "summary.json")
        if planner == "radiation_aware"
        else _read_json(FINAL_ROOT / "baseline_original" / "simulation" / "summary.json")
    )
    survivor_distribution = _nested(
        evaluation,
        "survivor_radiation",
        "all_survivors",
        "pre_discovery_total_dose",
    )
    return {
        "scenario": scenario.scenario_id,
        "scenario_type": scenario.scenario_type,
        "planner": "Radiation-aware" if planner == "radiation_aware" else "Original",
        "radiation_quantity": scenario.contract.quantity,
        "radiation_rate_unit": scenario.contract.rate_unit,
        "radiation_dose_unit": scenario.contract.dose_unit,
        "sar_likelihood_score": _nested(evaluation, "sar", "total_likelihood_score"),
        "sar_time_discounted_score": _nested(
            evaluation, "sar", "total_time_discounted_score"
        ),
        "survivors_found": _nested(
            evaluation, "survivor_radiation", "found_count"
        ),
        "route_distance_m": _nested(evaluation, "trajectory", "total_distance_m"),
        "uav_excess_exposure_distance_integral": _nested(
            evaluation,
            "uav_radiation",
            "uav_excess_exposure_distance_integral",
        ),
        "uav_total_exposure_distance_integral": _nested(
            evaluation,
            "uav_radiation",
            "uav_total_exposure_distance_integral",
        ),
        "uav_cumulative_excess_dose": _nested(
            evaluation, "uav_radiation", "uav_cumulative_excess_dose"
        ),
        "uav_cumulative_total_dose": _nested(
            evaluation, "uav_radiation", "uav_cumulative_total_dose"
        ),
        "survivor_total_pre_discovery_dose": survivor_distribution["sum"],
        "survivor_mean_pre_discovery_dose": survivor_distribution["mean"],
        "survivor_median_pre_discovery_dose": survivor_distribution["median"],
        "survivor_p95_pre_discovery_dose": survivor_distribution["p95"],
        "survivor_max_pre_discovery_dose": survivor_distribution["max"],
        "radiation_episodes": (
            simulation.get("radiation_episodes", 0)
            if isinstance(simulation, dict)
            else 0
        ),
        "radiation_steps": (
            simulation.get("radiation_steps", 0)
            if isinstance(simulation, dict)
            else 0
        ),
        "radiation_priority_decisions": (
            simulation.get("radiation_priority_decisions", 0)
            if isinstance(simulation, dict)
            else 0
        ),
    }


def generate_combined_outputs(
    context: FrozenContext,
    scenarios: tuple[FrozenRadiationScenario, ...],
) -> None:
    combined = FINAL_ROOT / "combined"
    figures = combined / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = [
        _summary_row(scenario, planner)
        for scenario in scenarios
        for planner in ("original", "radiation_aware")
    ]
    paired_changes = {}
    for scenario in scenarios:
        comparison = _read_json(
            FINAL_ROOT
            / scenario.directory_name
            / "comparison"
            / "comparison_summary.json"
        )
        paired_changes[scenario.scenario_id] = comparison
        relative_by_metric = {
            row["metric"]: row["relative_change_percent"]
            for row in comparison["metrics"]
        }
        for row in rows:
            if row["scenario"] != scenario.scenario_id:
                continue
            is_aware = row["planner"] == "Radiation-aware"
            for metric_name in (
                "sar_likelihood_score",
                "sar_time_discounted_score",
                "survivors_found",
                "route_distance",
                "uav_excess_exposure_distance_integral",
                "uav_total_exposure_distance_integral",
                "survivor_total_pre_discovery_dose",
                "survivor_mean_pre_discovery_dose",
                "survivor_p95_pre_discovery_dose",
                "survivor_max_pre_discovery_dose",
            ):
                row[f"{metric_name}_relative_change_percent_vs_original"] = (
                    relative_by_metric[metric_name] if is_aware else None
                )
    _csv(combined / "final_summary.csv", rows)
    _json(
        combined / "final_summary.json",
        {
            "rows": rows,
            "paired_comparisons": paired_changes,
            "cross_scenario_radiation_unit_warning": (
                "Point scenarios use uSv/h synthetic gamma dose rate; Surface "
                "uses uGy/h collision air kerma. Absolute radiation values are "
                "not compared across those unit contracts."
            ),
        },
    )

    original = load_executed_trajectory(
        FINAL_ROOT / "baseline_original" / "simulation" / "mission_steps.csv"
    )
    figure, axes = plt.subplots(3, 2, figsize=(15, 20), constrained_layout=True)
    for row_index, scenario in enumerate(scenarios):
        aware = load_executed_trajectory(
            FINAL_ROOT
            / scenario.directory_name
            / "radiation_aware"
            / "simulation"
            / "mission_steps.csv"
        )
        _, _, truth = _truth_grid(context, scenario, resolution_m=5.0)
        truth_log = np.log10(np.maximum(truth, 1e-12))
        minimum = float(truth_log.min())
        maximum = float(truth_log.max())
        for col_index, (trajectory, label) in enumerate(
            ((original, "Original"), (aware, "Radiation-aware"))
        ):
            axis = axes[row_index, col_index]
            axis.imshow(
                truth_log,
                origin="lower",
                extent=(
                    context.bounds[0], context.bounds[2],
                    context.bounds[1], context.bounds[3],
                ),
                cmap="magma",
                vmin=minimum,
                vmax=maximum,
            )
            _draw_route(axis, trajectory)
            _draw_truth_sources(axis, scenario)
            _format_axis(
                axis,
                context,
                f"{scenario.scenario_type}: {label}\n({scenario.contract.rate_unit})",
            )
    figure.suptitle("FINAL experiment trajectories (row-wise radiation scale)", fontsize=16)
    figure.savefig(figures / "all_scenarios_trajectory_matrix.png", dpi=190)
    plt.close(figure)

    baseline_path = FINAL_ROOT / "baseline_original" / "simulation" / "mission_steps.csv"
    baseline_hash = _sha256(baseline_path)
    scenario_checks = {}
    for scenario in scenarios:
        root = FINAL_ROOT / scenario.directory_name
        original_parameters = _read_json(
            root / "original" / "evaluation" / "evaluation_parameters.json"
        )
        aware_parameters = _read_json(
            root / "radiation_aware" / "evaluation" / "evaluation_parameters.json"
        )
        simulation = _read_json(
            root / "radiation_aware" / "simulation" / "summary.json"
        )
        scenario_checks[scenario.scenario_id] = {
            "scenario_config_sha256": _sha256(root / "scenario_config.json"),
            "original_evaluation_used_canonical_route": (
                original_parameters["input_sha256"]["trajectory"] == baseline_hash
            ),
            "same_heatmap_hash": (
                original_parameters["input_sha256"]["heatmap"]
                == aware_parameters["input_sha256"]["heatmap"]
                == context.heatmap_sha256
            ),
            "same_survivor_hash": (
                original_parameters["input_sha256"]["survivors"]
                == aware_parameters["input_sha256"]["survivors"]
                == context.survivor_sha256
            ),
            "same_fov": (
                original_parameters["fov_deg"] == aware_parameters["fov_deg"] == FOV_DEG
            ),
            "same_altitude": (
                original_parameters["platform_altitude_m"]
                == aware_parameters["platform_altitude_m"]
                == PLATFORM_ALTITUDE_M
            ),
            "radiation_aware_complete": simulation["status"] == "COMPLETE",
            "radiation_unit_contract": {
                "quantity": scenario.contract.quantity,
                "rate_unit": scenario.contract.rate_unit,
                "dose_unit": scenario.contract.dose_unit,
            },
        }
    integrity = {
        "canonical_original_route_sha256": baseline_hash,
        "heatmap_sha256": context.heatmap_sha256,
        "survivors_sha256": context.survivor_sha256,
        "planner_seed": PLANNER_SEED,
        "same_start_definition": "frozen SAR environment centre mapped by native Greedy",
        "original_radiation_blind": True,
        "planner_full_truth_access": False,
        "sensor_truth_access_only_at_executed_nodes": True,
        "evaluator_truth_access": True,
        "radiation_changes_sar_truth": False,
        "radiation_changes_survivor_truth": False,
        "scenario_checks": scenario_checks,
        "all_checks_pass": all(
            all(
                value
                for key, value in checks.items()
                if key not in {"scenario_config_sha256", "radiation_unit_contract"}
            )
            for checks in scenario_checks.values()
        ),
    }
    _json(combined / "integrity_checks.json", integrity)


def _record_original_reference(scenario: FrozenRadiationScenario) -> None:
    baseline_summary = _read_json(
        FINAL_ROOT / "baseline_original" / "simulation" / "summary.json"
    )
    _json(
        FINAL_ROOT / scenario.directory_name / "original" / "baseline_route_reference.json",
        {
            "canonical_baseline": str(
                FINAL_ROOT / "baseline_original" / "simulation" / "mission_steps.csv"
            ),
            "route_sha256": baseline_summary["route_sha256"],
            "radiation_blind": True,
        },
    )


def run_one_scenario(
    context: FrozenContext,
    scenario: FrozenRadiationScenario,
    baseline_path: Path,
) -> None:
    _record_original_reference(scenario)
    aware_path = run_radiation_aware_mission(context, scenario)
    evaluate_route(
        context,
        scenario,
        planner_name="original",
        trajectory_path=baseline_path,
    )
    evaluate_route(
        context,
        scenario,
        planner_name="radiation_aware",
        trajectory_path=aware_path,
    )
    generate_comparison(context, scenario)


def run_final_experiment() -> None:
    context = load_frozen_context()
    snapshot_code_state(context)
    baseline_path = generate_original_baseline(context)
    single = freeze_point_scenario(context, multi=False)
    run_one_scenario(context, single, baseline_path)
    multiple = freeze_point_scenario(context, multi=True)
    run_one_scenario(context, multiple, baseline_path)
    surface = freeze_surface_scenario(context)
    run_one_scenario(context, surface, baseline_path)
    generate_combined_outputs(context, (single, multiple, surface))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("preflight", "baseline", "single", "multi", "surface", "combined", "all"),
        nargs="?",
        default="all",
    )
    args = parser.parse_args()
    context = load_frozen_context()
    baseline_path = FINAL_ROOT / "baseline_original" / "simulation" / "mission_steps.csv"
    if args.phase in ("preflight", "all"):
        snapshot_code_state(context)
        if args.phase == "preflight":
            print(json.dumps(_common_parameters(context), indent=2, sort_keys=True))
            return
    if args.phase in ("baseline", "all"):
        baseline_path = generate_original_baseline(context)
        if args.phase == "baseline":
            return
    if not baseline_path.exists():
        raise FileNotFoundError("Generate the canonical baseline first.")
    scenarios: dict[str, FrozenRadiationScenario] = {}
    if args.phase in ("single", "all", "combined"):
        scenarios["single"] = freeze_point_scenario(context, multi=False)
    if args.phase in ("multi", "all", "combined"):
        scenarios["multi"] = freeze_point_scenario(context, multi=True)
    if args.phase in ("surface", "all", "combined"):
        scenarios["surface"] = freeze_surface_scenario(context)
    if args.phase == "single":
        run_one_scenario(context, scenarios["single"], baseline_path)
    elif args.phase == "multi":
        run_one_scenario(context, scenarios["multi"], baseline_path)
    elif args.phase == "surface":
        run_one_scenario(context, scenarios["surface"], baseline_path)
    elif args.phase == "combined":
        generate_combined_outputs(
            context,
            (scenarios["single"], scenarios["multi"], scenarios["surface"]),
        )
    elif args.phase == "all":
        for name in ("single", "multi", "surface"):
            run_one_scenario(context, scenarios[name], baseline_path)
        generate_combined_outputs(
            context,
            (scenarios["single"], scenarios["multi"], scenarios["surface"]),
        )


if __name__ == "__main__":
    main()
