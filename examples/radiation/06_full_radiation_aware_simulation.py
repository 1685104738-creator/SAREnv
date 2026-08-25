"""Run full Single-Point radiation-aware missions on the Direct-Small dataset.

This first implementation experiment assumes perfect conversion from the
50 m platform observation to a scalar ground-equivalent value at 1 m AGL.
Only the sensor reads radiation truth; the estimator and planner receive
executed-position measurements and online estimates respectively.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sarenv import DatasetLoader, load_lost_person_locations
from sarenv.analytics.mission_mode import MissionMode, MissionModeController
from sarenv.analytics.paths import (
    generate_greedy_path,
    native_greedy_step_allowance,
)
from sarenv.analytics.radiation_priority import RadiationGreedyStepPlanner
from sarenv.analytics.route_execution import (
    ExecutedNodeRadiationObserver,
    ExecutedSARState,
    PrecomputedRouteController,
)
from sarenv.radiation import GridSpec, PointSource, PointSourceConfig
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = (
    REPOSITORY_ROOT
    / "examples"
    / "sarenv_dataset"
    / "sarenv_outputs"
    / "radiation_area_01_small_20m"
)
OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "radiation_aware_adaptive_entry"

SCENARIO_ID = "single_point_small_20m_perfect_ground_equivalent"
PLANNER_SEED = 42
FOV_DEG = 45.0
PLATFORM_ALTITUDE_M = 50.0
VALUE_REFERENCE_HEIGHT_M = 1.0
BACKGROUND_USV_H = 0.20
POINT_REFERENCE_EXCESS_USV_H = 20_000.0
POINT_OFFSET_FROM_SEARCH_CENTER_M = (-130.0, -190.0)

# Provisional planner benchmark scale, not a regulatory hazard classification.
NOMINAL_HAZARD_REFERENCE_EXCESS_USV_H = 10.0
INITIAL_ENTRY_MULTIPLIERS = (0.01, 0.1, 1.0)
HYSTERESIS_MARGIN = NOMINAL_HYSTERESIS_MARGIN
REFERENCE_RADIUS_FRACTION = 0.5

MEASUREMENT_QUANTITY = "ground_equivalent_excess_gamma_dose_rate"
MEASUREMENT_UNIT = "uSv/h"
SAMPLING_CONTRACT = "one_noise_free_measurement_per_executed_route_node"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError(f"Cannot write empty mission output: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _case_name(initial_entry_multiplier: float) -> str:
    return {
        0.01: "initial_entry_0p01x",
        0.1: "initial_entry_0p1x",
        1.0: "initial_entry_1x_control",
    }[float(initial_entry_multiplier)]


def _truth_contour_levels() -> np.ndarray:
    """Return one fixed scale shared by every sensitivity figure."""
    return np.asarray([1.0, 10.0, 100.0, 1_000.0, 10_000.0])


def _plot_trajectory(
    output_path: Path,
    *,
    probability_map: np.ndarray,
    bounds: tuple[float, float, float, float],
    radiation_grid: GridSpec,
    truth_field: np.ndarray,
    source_position: tuple[float, float],
    survivor_points,
    steps: list[dict[str, object]],
    events: list[dict[str, object]],
    initial_entry_multiplier: float,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    probability_image = axis.imshow(
        probability_map,
        origin="lower",
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        cmap="YlOrRd",
        interpolation="nearest",
    )
    figure.colorbar(
        probability_image,
        ax=axis,
        label="Lost-person probability per SAR cell",
        fraction=0.045,
    )
    x_centres = radiation_grid.minx + (
        np.arange(radiation_grid.width) + 0.5
    ) * radiation_grid.resolution_m
    y_centres = radiation_grid.miny + (
        np.arange(radiation_grid.height) + 0.5
    ) * radiation_grid.resolution_m
    axis.contour(
        x_centres,
        y_centres,
        truth_field,
        levels=_truth_contour_levels(),
        colors="black",
        linewidths=0.7,
        alpha=0.45,
    )

    for mode, colour, label in (
        (MissionMode.NORMAL.value, "limegreen", "NORMAL execution"),
        (MissionMode.RADIATION.value, "deepskyblue", "RADIATION execution"),
    ):
        labelled = False
        for previous, current in zip(steps, steps[1:]):
            if current["mode_before"] != mode:
                continue
            axis.plot(
                [previous["x_m"], current["x_m"]],
                [previous["y_m"], current["y_m"]],
                color=colour,
                linewidth=1.25,
                alpha=0.9,
                label=label if not labelled else None,
            )
            labelled = True

    start = steps[0]
    axis.scatter(
        [start["x_m"]],
        [start["y_m"]],
        marker="o",
        s=75,
        c="white",
        edgecolors="black",
        zorder=6,
        label="Start",
    )
    axis.scatter(
        [point.x for point in survivor_points],
        [point.y for point in survivor_points],
        marker=".",
        s=18,
        c="purple",
        alpha=0.55,
        zorder=5,
        label="Saved survivors",
    )
    enter_events = [event for event in events if event["transition_reason"] == "hazard_enter_threshold"]
    exit_events = [event for event in events if event["transition_reason"] == "hazard_exit_threshold"]
    if enter_events:
        axis.scatter(
            [event["x_m"] for event in enter_events],
            [event["y_m"] for event in enter_events],
            marker="X",
            s=90,
            c="magenta",
            edgecolors="black",
            zorder=7,
            label="Enter RADIATION",
        )
    if exit_events:
        axis.scatter(
            [event["x_m"] for event in exit_events],
            [event["y_m"] for event in exit_events],
            marker="D",
            s=60,
            c="cyan",
            edgecolors="black",
            zorder=7,
            label="Exit RADIATION",
        )
    axis.scatter(
        [source_position[0]],
        [source_position[1]],
        marker="*",
        s=170,
        c="yellow",
        edgecolors="black",
        zorder=8,
        label="Point truth (evaluation only)",
    )
    axis.set_title(
        "Full Radiation-aware Mission\n"
        f"Initial entry multiplier = {initial_entry_multiplier:g}×"
    )
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.set_xlim(bounds[0], bounds[2])
    axis.set_ylim(bounds[1], bounds[3])
    axis.set_aspect("equal")
    axis.legend(loc="upper right", fontsize=8)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_estimate(
    output_path: Path,
    *,
    estimator: IncrementalRadiationGrid,
    steps: list[dict[str, object]],
    source_position: tuple[float, float],
) -> None:
    bounds = estimator.grid.bounds
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    image = axes[0].imshow(
        estimator.mean,
        origin="lower",
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        cmap="inferno",
        vmin=0.0,
        vmax=POINT_REFERENCE_EXCESS_USV_H,
    )
    figure.colorbar(image, ax=axes[0], label="Estimated excess (uSv/h)")
    axes[1].imshow(
        estimator.observed_mask,
        origin="lower",
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    for axis in axes:
        axis.plot(
            [step["x_m"] for step in steps],
            [step["y_m"] for step in steps],
            color="deepskyblue",
            linewidth=0.8,
            alpha=0.7,
        )
        axis.scatter(
            [source_position[0]],
            [source_position[1]],
            c="yellow",
            edgecolors="black",
            marker="*",
            s=130,
        )
        axis.set_aspect("equal")
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")
    axes[0].set_title("Numeric estimated excess (unsupported = 0)")
    axes[1].set_title("Independent measurement support mask")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run_full_mission(
    initial_entry_multiplier: float,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, object]:
    """Run one complete mission without a fixed distance budget."""
    started = time.perf_counter()
    item = DatasetLoader(str(DATASET_DIRECTORY)).load_environment()
    if item.size != "small":
        raise ValueError("Formal runner requires an authoritative Small dataset.")
    if not np.isclose(item.meter_per_bin, 20.0):
        raise ValueError("Formal runner requires the configured 20 m SAR dataset.")
    if not np.isclose(item.heatmap.sum(), 1.0, atol=1e-12):
        raise ValueError("Small probability truth must sum to one.")

    survivor_path = DATASET_DIRECTORY / "lost_persons.json"
    survivors = load_lost_person_locations(survivor_path)
    if survivors.environment_size != item.size:
        raise ValueError("Saved survivors do not match the SAR environment size.")
    if survivors.bounds != tuple(item.bounds):
        raise ValueError("Saved survivors do not match the SAR bounds.")

    probability_map = np.asarray(item.heatmap, dtype=float)
    bounds = tuple(float(value) for value in item.bounds)
    search_center = (
        (bounds[0] + bounds[2]) / 2.0,
        (bounds[1] + bounds[3]) / 2.0,
    )
    max_radius_m = float(item.radius_km) * 1_000.0
    detection_radius_m = PLATFORM_ALTITUDE_M * np.tan(
        np.radians(FOV_DEG / 2.0)
    )
    mission_step_allowance = native_greedy_step_allowance(probability_map.shape)
    planner_rng = np.random.default_rng(PLANNER_SEED)

    initial_route = generate_greedy_path(
        center_x=search_center[0],
        center_y=search_center[1],
        num_drones=1,
        probability_map=probability_map,
        bounds=bounds,
        max_radius=max_radius_m,
        fov_deg=FOV_DEG,
        altitude=PLATFORM_ALTITUDE_M,
        max_steps=mission_step_allowance,
        rng=planner_rng,
    )[0]
    state = ExecutedSARState(
        map_shape=probability_map.shape,
        bounds=bounds,
        detection_radius_m=detection_radius_m,
    )
    normal_controller = PrecomputedRouteController(initial_route, state)

    source_position = (
        search_center[0] + POINT_OFFSET_FROM_SEARCH_CENTER_M[0],
        search_center[1] + POINT_OFFSET_FROM_SEARCH_CENTER_M[1],
    )
    point_source = PointSource(
        PointSourceConfig(
            source_id="single_point_formal_first_run",
            x_m=source_position[0],
            y_m=source_position[1],
            reference_excess_uSv_h=POINT_REFERENCE_EXCESS_USV_H,
            crs=str(item.projected_crs),
        )
    )
    truth = CompositeRadiationField(
        crs=str(item.projected_crs),
        background_uSv_h=BACKGROUND_USV_H,
        point_sources=(point_source,),
    )
    sensor = NoiseFreeRadiationSensor(
        truth.query_total_dose_rate,
        quantity=MEASUREMENT_QUANTITY,
        unit=MEASUREMENT_UNIT,
        value_reference_height_m=VALUE_REFERENCE_HEIGHT_M,
        background_value_to_subtract=BACKGROUND_USV_H,
    )
    radiation_grid = GridSpec.from_bounds(bounds, str(item.projected_crs))
    estimator = IncrementalRadiationGrid(
        radiation_grid,
        quantity=MEASUREMENT_QUANTITY,
        unit=MEASUREMENT_UNIT,
        update_radius_m=RADIATION_UPDATE_RADIUS_M,
        kernel_sigma_m=RADIATION_KERNEL_SIGMA_M,
        value_reference_height_m=VALUE_REFERENCE_HEIGHT_M,
    )
    base_hazard_reference_excess = NOMINAL_HAZARD_REFERENCE_EXCESS_USV_H
    trigger = AdaptiveHysteresisRadiationTrigger(
        base_hazard_reference=base_hazard_reference_excess,
        initial_entry_multiplier=float(initial_entry_multiplier),
        quantity=MEASUREMENT_QUANTITY,
        unit=MEASUREMENT_UNIT,
        hysteresis_margin=HYSTERESIS_MARGIN,
        confirmation_samples=NOISE_FREE_CONFIRMATION_SAMPLES,
    )
    observer = ExecutedNodeRadiationObserver(sensor, estimator, trigger)
    radiation_planner = RadiationGreedyStepPlanner(
        probability_map=probability_map,
        bounds=bounds,
        search_center=search_center,
        max_radius=max_radius_m,
        detection_radius_m=detection_radius_m,
        estimator=estimator,
        hazard_reference_excess=base_hazard_reference_excess,
        reference_radius_fraction=REFERENCE_RADIUS_FRACTION,
        rng=planner_rng,
    )
    controller = MissionModeController(
        state=state,
        normal_route_controller=normal_controller,
        observer=observer,
        radiation_planner=radiation_planner,
        initial_budget_m=None,
        mission_step_allowance=mission_step_allowance,
        planner_rng=planner_rng,
        probability_map=probability_map,
        bounds=bounds,
        search_center=search_center,
        max_radius=max_radius_m,
        fov_deg=FOV_DEG,
        altitude_m=PLATFORM_ALTITUDE_M,
    )

    steps: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    maximum_controller_calls = mission_step_allowance * 2 + 10
    while not controller.is_complete:
        if len(events) >= maximum_controller_calls:
            raise RuntimeError("Mission exceeded its derived controller-call guard.")
        result = controller.advance(simulated_time_s=0.0)
        execution = result.route_execution
        observation = result.radiation_observation
        event_x = execution.position_m[0] if execution is not None else None
        event_y = execution.position_m[1] if execution is not None else None
        hazard_transition = (
            observation.hazard_transition if observation is not None else None
        )
        events.append(
            {
                "controller_call_index": result.controller_call_index,
                "executed_step_index": result.executed_step_index,
                "mode_before": result.mode_before.value,
                "mode_after": result.mode_after.value,
                "moved": result.moved,
                "x_m": event_x,
                "y_m": event_y,
                "event_type": (
                    hazard_transition.upper()
                    if hazard_transition is not None
                    else ("COMPLETE" if result.mission_complete else None)
                ),
                "episode_index": (
                    observation.hazard_episode_index
                    if observation is not None
                    else None
                ),
                "entry_multiplier_initial": (
                    observation.entry_multiplier_initial
                    if observation is not None
                    else float(initial_entry_multiplier)
                ),
                "entry_threshold_current": (
                    observation.entry_threshold_current
                    if observation is not None
                    else trigger.current_entry_threshold
                ),
                "exit_threshold_current": (
                    observation.exit_threshold_current
                    if observation is not None
                    else trigger.current_exit_threshold
                ),
                "base_hazard_reference": base_hazard_reference_excess,
                "trigger_measurement": (
                    observation.measurement.value
                    if observation is not None
                    else None
                ),
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
                candidate_records.append(
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
                        "radiation_equivalent_priority": assessment.radiation_equivalent_priority,
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
            steps.append(
                {
                    "step_index": result.executed_step_index,
                    "mode_before": result.mode_before.value,
                    "mode_after": result.mode_after.value,
                    "x_m": execution.position_m[0],
                    "y_m": execution.position_m[1],
                    "sar_row": execution.grid_position[0],
                    "sar_col": execution.grid_position[1],
                    "step_distance_m": execution.distance_from_previous_m,
                    "total_distance_m": state.total_executed_distance_m,
                    "measurement_excess_uSv_h": measurement.value,
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
            )

    if not steps:
        raise RuntimeError("Full mission completed without executing a node.")
    if not candidate_records:
        candidate_records.append(
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

    output_directory = output_root / _case_name(initial_entry_multiplier)
    output_directory.mkdir(parents=True, exist_ok=True)
    mission_steps_path = output_directory / "mission_steps.csv"
    events_path = output_directory / "mode_events.csv"
    measurements_path = output_directory / "radiation_measurements.csv"
    candidates_path = output_directory / "candidate_decisions.csv"
    estimate_path = output_directory / "estimated_radiation_grid.npz"
    parameters_path = output_directory / "resolved_parameters.json"
    summary_path = output_directory / "summary.json"
    trajectory_path = output_directory / "trajectory.png"
    estimate_figure_path = output_directory / "estimated_radiation.png"

    _write_csv(mission_steps_path, steps)
    _write_csv(events_path, events)
    _write_csv(candidates_path, candidate_records)
    _write_csv(
        measurements_path,
        [
            {
                "step_index": step["step_index"],
                "x_m": step["x_m"],
                "y_m": step["y_m"],
                "platform_altitude_m": step["platform_altitude_m"],
                "value_reference_height_m": step["value_reference_height_m"],
                "value": step["measurement_excess_uSv_h"],
                "quantity": MEASUREMENT_QUANTITY,
                "unit": MEASUREMENT_UNIT,
                "sampling_contract": SAMPLING_CONTRACT,
            }
            for step in steps
        ],
    )
    np.savez_compressed(
        estimate_path,
        mean=np.asarray(estimator.mean),
        weight_sum=np.asarray(estimator.weight_sum),
        observed_mask=np.asarray(estimator.observed_mask),
        bounds=np.asarray(estimator.grid.bounds),
        resolution_m=estimator.grid.resolution_m,
        value_reference_height_m=VALUE_REFERENCE_HEIGHT_M,
        quantity=MEASUREMENT_QUANTITY,
        unit=MEASUREMENT_UNIT,
    )

    truth_field = point_source.rasterize(
        radiation_grid,
        z_m=VALUE_REFERENCE_HEIGHT_M,
    ) + BACKGROUND_USV_H
    _plot_trajectory(
        trajectory_path,
        probability_map=probability_map,
        bounds=bounds,
        radiation_grid=radiation_grid,
        truth_field=truth_field,
        source_position=source_position,
        survivor_points=survivors.points,
        steps=steps,
        events=events,
        initial_entry_multiplier=initial_entry_multiplier,
    )
    _plot_estimate(
        estimate_figure_path,
        estimator=estimator,
        steps=steps,
        source_position=source_position,
    )

    resolved_parameters = {
        "scenario_id": SCENARIO_ID,
        "scenario_type": "point_only",
        "dataset_directory": str(DATASET_DIRECTORY),
        "dataset_size": item.size,
        "dataset_bounds_projected": list(bounds),
        "dataset_crs": str(item.projected_crs),
        "sar_resolution_m": float(item.meter_per_bin),
        "sar_shape": list(probability_map.shape),
        "sar_probability_sum": float(probability_map.sum()),
        "radiation_resolution_m": radiation_grid.resolution_m,
        "radiation_grid_shape": list(radiation_grid.shape),
        "fov_deg": FOV_DEG,
        "camera_detection_radius_m": float(detection_radius_m),
        "platform_altitude_m": PLATFORM_ALTITUDE_M,
        "value_reference_height_m": VALUE_REFERENCE_HEIGHT_M,
        "altitude_correction_assumption": "perfect_ground_equivalent_scalar",
        "measurement_quantity": MEASUREMENT_QUANTITY,
        "measurement_unit": MEASUREMENT_UNIT,
        "background_uSv_h": BACKGROUND_USV_H,
        "estimated_quantity_contract": "excess_above_background_numeric_zero_prior",
        "point_source_truth_debug_evaluation_only": point_source.to_metadata(),
        "planner_seed": PLANNER_SEED,
        "hazard_reference_nominal_excess_uSv_h": NOMINAL_HAZARD_REFERENCE_EXCESS_USV_H,
        "hazard_reference_multiplier": 1.0,
        "hazard_reference_resolved_excess_uSv_h": base_hazard_reference_excess,
        "hazard_reference_resolved_total_uSv_h": base_hazard_reference_excess + BACKGROUND_USV_H,
        "initial_entry_multiplier": float(initial_entry_multiplier),
        "entry_increment_excess_uSv_h": trigger.initial_entry_threshold,
        "initial_entry_threshold_excess_uSv_h": trigger.initial_entry_threshold,
        "final_entry_threshold_excess_uSv_h": trigger.current_entry_threshold,
        "hysteresis_margin": HYSTERESIS_MARGIN,
        "final_T_enter_excess_uSv_h": trigger.current_entry_threshold,
        "final_T_exit_excess_uSv_h": trigger.current_exit_threshold,
        "adaptive_entry_formula": "min(episode_index * initial_entry_multiplier * base_hazard_reference, base_hazard_reference)",
        "exit_formula": "(1 - hysteresis_margin) * current_episode_entry_threshold",
        "confirmation_samples": trigger.confirmation_samples,
        "derived_sar_reference_score": radiation_planner.sar_reference.score,
        "sar_reference_derivation": {
            "method": radiation_planner.sar_reference.method,
            "reference_radius_fraction": radiation_planner.sar_reference.radius_fraction,
            "reference_radius_m": radiation_planner.sar_reference.radius_m,
            "ring_cell_count": radiation_planner.sar_reference.ring_cell_count,
        },
        "candidate_radiation_aggregation": "strict_max_over_overlapping_radiation_grid_cells",
        "gaussian_sigma_m": estimator.kernel_sigma_m,
        "gaussian_update_radius_m": estimator.update_radius_m,
        "mission_completion_rule": "all_searchable_cells_covered_or_native_greedy_move_ceiling",
        "native_mission_move_allowance": mission_step_allowance,
        "distance_budget_m": None,
        "old_debug_crop_used": False,
        "conditional_probability_renormalisation_used": False,
        "sampling_contract": SAMPLING_CONTRACT,
        "simulated_time_model": "not_modelled_node_sampling_time_is_zero",
        "heatmap_sha256": _sha256(DATASET_DIRECTORY / "heatmap.npy"),
        "survivors_sha256": _sha256(survivor_path),
        "survivor_count": len(survivors.points),
    }
    parameters_path.write_text(
        json.dumps(resolved_parameters, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    enter_events = [event for event in events if event["transition_reason"] == "hazard_enter_threshold"]
    exit_events = [event for event in events if event["transition_reason"] == "hazard_exit_threshold"]
    planner_time_s = sum(
        float(step["radiation_planning_time_s"] or 0.0) for step in steps
    )
    continuation_time_s = sum(
        float(step["continuation_planning_time_s"] or 0.0) for step in steps
    )
    estimator_time_s = sum(float(step["estimator_update_time_s"]) for step in steps)
    chosen_candidate_records = [
        record for record in candidate_records if record["chosen"] is True
    ]
    summary = {
        "status": "COMPLETE" if controller.is_complete else "INCOMPLETE",
        "scenario_id": SCENARIO_ID,
        "initial_entry_multiplier": float(initial_entry_multiplier),
        "base_hazard_reference_excess_uSv_h": base_hazard_reference_excess,
        "initial_entry_threshold_excess_uSv_h": trigger.initial_entry_threshold,
        "entry_increment_excess_uSv_h": trigger.initial_entry_threshold,
        "final_entry_threshold_excess_uSv_h": trigger.current_entry_threshold,
        "final_exit_threshold_excess_uSv_h": trigger.current_exit_threshold,
        "number_of_threshold_increments": trigger.threshold_increment_count,
        "entry_threshold_progression_excess_uSv_h": [
            event["entry_threshold_current"] for event in enter_events
        ],
        "exit_threshold_progression_excess_uSv_h": [
            event["exit_threshold_current"] for event in exit_events
        ],
        "total_executed_nodes": state.executed_node_count,
        "total_executed_moves": state.executed_move_count,
        "total_route_distance_m": state.total_executed_distance_m,
        "normal_steps": sum(step["mode_before"] == MissionMode.NORMAL.value for step in steps),
        "radiation_steps": sum(step["mode_before"] == MissionMode.RADIATION.value for step in steps),
        "radiation_episodes": len(enter_events),
        "radiation_priority_decisions": sum(
            record["planner_decision_reason"] == "radiation_priority"
            for record in chosen_candidate_records
        ),
        "original_greedy_fallback_decisions": sum(
            str(record["planner_decision_reason"]).endswith(
                "_original_greedy_step"
            )
            for record in chosen_candidate_records
        ),
        "enter_events": len(enter_events),
        "exit_events": len(exit_events),
        "mode_sequence": [mode.value for mode in controller.mode_history],
        "completion_reason": controller.completion_reason,
        "planner_runtime_s": planner_time_s + continuation_time_s,
        "radiation_step_planning_runtime_s": planner_time_s,
        "normal_continuation_planning_runtime_s": continuation_time_s,
        "total_continuation_replanning_runtime_s": continuation_time_s,
        "average_continuation_planning_runtime_per_exit_s": (
            continuation_time_s / len(exit_events) if exit_events else 0.0
        ),
        "estimator_runtime_s": estimator_time_s,
        "wall_runtime_s": time.perf_counter() - started,
        "measurement_count": len(steps),
        "observed_sar_cells": len(state.observed_cells),
        "searchable_sar_cells": len(controller.searchable_cells),
        "searchable_sar_coverage_fraction": (
            len(controller.searchable_cells.intersection(state.observed_cells))
            / len(controller.searchable_cells)
        ),
        "final_estimated_supported_cells": int(np.count_nonzero(estimator.observed_mask)),
        "final_estimated_total_cells": int(estimator.observed_mask.size),
        "final_estimated_coverage_fraction": float(np.mean(estimator.observed_mask)),
        "heatmap_sha256": resolved_parameters["heatmap_sha256"],
        "survivors_sha256": resolved_parameters["survivors_sha256"],
        "outputs": {
            "mission_steps_csv": str(mission_steps_path),
            "mode_events_csv": str(events_path),
            "radiation_measurements_csv": str(measurements_path),
            "candidate_decisions_csv": str(candidates_path),
            "estimated_radiation_grid_npz": str(estimate_path),
            "resolved_parameters_json": str(parameters_path),
            "trajectory_png": str(trajectory_path),
            "estimated_radiation_png": str(estimate_figure_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_sensitivity_suite() -> list[dict[str, object]]:
    """Run the three independent adaptive-entry cases without automatic tuning."""
    summaries = [
        run_full_mission(multiplier)
        for multiplier in INITIAL_ENTRY_MULTIPLIERS
    ]
    suite_path = OUTPUT_ROOT / "sensitivity_summary.json"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summaries


if __name__ == "__main__":
    run_sensitivity_suite()
