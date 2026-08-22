"""Run one DEBUG NODE-SAMPLING Radiation-priority Greedy mission.

This is a structural demonstration, not a dissertation experiment runner.
Every reached route node produces one measurement at an intentionally
unmodelled timestamp of 0.0 s. No UAV speed, sampling frequency, or sampling
spacing is implied by this script.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sarenv import DatasetLoader
from sarenv.analytics.mission_mode import MissionMode, MissionModeController
from sarenv.analytics.paths import (
    generate_greedy_path,
    greedy_world_to_grid_position,
)
from sarenv.analytics.radiation_priority import RadiationGreedyStepPlanner
from sarenv.analytics.route_execution import (
    ExecutedNodeRadiationObserver,
    ExecutedSARState,
    PrecomputedRouteController,
)
from sarenv.radiation import GridSpec, PointSource, PointSourceConfig
from sarenv.radiation.composite.field import CompositeRadiationField
from sarenv.radiation.online.estimator import IncrementalRadiationGrid
from sarenv.radiation.online.sensor import NoiseFreeRadiationSensor
from sarenv.radiation.online.trigger import (
    AboveBackgroundCriterion,
    ConsecutiveRadiationTrigger,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = (
    REPOSITORY_ROOT
    / "examples"
    / "sarenv_dataset"
    / "sarenv_outputs"
    / "radiation_area_01"
)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "radiation_priority_debug"

DEBUG_SAMPLING_MODE = "DEBUG_NODE_SAMPLING"
DEBUG_CROP_SIDE_CELLS = 21
DEBUG_INITIAL_BUDGET_M = 1_200.0
DEBUG_POINT_REFERENCE_EXCESS_USV_H = 20_000.0
DEBUG_POINT_SOURCE_ID = "preplanner_point_only"

# Approved project baseline and existing SAREnv camera default, respectively.
UAV_ALTITUDE_M = 50.0
FOV_DEG = 45.0
QUANTITY = "synthetic_total_gamma_dose_rate"
UNIT = "uSv/h"


def _debug_probability_crop(item):
    """Return a centred conditional-probability crop of an existing dataset."""
    center_x = (item.bounds[0] + item.bounds[2]) / 2.0
    center_y = (item.bounds[1] + item.bounds[3]) / 2.0
    center_row, center_col = greedy_world_to_grid_position(
        center_x,
        center_y,
        map_shape=item.heatmap.shape,
        bounds=item.bounds,
    )
    half_side = DEBUG_CROP_SIDE_CELLS // 2
    min_row = center_row - half_side
    max_row = center_row + half_side + 1
    min_col = center_col - half_side
    max_col = center_col + half_side + 1
    if min_row < 0 or min_col < 0:
        raise ValueError("DEBUG crop extends below the stored SAR raster.")
    if max_row > item.heatmap.shape[0] or max_col > item.heatmap.shape[1]:
        raise ValueError("DEBUG crop extends above the stored SAR raster.")

    probability_map = np.asarray(
        item.heatmap[min_row:max_row, min_col:max_col],
        dtype=float,
    ).copy()
    local_probability_mass = float(probability_map.sum())
    if local_probability_mass <= 0.0:
        raise ValueError("DEBUG crop contains no lost-person probability mass.")
    probability_map /= local_probability_mass

    resolution_m = float(item.meter_per_bin)
    bounds = (
        item.bounds[0] + min_col * resolution_m,
        item.bounds[1] + min_row * resolution_m,
        item.bounds[0] + max_col * resolution_m,
        item.bounds[1] + max_row * resolution_m,
    )
    return probability_map, bounds, (center_x, center_y), local_probability_mass


def _step_record(result, controller):
    execution = result.route_execution
    observation = result.radiation_observation
    if execution is None or observation is None:
        raise ValueError("An executed DEBUG step must include its measurement.")

    plan = result.radiation_plan
    candidate = plan.selected_candidate if plan is not None else None
    estimate = plan.decision.radiation_estimate if plan is not None else None
    return {
        "step_index": result.executed_step_index,
        "sampling_mode": DEBUG_SAMPLING_MODE,
        "mode": result.mode_before.value,
        "mode_after_step": result.mode_after.value,
        "x_m": execution.position_m[0],
        "y_m": execution.position_m[1],
        "executed_distance_step_m": execution.distance_from_previous_m,
        "total_executed_distance_m": controller.state.total_executed_distance_m,
        "remaining_budget_m": controller.remaining_budget_m,
        "simulated_time_s": observation.measurement.simulated_time_s,
        "radiation_measurement": observation.measurement.value,
        "radiation_quantity": observation.measurement.quantity,
        "radiation_unit": observation.measurement.unit,
        "radiation_trigger_confirmed": observation.anomaly_confirmed,
        "normal_route_interrupted": result.normal_route_interrupted,
        "selected_next_x_m": candidate.x_m if candidate is not None else None,
        "selected_next_y_m": candidate.y_m if candidate is not None else None,
        "sar_score": candidate.sar_score if candidate is not None else None,
        "predicted_radiation": (
            estimate.value if estimate is not None else None
        ),
        "planning_time_s": plan.planning_time_s if plan is not None else None,
    }


def _event_record(result):
    plan = result.radiation_plan
    return {
        "controller_call_index": result.controller_call_index,
        "executed_step_index": result.executed_step_index,
        "mode_before": result.mode_before.value,
        "mode_after": result.mode_after.value,
        "moved": result.moved,
        "normal_route_interrupted": result.normal_route_interrupted,
        "normal_route_replanned": result.normal_route_replanned,
        "radiation_decision_reason": (
            plan.decision.reason if plan is not None else None
        ),
        "radiation_planning_time_s": (
            plan.planning_time_s if plan is not None else None
        ),
        "continuation_planning_time_s": result.continuation_planning_time_s,
        "mission_complete": result.mission_complete,
    }


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError(f"Cannot write an empty DEBUG record set: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _plot_trajectory(
    output_path: Path,
    probability_map: np.ndarray,
    bounds: tuple[float, float, float, float],
    truth_field: np.ndarray,
    source_position: tuple[float, float],
    steps: list[dict[str, object]],
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(
        probability_map,
        origin="lower",
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        cmap="YlOrRd",
    )
    figure.colorbar(image, ax=axis, label="Conditional lost-person probability")

    truth_levels = np.linspace(float(truth_field.min()), float(truth_field.max()), 7)
    x_coordinates = np.linspace(bounds[0], bounds[2], truth_field.shape[1])
    y_coordinates = np.linspace(bounds[1], bounds[3], truth_field.shape[0])
    axis.contour(
        x_coordinates,
        y_coordinates,
        truth_field,
        levels=truth_levels[1:],
        colors="black",
        linewidths=0.6,
        alpha=0.38,
    )

    labelled_modes: set[str] = set()
    for previous, current in zip(steps, steps[1:]):
        mode = str(current["mode"])
        if mode == MissionMode.RADIATION.value:
            colour = "deepskyblue"
            label = "Radiation-priority execution"
        else:
            colour = "limegreen"
            label = "Normal Greedy execution"
        axis.plot(
            [previous["x_m"], current["x_m"]],
            [previous["y_m"], current["y_m"]],
            color=colour,
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label=label if mode not in labelled_modes else None,
        )
        labelled_modes.add(mode)

    start = steps[0]
    axis.scatter(
        [start["x_m"]],
        [start["y_m"]],
        c="white",
        edgecolors="black",
        s=85,
        marker="o",
        zorder=6,
        label="Start",
    )
    trigger_steps = [step for step in steps if step["normal_route_interrupted"]]
    if trigger_steps:
        axis.scatter(
            [step["x_m"] for step in trigger_steps],
            [step["y_m"] for step in trigger_steps],
            c="magenta",
            edgecolors="black",
            s=95,
            marker="X",
            zorder=7,
            label="Radiation trigger",
        )
    axis.scatter(
        [source_position[0]],
        [source_position[1]],
        c="cyan",
        edgecolors="black",
        s=150,
        marker="*",
        zorder=7,
        label="Point-source truth (debug only)",
    )
    axis.set_title("Radiation-priority Greedy — DEBUG NODE-SAMPLING MODE")
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.set_aspect("equal")
    axis.legend(loc="best", fontsize=8)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_estimate(
    output_path: Path,
    estimator: IncrementalRadiationGrid,
    source_position: tuple[float, float],
    steps: list[dict[str, object]],
) -> None:
    estimate = np.ma.masked_invalid(estimator.mean)
    colour_map = plt.get_cmap("inferno").copy()
    colour_map.set_bad(color="white", alpha=0.0)
    bounds = estimator.grid.bounds
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(
        estimate,
        origin="lower",
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        cmap=colour_map,
    )
    figure.colorbar(image, ax=axis, label=f"Estimated radiation ({estimator.unit})")
    axis.plot(
        [step["x_m"] for step in steps],
        [step["y_m"] for step in steps],
        color="deepskyblue",
        linewidth=1.5,
        alpha=0.8,
        label="Executed path",
    )
    axis.scatter(
        [source_position[0]],
        [source_position[1]],
        c="cyan",
        edgecolors="black",
        s=140,
        marker="*",
        label="Truth source (debug only)",
    )
    axis.set_title(
        "Online estimate — DEBUG NODE-SAMPLING MODE\n"
        "White/blank = unknown (never converted to zero)"
    )
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.set_aspect("equal")
    axis.legend(loc="best", fontsize=8)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run_debug_mission(
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict[str, object]:
    """Run one complete structural mission and save transparent debug outputs."""
    item = DatasetLoader(str(DATASET_DIRECTORY)).load_environment()
    probability_map, bounds, search_center, original_crop_mass = (
        _debug_probability_crop(item)
    )
    max_radius_m = float(item.radius_km) * 1_000.0
    detection_radius_m = UAV_ALTITUDE_M * np.tan(np.radians(FOV_DEG / 2.0))

    initial_route = generate_greedy_path(
        center_x=search_center[0],
        center_y=search_center[1],
        num_drones=1,
        probability_map=probability_map,
        bounds=bounds,
        max_radius=max_radius_m,
        fov_deg=FOV_DEG,
        altitude=UAV_ALTITUDE_M,
        budget=DEBUG_INITIAL_BUDGET_M,
    )[0]
    state = ExecutedSARState(
        map_shape=probability_map.shape,
        bounds=bounds,
        detection_radius_m=detection_radius_m,
    )
    normal_controller = PrecomputedRouteController(initial_route, state)

    source = PointSource(
        PointSourceConfig(
            source_id=DEBUG_POINT_SOURCE_ID,
            x_m=search_center[0],
            y_m=search_center[1],
            reference_excess_uSv_h=DEBUG_POINT_REFERENCE_EXCESS_USV_H,
            crs=item.projected_crs,
        )
    )
    truth = CompositeRadiationField(
        crs=item.projected_crs,
        point_sources=(source,),
    )
    sensor = NoiseFreeRadiationSensor(
        truth.query_total_dose_rate,
        quantity=QUANTITY,
        unit=UNIT,
    )
    radiation_grid = GridSpec.from_bounds(bounds, item.projected_crs)
    estimator = IncrementalRadiationGrid(
        radiation_grid,
        quantity=QUANTITY,
        unit=UNIT,
    )
    trigger = ConsecutiveRadiationTrigger(
        AboveBackgroundCriterion(
            truth.background_uSv_h,
            quantity=QUANTITY,
            unit=UNIT,
        )
    )
    observer = ExecutedNodeRadiationObserver(sensor, estimator, trigger)
    radiation_planner = RadiationGreedyStepPlanner(
        probability_map=probability_map,
        bounds=bounds,
        search_center=search_center,
        max_radius=max_radius_m,
        detection_radius_m=detection_radius_m,
        estimator=estimator,
    )
    controller = MissionModeController(
        state=state,
        normal_route_controller=normal_controller,
        observer=observer,
        radiation_planner=radiation_planner,
        initial_budget_m=DEBUG_INITIAL_BUDGET_M,
        probability_map=probability_map,
        bounds=bounds,
        search_center=search_center,
        max_radius=max_radius_m,
        fov_deg=FOV_DEG,
        altitude_m=UAV_ALTITUDE_M,
    )

    steps: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    while not controller.is_complete:
        result = controller.advance(
            simulated_time_s=0.0  # Deliberately unmodelled in DEBUG node mode.
        )
        events.append(_event_record(result))
        if result.moved:
            steps.append(_step_record(result, controller))

    if not steps:
        raise RuntimeError("DEBUG mission completed without executing any node.")

    normal_steps = sum(step["mode"] == MissionMode.NORMAL.value for step in steps)
    radiation_steps = sum(
        step["mode"] == MissionMode.RADIATION.value for step in steps
    )
    radiation_episodes = sum(event["normal_route_interrupted"] for event in events)
    continuation_replans = sum(event["normal_route_replanned"] for event in events)
    radiation_records = [
        step for step in steps if step["mode"] == MissionMode.RADIATION.value
    ]
    trigger_locations = [
        [step["x_m"], step["y_m"]]
        for step in steps
        if step["normal_route_interrupted"]
    ]

    output_directory.mkdir(parents=True, exist_ok=True)
    mission_steps_path = output_directory / "mission_steps.csv"
    mode_events_path = output_directory / "mode_events.csv"
    measurements_path = output_directory / "radiation_measurements.csv"
    estimate_path = output_directory / "estimated_radiation_grid.npz"
    trajectory_path = output_directory / "trajectory_debug.png"
    estimate_figure_path = output_directory / "estimated_radiation_debug.png"
    summary_path = output_directory / "debug_summary.json"

    _write_csv(mission_steps_path, steps)
    _write_csv(mode_events_path, events)
    _write_csv(
        measurements_path,
        [
            {
                "step_index": step["step_index"],
                "sampling_mode": step["sampling_mode"],
                "x_m": step["x_m"],
                "y_m": step["y_m"],
                "simulated_time_s": step["simulated_time_s"],
                "value": step["radiation_measurement"],
                "quantity": step["radiation_quantity"],
                "unit": step["radiation_unit"],
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
    )

    truth_field = (
        source.rasterize(radiation_grid, z_m=UAV_ALTITUDE_M)
        + truth.background_uSv_h
    )
    _plot_trajectory(
        trajectory_path,
        probability_map,
        bounds,
        truth_field,
        search_center,
        steps,
    )
    _plot_estimate(estimate_figure_path, estimator, search_center, steps)

    summary = {
        "status": "COMPLETE",
        "purpose": "DEBUG_STRUCTURAL_DEMONSTRATION_ONLY",
        "sampling_mode": DEBUG_SAMPLING_MODE,
        "dataset_directory": str(DATASET_DIRECTORY),
        "source_model": source.config.model_name,
        "source_position_m": list(search_center),
        "source_reference_excess_uSv_h": source.config.reference_excess_uSv_h,
        "background_uSv_h": truth.background_uSv_h,
        "debug_crop_shape": list(probability_map.shape),
        "debug_crop_original_probability_mass": original_crop_mass,
        "debug_crop_probability_contract": "conditional_sum_to_one",
        "initial_budget_m": DEBUG_INITIAL_BUDGET_M,
        "initial_normal_route_nodes": len(initial_route.coords),
        "initial_normal_route_invalidated": normal_controller.invalidated,
        "discarded_initial_normal_nodes": normal_controller.discarded_node_count,
        "normal_steps": int(normal_steps),
        "radiation_steps": int(radiation_steps),
        "radiation_episodes": int(radiation_episodes),
        "continuation_replans": int(continuation_replans),
        "trigger_locations_m": trigger_locations,
        "radiation_step_distances_m": sorted(
            {
                round(float(step["executed_distance_step_m"]), 9)
                for step in radiation_records
            }
        ),
        "minimum_radiation_step_sar_score": (
            min(float(step["sar_score"]) for step in radiation_records)
            if radiation_records
            else None
        ),
        "repeated_executed_positions": (
            len(steps)
            - len({(step["x_m"], step["y_m"]) for step in steps})
        ),
        "executed_distance_m": state.total_executed_distance_m,
        "remaining_budget_m": controller.remaining_budget_m,
        "mode_history": [mode.value for mode in controller.mode_history],
        "estimated_observed_cells": int(np.count_nonzero(estimator.observed_mask)),
        "estimated_total_cells": int(estimator.observed_mask.size),
        "formal_uav_speed_m_s": None,
        "formal_sampling_frequency_hz": None,
        "formal_sampling_spacing_m": None,
        "outputs": {
            "mission_steps_csv": str(mission_steps_path),
            "mode_events_csv": str(mode_events_path),
            "radiation_measurements_csv": str(measurements_path),
            "estimated_radiation_npz": str(estimate_path),
            "trajectory_png": str(trajectory_path),
            "estimated_radiation_png": str(estimate_figure_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run_debug_mission()
