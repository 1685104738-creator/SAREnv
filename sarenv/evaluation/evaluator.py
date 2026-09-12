"""Offline evaluator for completed SAREnv radiation-aware missions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
import numpy as np

from ..analytics.metrics import PathEvaluator
from ..core.loading import DatasetLoader, SARDatasetItem
from ..io.lost_person import LostPersonLocations, load_lost_person_locations
from .exposure import (
    DEFAULT_USV_CONTRACT,
    RadiationQuantityContract,
    RadiationTruthQuery,
    RouteIntegrationSamples,
    RouteRadiationIntegral,
    SurvivorRadiationRecord,
    dose_from_distance_integral,
    evaluate_survivor_radiation,
    integrate_radiation_along_route,
    summarise_survivor_radiation,
)
from .trajectory import (
    ExecutedTrajectory,
    SurvivorDiscovery,
    calculate_turn_diagnostics,
    discover_survivors,
    load_executed_trajectory,
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Explicit post-processing inputs independent of planner runtime."""

    trajectory_path: Path
    dataset_directory: Path
    survivors_path: Path
    scenario_parameters_path: Path
    output_directory: Path
    platform_altitude_m: float
    survivor_reference_height_m: float
    fov_deg: float
    background_uSv_h: float | None = None
    background_rate: float | None = None
    radiation_quantity: str = DEFAULT_USV_CONTRACT.quantity
    radiation_rate_unit: str = DEFAULT_USV_CONTRACT.rate_unit
    radiation_dose_unit: str = DEFAULT_USV_CONTRACT.dose_unit
    radiation_integration_step_m: float = 1.0
    evaluation_speed_m_s: float | None = None
    speed_status: str = "not_provided"
    sar_discount_factor: float = 0.999
    write_detail_outputs: bool = True
    write_cumulative_metrics: bool = True
    generate_figures: bool = True

    def __post_init__(self) -> None:
        numeric_positive = {
            "platform_altitude_m": self.platform_altitude_m,
            "fov_deg": self.fov_deg,
            "radiation_integration_step_m": self.radiation_integration_step_m,
        }
        for name, value in numeric_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.survivor_reference_height_m) or (
            self.survivor_reference_height_m < 0.0
        ):
            raise ValueError(
                "survivor_reference_height_m must be finite and non-negative."
            )
        _ = self.resolved_background_rate
        _ = self.radiation_contract
        if self.evaluation_speed_m_s is not None and (
            not math.isfinite(self.evaluation_speed_m_s)
            or self.evaluation_speed_m_s <= 0.0
        ):
            raise ValueError("evaluation_speed_m_s must be finite and positive.")
        if not 0.0 < self.sar_discount_factor <= 1.0:
            raise ValueError("sar_discount_factor must lie in (0, 1].")
        for name in (
            "write_detail_outputs",
            "write_cumulative_metrics",
            "generate_figures",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")

    @property
    def detection_radius_m(self) -> float:
        return float(
            self.platform_altitude_m * np.tan(np.radians(self.fov_deg / 2.0))
        )

    @property
    def resolved_background_rate(self) -> float:
        if self.background_rate is None:
            if self.background_uSv_h is None:
                raise ValueError("A background radiation rate must be provided.")
            if (
                not math.isfinite(self.background_uSv_h)
                or self.background_uSv_h < 0.0
            ):
                raise ValueError(
                    "background_uSv_h must be finite and non-negative."
                )
            return float(self.background_uSv_h)
        if self.background_uSv_h is not None and not math.isclose(
            self.background_rate,
            self.background_uSv_h,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("Generic and legacy background values disagree.")
        if not math.isfinite(self.background_rate) or self.background_rate < 0.0:
            raise ValueError("background_rate must be finite and non-negative.")
        return float(self.background_rate)

    @property
    def radiation_contract(self) -> RadiationQuantityContract:
        return RadiationQuantityContract(
            quantity=self.radiation_quantity,
            rate_unit=self.radiation_rate_unit,
            dose_unit=self.radiation_dose_unit,
        )


@dataclass(frozen=True)
class PostRunEvaluationResult:
    """In-memory summaries and output paths from one evaluation."""

    evaluation_summary: dict[str, object]
    sar_metrics: dict[str, object]
    uav_radiation_summary: dict[str, object]
    survivor_radiation_summary: dict[str, object]
    output_paths: dict[str, Path]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError(f"Cannot write empty evaluation output: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def evaluate_native_sar_metrics(
    trajectory: ExecutedTrajectory,
    item: SARDatasetItem,
    survivors: LostPersonLocations,
    *,
    fov_deg: float,
    altitude_m: float,
    discount_factor: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Call the production SAREnv ``PathEvaluator`` without redefining scores."""
    victims = gpd.GeoDataFrame(
        {"survivor_id": np.arange(len(survivors.points), dtype=int)},
        geometry=list(survivors.points),
        crs=survivors.projected_crs,
    )
    evaluator = PathEvaluator(
        np.asarray(item.heatmap, dtype=float),
        tuple(float(value) for value in item.bounds),
        victims,
        float(fov_deg),
        float(altitude_m),
        float(item.meter_per_bin),
    )
    raw = evaluator.calculate_all_metrics([trajectory.line], discount_factor)

    interpolation_resolution = evaluator.interpolation_resolution
    if trajectory.total_distance_m > 0.0:
        count = int(np.ceil(trajectory.total_distance_m / interpolation_resolution)) + 1
        sample_distances = np.linspace(0.0, trajectory.total_distance_m, count)
    else:
        sample_distances = np.asarray([0.0])
    observed_cells: set[tuple[int, int]] = set()
    for distance_m in sample_distances:
        point = trajectory.interpolate(float(distance_m))
        observed_cells.update(evaluator.get_visible_cells(point.x, point.y))

    victim_metrics = raw["victim_detection_metrics"]
    cumulative_distances = np.asarray(raw["cumulative_distances"][0], dtype=float)
    cumulative_probability = np.asarray(
        raw["cumulative_likelihoods"][0],
        dtype=float,
    )
    heatmap_sum = float(np.asarray(item.heatmap, dtype=float).sum())
    metrics = {
        "implementation": "sarenv.analytics.metrics.PathEvaluator",
        "metric_definition_status": "reused_original_sarenv_production_implementation",
        "discount_factor": float(discount_factor),
        "detection_radius_m": float(evaluator.detection_radius),
        "native_path_sampling_resolution_m": interpolation_resolution,
        "searched_observed_sar_cell_count": len(observed_cells),
        "total_sar_cell_count": int(np.asarray(item.heatmap).size),
        "total_likelihood_score": float(raw["total_likelihood_score"]),
        "probability_coverage_fraction": (
            float(raw["total_likelihood_score"]) / heatmap_sum
            if heatmap_sum > 0.0
            else 0.0
        ),
        "total_time_discounted_score": float(raw["total_time_discounted_score"]),
        "victim_detection_percentage": float(victim_metrics["percentage_found"]),
        "found_victim_indices": [
            int(value) for value in victim_metrics["found_victim_indices"]
        ],
        "area_covered_km2": float(raw["area_covered"]),
        "total_path_length_km": float(raw["total_path_length"]),
    }
    return metrics, cumulative_distances, cumulative_probability


class PostRunEvaluator:
    """Evaluate saved route outputs without accessing planner state or estimates."""

    def __init__(
        self,
        config: EvaluationConfig,
        *,
        excess_truth_query: RadiationTruthQuery,
        source_position_m: tuple[float, float] | None = None,
        source_positions_m: tuple[tuple[float, float], ...] | None = None,
        route_integration_samples: RouteIntegrationSamples | None = None,
    ) -> None:
        if not isinstance(config, EvaluationConfig):
            raise TypeError("config must be an EvaluationConfig.")
        if not callable(excess_truth_query):
            raise TypeError("excess_truth_query must be callable.")
        self.config = config
        self.excess_truth_query = excess_truth_query
        self.route_integration_samples = route_integration_samples
        if source_positions_m is not None and source_position_m is not None:
            raise ValueError("Use source_position_m or source_positions_m, not both.")
        self.source_positions_m = (
            tuple(source_positions_m)
            if source_positions_m is not None
            else (() if source_position_m is None else (source_position_m,))
        )

    def evaluate(self) -> PostRunEvaluationResult:
        """Run SAR, UAV-radiation and survivor-radiation post-processing."""
        config = self.config
        trajectory = load_executed_trajectory(config.trajectory_path)
        item = DatasetLoader(str(config.dataset_directory)).load_environment()
        survivors = load_lost_person_locations(config.survivors_path)
        scenario_parameters = json.loads(
            config.scenario_parameters_path.read_text(encoding="utf-8")
        )
        self._validate_contract(trajectory, item, survivors, scenario_parameters)

        discoveries = discover_survivors(
            trajectory,
            survivors.points,
            config.detection_radius_m,
        )
        sar_metrics, sar_curve_distances, sar_curve_probability = (
            evaluate_native_sar_metrics(
                trajectory,
                item,
                survivors,
                fov_deg=config.fov_deg,
                altitude_m=config.platform_altitude_m,
                discount_factor=config.sar_discount_factor,
            )
        )
        continuous_found = sum(discovery.found for discovery in discoveries)
        sar_metrics["continuous_trajectory_discovery"] = {
            "method": "exact_first_circle_entry_along_piecewise_linear_route",
            "found_count": continuous_found,
            "not_found_count": len(discoveries) - continuous_found,
            "found_fraction": (
                continuous_found / len(discoveries) if discoveries else 0.0
            ),
        }

        uav_integral = integrate_radiation_along_route(
            trajectory,
            self.excess_truth_query,
            query_height_m=config.platform_altitude_m,
            background_rate=config.resolved_background_rate,
            contract=config.radiation_contract,
            integration_step_m=config.radiation_integration_step_m,
            route_samples=self.route_integration_samples,
        )
        uav_summary = self._uav_summary(uav_integral)
        survivor_records = evaluate_survivor_radiation(
            trajectory,
            survivors.points,
            discoveries,
            self.excess_truth_query,
            query_height_m=config.survivor_reference_height_m,
            background_rate=config.resolved_background_rate,
            contract=config.radiation_contract,
            evaluation_speed_m_s=config.evaluation_speed_m_s,
        )
        survivor_summary = summarise_survivor_radiation(survivor_records)
        turn_diagnostics = calculate_turn_diagnostics(trajectory)

        output_directory = config.output_directory.resolve()
        if (
            config.write_detail_outputs
            or config.write_cumulative_metrics
            or config.generate_figures
        ):
            output_directory.mkdir(parents=True, exist_ok=True)
        paths = self._output_paths(output_directory)
        parameters = self._evaluation_parameters(
            trajectory,
            item,
            survivors,
            scenario_parameters,
        )
        evaluation_summary = {
            "scenario_id": scenario_parameters.get("scenario_id"),
            "sar": sar_metrics,
            "uav_radiation": uav_summary,
            "survivor_radiation": survivor_summary,
            "trajectory": {
                "node_count": trajectory.node_count,
                "segment_count": trajectory.segment_count,
                "total_distance_m": trajectory.total_distance_m,
                "turn_diagnostics": turn_diagnostics,
            },
            "evaluation_parameters": parameters,
        }

        if config.write_detail_outputs:
            _write_json(paths["sar_metrics_json"], sar_metrics)
            _write_json(paths["uav_radiation_summary_json"], uav_summary)
            _write_json(paths["survivor_radiation_summary_json"], survivor_summary)
            _write_json(paths["evaluation_parameters_json"], parameters)
            _write_json(paths["evaluation_summary_json"], evaluation_summary)
            _write_csv(
                paths["survivor_radiation_csv"],
                [record.to_dict() for record in survivor_records],
            )
        if config.write_cumulative_metrics:
            self._write_cumulative_curves(
                paths["cumulative_metrics_csv"],
                uav_integral,
                discoveries,
                sar_curve_distances,
                sar_curve_probability,
            )
        if config.generate_figures:
            truth_grid = self._truth_visualisation_grid(
                tuple(float(value) for value in item.bounds),
                config.survivor_reference_height_m,
            )
            self._plot_evaluation_overview(
                paths["evaluation_trajectory_overview_png"],
                trajectory,
                item,
                survivors,
                discoveries,
            )
            self._plot_uav_radiation_route(
                paths["uav_route_radiation_truth_png"],
                trajectory,
                item,
                uav_integral,
                truth_grid,
            )
            self._plot_survivor_dose(
                paths["survivor_pre_discovery_radiation_png"],
                item,
                survivor_records,
                truth_grid,
            )
            self._plot_survivor_discovery(
                paths["survivor_discovery_distance_png"],
                item,
                survivor_records,
                truth_grid,
            )
            self._plot_cumulative_curves(
                paths["cumulative_curves_png"],
                uav_integral,
                discoveries,
                sar_curve_distances,
                sar_curve_probability,
            )
        return PostRunEvaluationResult(
            evaluation_summary=evaluation_summary,
            sar_metrics=sar_metrics,
            uav_radiation_summary=uav_summary,
            survivor_radiation_summary=survivor_summary,
            output_paths=paths,
        )

    def _validate_contract(
        self,
        trajectory: ExecutedTrajectory,
        item: SARDatasetItem,
        survivors: LostPersonLocations,
        scenario_parameters: dict[str, object],
    ) -> None:
        if survivors.environment_size != item.size:
            raise ValueError("Survivors and SAR dataset sizes do not match.")
        if tuple(survivors.bounds) != tuple(item.bounds):
            raise ValueError("Survivors and SAR dataset bounds do not match.")
        if str(survivors.projected_crs) != str(item.projected_crs):
            raise ValueError("Survivors and SAR dataset CRS do not match.")
        expected_bounds = tuple(
            float(value) for value in scenario_parameters["dataset_bounds_projected"]
        )
        if not np.allclose(expected_bounds, item.bounds, rtol=0.0, atol=1e-7):
            raise ValueError("Scenario and SAR dataset bounds do not match.")
        if not math.isclose(
            float(scenario_parameters["platform_altitude_m"]),
            self.config.platform_altitude_m,
        ):
            raise ValueError("Evaluation platform altitude conflicts with scenario.")
        if not math.isclose(
            float(scenario_parameters["survivor_reference_height_m"]),
            self.config.survivor_reference_height_m,
        ):
            raise ValueError("Evaluation ground reference height conflicts with scenario.")
        if not math.isclose(
            float(scenario_parameters["uav_measurement_height_m"]),
            self.config.platform_altitude_m,
        ):
            raise ValueError("UAV measurement height conflicts with scenario.")
        minx, miny, maxx, maxy = item.bounds
        if not all(
            minx <= node.x_m <= maxx and miny <= node.y_m <= maxy
            for node in trajectory.nodes
        ):
            raise ValueError("Executed trajectory leaves the saved dataset bounds.")

    def _uav_summary(self, integral: RouteRadiationIntegral) -> dict[str, object]:
        speed = self.config.evaluation_speed_m_s
        contract = integral.contract
        return {
            "truth_source": "post_run_ground_truth_query_not_planner_estimator",
            "truth_query_height_m": integral.query_height_m,
            "radiation_quantity": contract.quantity,
            "rate_unit": contract.rate_unit,
            "dose_unit": contract.dose_unit,
            "distance_integral_unit": contract.distance_integral_unit,
            "integration_method": "piecewise_linear_route_fixed_distance_trapezoid",
            "integration_step_m": integral.integration_step_m,
            "sample_count": int(integral.distances_m.size),
            "sampled_excess_rate_min": float(
                integral.excess_rates_uSv_h.min()
            ),
            "sampled_excess_rate_max": float(
                integral.excess_rates_uSv_h.max()
            ),
            "sampled_total_rate_min": float(integral.total_rates_uSv_h.min()),
            "sampled_total_rate_max": float(integral.total_rates_uSv_h.max()),
            "uav_excess_exposure_distance_integral": (
                integral.excess_distance_integral_uSv_h_m
            ),
            "uav_total_exposure_distance_integral": (
                integral.total_distance_integral_uSv_h_m
            ),
            "evaluation_speed_m_s": speed,
            "speed_status": self.config.speed_status,
            "uav_cumulative_excess_dose": dose_from_distance_integral(
                integral.excess_distance_integral_uSv_h_m,
                speed,
            ),
            "uav_cumulative_total_dose": dose_from_distance_integral(
                integral.total_distance_integral_uSv_h_m,
                speed,
            ),
        }

    def _evaluation_parameters(
        self,
        trajectory: ExecutedTrajectory,
        item: SARDatasetItem,
        survivors: LostPersonLocations,
        scenario_parameters: dict[str, object],
    ) -> dict[str, object]:
        config = self.config
        return {
            "schema_version": 1,
            "evaluator": "sarenv.evaluation.PostRunEvaluator",
            "source_trajectory_file": str(config.trajectory_path.resolve()),
            "scenario_id": scenario_parameters.get("scenario_id"),
            "dataset_directory": str(config.dataset_directory.resolve()),
            "survivors_file": str(config.survivors_path.resolve()),
            "scenario_parameters_file": str(config.scenario_parameters_path.resolve()),
            "platform_altitude_m": config.platform_altitude_m,
            "survivor_value_reference_height_m": config.survivor_reference_height_m,
            "uav_truth_query_height_m": config.platform_altitude_m,
            "survivor_truth_query_height_m": config.survivor_reference_height_m,
            "fov_deg": config.fov_deg,
            "detection_radius_m": config.detection_radius_m,
            "sar_resolution_m": float(item.meter_per_bin),
            "radiation_resolution_m": scenario_parameters.get(
                "radiation_resolution_m"
            ),
            "radiation_integration_step_m": config.radiation_integration_step_m,
            "evaluation_speed_m_s": config.evaluation_speed_m_s,
            "speed_status": config.speed_status,
            "background_rate": config.resolved_background_rate,
            "radiation_quantity": config.radiation_quantity,
            "radiation_rate_unit": config.radiation_rate_unit,
            "radiation_dose_unit": config.radiation_dose_unit,
            "mission_route_distance_m": trajectory.total_distance_m,
            "trajectory_node_count": trajectory.node_count,
            "trajectory_segment_count": trajectory.segment_count,
            "survivor_count": len(survivors.points),
            "sar_discount_factor": config.sar_discount_factor,
            "constant_speed_model": True,
            "acceleration_modelled": False,
            "turn_time_penalty_modelled": False,
            "planner_estimator_used_for_truth_metrics": False,
            "input_sha256": {
                "trajectory": _sha256(config.trajectory_path),
                "heatmap": _sha256(config.dataset_directory / "heatmap.npy"),
                "survivors": _sha256(config.survivors_path),
                "scenario_parameters": _sha256(config.scenario_parameters_path),
            },
        }

    @staticmethod
    def _output_paths(output_directory: Path) -> dict[str, Path]:
        return {
            "evaluation_summary_json": output_directory / "evaluation_summary.json",
            "sar_metrics_json": output_directory / "sar_metrics.json",
            "uav_radiation_summary_json": output_directory / "uav_radiation_summary.json",
            "survivor_radiation_csv": output_directory / "survivor_radiation.csv",
            "survivor_radiation_summary_json": output_directory / "survivor_radiation_summary.json",
            "evaluation_parameters_json": output_directory / "evaluation_parameters.json",
            "cumulative_metrics_csv": output_directory / "cumulative_metrics.csv",
            "evaluation_trajectory_overview_png": output_directory / "evaluation_trajectory_overview.png",
            "uav_route_radiation_truth_png": output_directory / "uav_route_radiation_truth.png",
            "survivor_pre_discovery_radiation_png": output_directory / "survivor_pre_discovery_radiation.png",
            "survivor_discovery_distance_png": output_directory / "survivor_discovery_distance.png",
            "cumulative_curves_png": output_directory / "cumulative_curves.png",
        }

    def _write_cumulative_curves(
        self,
        path: Path,
        integral: RouteRadiationIntegral,
        discoveries: tuple[SurvivorDiscovery, ...],
        sar_distances: np.ndarray,
        sar_probability: np.ndarray,
    ) -> None:
        found_distances = np.sort(
            np.asarray(
                [
                    discovery.first_discovery_distance_m
                    for discovery in discoveries
                    if discovery.found
                ],
                dtype=float,
            )
        )
        probability = np.interp(
            integral.distances_m,
            sar_distances,
            sar_probability,
        )
        found_counts = np.searchsorted(
            found_distances,
            integral.distances_m,
            side="right",
        )
        speed = self.config.evaluation_speed_m_s
        records = []
        for index, distance_m in enumerate(integral.distances_m):
            records.append(
                {
                    "path_distance_m": float(distance_m),
                    "time_s": (
                        float(distance_m / speed) if speed is not None else None
                    ),
                    "cumulative_sar_probability": float(probability[index]),
                    "cumulative_survivors_found": int(found_counts[index]),
                    "cumulative_uav_excess_distance_integral": float(
                        integral.cumulative_excess_integral_uSv_h_m[index]
                    ),
                    "cumulative_uav_total_distance_integral": float(
                        integral.cumulative_total_integral_uSv_h_m[index]
                    ),
                    "cumulative_uav_excess_dose": (
                        dose_from_distance_integral(
                            float(integral.cumulative_excess_integral_uSv_h_m[index]),
                            speed,
                        )
                    ),
                    "cumulative_uav_total_dose": (
                        dose_from_distance_integral(
                            float(integral.cumulative_total_integral_uSv_h_m[index]),
                            speed,
                        )
                    ),
                    "radiation_distance_integral_unit": (
                        integral.contract.distance_integral_unit
                    ),
                    "radiation_dose_unit": integral.contract.dose_unit,
                }
            )
        _write_csv(path, records)

    def _truth_visualisation_grid(
        self,
        bounds: tuple[float, float, float, float],
        query_height_m: float,
    ) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
        minx, miny, maxx, maxy = bounds
        visual_resolution = max(1.0, math.ceil(max(maxx - minx, maxy - miny) / 500.0))
        x_values = np.arange(minx + visual_resolution / 2.0, maxx, visual_resolution)
        y_values = np.arange(miny + visual_resolution / 2.0, maxy, visual_resolution)
        field = np.empty((y_values.size, x_values.size), dtype=float)
        for row, y_m in enumerate(y_values):
            for col, x_m in enumerate(x_values):
                field[row, col] = self.config.resolved_background_rate + float(
                    self.excess_truth_query(x_m, y_m, query_height_m)
                )
        return field, bounds, float(visual_resolution)

    @staticmethod
    def _route_segments(trajectory: ExecutedTrajectory) -> np.ndarray:
        coordinates = np.asarray(trajectory.coordinates, dtype=float)
        return np.stack((coordinates[:-1], coordinates[1:]), axis=1)

    def _plot_evaluation_overview(
        self,
        path: Path,
        trajectory: ExecutedTrajectory,
        item: SARDatasetItem,
        survivors: LostPersonLocations,
        discoveries: tuple[SurvivorDiscovery, ...],
    ) -> None:
        figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
        image = axis.imshow(
            item.heatmap,
            origin="lower",
            extent=(item.bounds[0], item.bounds[2], item.bounds[1], item.bounds[3]),
            cmap="YlOrRd",
            interpolation="nearest",
        )
        figure.colorbar(image, ax=axis, label="Lost-person probability per SAR cell")
        segments = self._route_segments(trajectory)
        segment_modes = [node.mode for node in trajectory.nodes[1:]]
        for mode, colour in (("NORMAL", "limegreen"), ("RADIATION", "deepskyblue")):
            selected = segments[np.asarray([value == mode for value in segment_modes])]
            if selected.size:
                axis.add_collection(
                    LineCollection(selected, colors=colour, linewidths=1.0, label=mode)
                )
        found_points = [
            survivors.points[result.survivor_id]
            for result in discoveries
            if result.found
        ]
        not_found_points = [
            survivors.points[result.survivor_id]
            for result in discoveries
            if not result.found
        ]
        if found_points:
            axis.scatter(
                [point.x for point in found_points],
                [point.y for point in found_points],
                c="mediumseagreen",
                edgecolors="black",
                s=28,
                label="Found survivor",
                zorder=5,
            )
        if not_found_points:
            axis.scatter(
                [point.x for point in not_found_points],
                [point.y for point in not_found_points],
                c="crimson",
                marker="X",
                s=42,
                label="Not found",
                zorder=6,
            )
        start = trajectory.nodes[0]
        axis.scatter([start.x_m], [start.y_m], c="white", edgecolors="black", s=75, label="Start", zorder=7)
        self._plot_source_positions(axis, size=160)
        axis.set_title("Post-run SAR and trajectory evaluation overview")
        self._format_map_axis(axis, item.bounds)
        axis.legend(fontsize=8, loc="upper right")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _plot_uav_radiation_route(
        self,
        path: Path,
        trajectory: ExecutedTrajectory,
        item: SARDatasetItem,
        integral: RouteRadiationIntegral,
        truth_grid: tuple[np.ndarray, tuple[float, float, float, float], float],
    ) -> None:
        field, bounds, _ = truth_grid
        figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
        norm = self._positive_log_norm(field)
        image = axis.imshow(field, origin="lower", extent=(bounds[0], bounds[2], bounds[1], bounds[3]), cmap="magma", norm=norm)
        figure.colorbar(image, ax=axis, label=f"Ground truth total rate at 1 m ({self.config.radiation_rate_unit})")
        segments = self._route_segments(trajectory)
        midpoint_distances = np.asarray(
            [
                0.5 * (start.cumulative_distance_m + end.cumulative_distance_m)
                for start, end in zip(trajectory.nodes, trajectory.nodes[1:])
            ]
        )
        midpoint_rates = np.interp(midpoint_distances, integral.distances_m, integral.total_rates_uSv_h)
        route_norm = Normalize(vmin=float(midpoint_rates.min()), vmax=float(midpoint_rates.max()))
        collection = LineCollection(segments, cmap="viridis", norm=route_norm, linewidths=1.4)
        collection.set_array(midpoint_rates)
        axis.add_collection(collection)
        figure.colorbar(collection, ax=axis, label=f"Local UAV total rate at 50 m ({self.config.radiation_rate_unit})")
        self._plot_source_positions(axis, size=150)
        axis.set_title("UAV route coloured by platform-altitude radiation truth")
        self._format_map_axis(axis, item.bounds)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _plot_survivor_dose(
        self,
        path: Path,
        item: SARDatasetItem,
        records: tuple[SurvivorRadiationRecord, ...],
        truth_grid: tuple[np.ndarray, tuple[float, float, float, float], float],
    ) -> None:
        field, bounds, _ = truth_grid
        figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
        axis.imshow(field, origin="lower", extent=(bounds[0], bounds[2], bounds[1], bounds[3]), cmap="Greys", norm=self._positive_log_norm(field), alpha=0.72)
        doses = np.asarray([record.pre_discovery_total_dose_uSv for record in records], dtype=float)
        dose_norm = Normalize(vmin=0.0, vmax=max(float(doses.max()), 1e-12))
        found = [record for record in records if record.found]
        not_found = [record for record in records if not record.found]
        scatter = None
        if found:
            scatter = axis.scatter([record.x_m for record in found], [record.y_m for record in found], c=[record.pre_discovery_total_dose_uSv for record in found], cmap="plasma", norm=dose_norm, marker="o", edgecolors="black", s=45, label="Found")
        if not_found:
            scatter = axis.scatter([record.x_m for record in not_found], [record.y_m for record in not_found], c=[record.pre_discovery_total_dose_uSv for record in not_found], cmap="plasma", norm=dose_norm, marker="X", edgecolors="white", linewidths=0.8, s=70, label="Not found; censored at mission end")
        if scatter is not None:
            figure.colorbar(scatter, ax=axis, label=f"Pre-discovery total dose ({self.config.radiation_dose_unit})")
        axis.set_title("Survivor pre-discovery radiation dose")
        self._format_map_axis(axis, item.bounds)
        axis.legend(fontsize=8)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _plot_survivor_discovery(
        self,
        path: Path,
        item: SARDatasetItem,
        records: tuple[SurvivorRadiationRecord, ...],
        truth_grid: tuple[np.ndarray, tuple[float, float, float, float], float],
    ) -> None:
        field, bounds, _ = truth_grid
        figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
        axis.imshow(field, origin="lower", extent=(bounds[0], bounds[2], bounds[1], bounds[3]), cmap="Greys", alpha=0.42)
        found = [record for record in records if record.found]
        not_found = [record for record in records if not record.found]
        scatter = None
        if found:
            scatter = axis.scatter([record.x_m for record in found], [record.y_m for record in found], c=[record.first_discovery_distance_m for record in found], cmap="viridis", marker="o", edgecolors="black", s=45, label="Found")
        if not_found:
            axis.scatter([record.x_m for record in not_found], [record.y_m for record in not_found], c="red", marker="X", s=70, label="Not found")
        if scatter is not None:
            figure.colorbar(scatter, ax=axis, label="First discovery route distance (m)")
        axis.set_title("Continuous-route survivor discovery distance")
        self._format_map_axis(axis, item.bounds)
        axis.legend(fontsize=8)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def _plot_cumulative_curves(
        self,
        path: Path,
        integral: RouteRadiationIntegral,
        discoveries: tuple[SurvivorDiscovery, ...],
        sar_distances: np.ndarray,
        sar_probability: np.ndarray,
    ) -> None:
        figure, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
        axes[0].plot(sar_distances, sar_probability, color="darkorange")
        axes[0].set_ylabel("Cumulative SAR probability")
        found_distances = np.sort([discovery.first_discovery_distance_m for discovery in discoveries if discovery.found])
        if found_distances.size:
            axes[1].step(found_distances, np.arange(1, found_distances.size + 1), where="post", color="seagreen")
        axes[1].set_ylabel("Survivors found")
        speed = self.config.evaluation_speed_m_s
        if speed is None:
            axes[2].plot(integral.distances_m, integral.cumulative_total_integral_uSv_h_m, label="Total", color="purple")
            axes[2].plot(integral.distances_m, integral.cumulative_excess_integral_uSv_h_m, label="Excess", color="crimson")
            axes[2].set_ylabel(
                "UAV exposure integral\n"
                f"({integral.contract.distance_integral_unit})"
            )
        else:
            denominator = speed * 3600.0
            axes[2].plot(integral.distances_m, integral.cumulative_total_integral_uSv_h_m / denominator, label="Total", color="purple")
            axes[2].plot(integral.distances_m, integral.cumulative_excess_integral_uSv_h_m / denominator, label="Excess", color="crimson")
            axes[2].set_ylabel(
                f"Cumulative UAV dose ({integral.contract.dose_unit})"
            )
        axes[2].legend()
        axes[2].set_xlabel("Executed route distance (m)")
        figure.suptitle("Post-run cumulative mission metrics")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _format_map_axis(axis, bounds) -> None:
        axis.set_xlim(bounds[0], bounds[2])
        axis.set_ylim(bounds[1], bounds[3])
        axis.set_aspect("equal")
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")

    def _plot_source_positions(self, axis, *, size: float) -> None:
        if not self.source_positions_m:
            return
        axis.scatter(
            [position[0] for position in self.source_positions_m],
            [position[1] for position in self.source_positions_m],
            marker="*",
            c="yellow",
            edgecolors="black",
            s=size,
            label="Source truth (evaluation only)",
            zorder=8,
        )

    @staticmethod
    def _positive_log_norm(field: np.ndarray) -> LogNorm:
        positive = np.asarray(field, dtype=float)
        positive = positive[positive > 0.0]
        if positive.size == 0:
            return LogNorm(vmin=1e-12, vmax=1.0)
        lower = max(float(positive.min()), 1e-12)
        upper = max(float(positive.max()), lower * (1.0 + 1e-12))
        return LogNorm(vmin=lower, vmax=upper)


__all__ = [
    "EvaluationConfig",
    "PostRunEvaluationResult",
    "PostRunEvaluator",
    "evaluate_native_sar_metrics",
]
