"""Speed-independent and constant-speed radiation exposure calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Callable

import numpy as np
from shapely.geometry import Point

from .trajectory import ExecutedTrajectory, SurvivorDiscovery


RadiationTruthQuery = Callable[[float, float, float], float]


@dataclass(frozen=True)
class RadiationQuantityContract:
    """Explicit quantity/unit semantics for one evaluation truth field."""

    quantity: str
    rate_unit: str
    dose_unit: str

    def __post_init__(self) -> None:
        if not self.quantity or not self.rate_unit or not self.dose_unit:
            raise ValueError("Radiation quantity and units must be non-empty.")

    @property
    def distance_integral_unit(self) -> str:
        return f"{self.rate_unit}*m"


DEFAULT_USV_CONTRACT = RadiationQuantityContract(
    quantity="synthetic_gamma_dose_rate",
    rate_unit="uSv/h",
    dose_unit="uSv",
)


@dataclass(frozen=True)
class RouteRadiationIntegral:
    """Radiation sampled along a route and integrated against path distance."""

    distances_m: np.ndarray
    excess_rates_uSv_h: np.ndarray
    total_rates_uSv_h: np.ndarray
    cumulative_excess_integral_uSv_h_m: np.ndarray
    cumulative_total_integral_uSv_h_m: np.ndarray
    query_height_m: float
    integration_step_m: float
    contract: RadiationQuantityContract = DEFAULT_USV_CONTRACT

    @property
    def excess_distance_integral_uSv_h_m(self) -> float:
        return float(self.cumulative_excess_integral_uSv_h_m[-1])

    @property
    def total_distance_integral_uSv_h_m(self) -> float:
        return float(self.cumulative_total_integral_uSv_h_m[-1])


@dataclass(frozen=True)
class RouteIntegrationSamples:
    """One reusable in-memory resampling of a trajectory for route integration."""

    distances_m: np.ndarray
    sampled_xy_m: np.ndarray
    trajectory_coordinate_sha256: str
    trajectory_total_distance_m: float
    integration_step_m: float

    def __post_init__(self) -> None:
        distances = np.asarray(self.distances_m, dtype=float)
        sampled_xy = np.asarray(self.sampled_xy_m, dtype=float)
        if distances.ndim != 1:
            raise ValueError("distances_m must be one-dimensional.")
        if sampled_xy.shape != (distances.size, 2):
            raise ValueError("sampled_xy_m must have shape (len(distances_m), 2).")
        if not np.isfinite(distances).all() or not np.isfinite(sampled_xy).all():
            raise ValueError("Route integration samples must be finite.")
        if not self.trajectory_coordinate_sha256:
            raise ValueError("trajectory_coordinate_sha256 must be non-empty.")
        if (
            not math.isfinite(self.trajectory_total_distance_m)
            or self.trajectory_total_distance_m < 0.0
        ):
            raise ValueError("trajectory_total_distance_m must be finite and non-negative.")
        if not math.isfinite(self.integration_step_m) or self.integration_step_m <= 0.0:
            raise ValueError("integration_step_m must be finite and positive.")

        # The cache is computational state, not an output.  Make accidental
        # mutation fail instead of silently changing later scenario evaluations.
        distances.setflags(write=False)
        sampled_xy.setflags(write=False)
        object.__setattr__(self, "distances_m", distances)
        object.__setattr__(self, "sampled_xy_m", sampled_xy)


def _trajectory_coordinate_sha256(trajectory: ExecutedTrajectory) -> str:
    coordinates = np.ascontiguousarray(trajectory.coordinates, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.asarray(coordinates.shape, dtype=np.int64).tobytes())
    digest.update(coordinates.tobytes())
    return digest.hexdigest()


def prepare_route_integration_samples(
    trajectory: ExecutedTrajectory,
    *,
    integration_step_m: float = 1.0,
) -> RouteIntegrationSamples:
    """Expand one route once at the evaluator's existing sample distances."""
    distances = _sample_distances(trajectory.total_distance_m, integration_step_m)
    return RouteIntegrationSamples(
        distances_m=distances,
        sampled_xy_m=trajectory.sample_coordinates(distances),
        trajectory_coordinate_sha256=_trajectory_coordinate_sha256(trajectory),
        trajectory_total_distance_m=trajectory.total_distance_m,
        integration_step_m=float(integration_step_m),
    )


def _validate_route_integration_samples(
    samples: RouteIntegrationSamples,
    trajectory: ExecutedTrajectory,
    integration_step_m: float,
) -> None:
    if not isinstance(samples, RouteIntegrationSamples):
        raise TypeError("route_samples must be a RouteIntegrationSamples instance.")
    if samples.integration_step_m != float(integration_step_m):
        raise ValueError("Cached route samples use a different integration step.")
    if samples.trajectory_total_distance_m != trajectory.total_distance_m:
        raise ValueError("Cached route samples use a different trajectory distance.")
    if samples.trajectory_coordinate_sha256 != _trajectory_coordinate_sha256(trajectory):
        raise ValueError("Cached route samples belong to a different trajectory.")
    expected_distances = _sample_distances(
        trajectory.total_distance_m,
        integration_step_m,
    )
    if not np.array_equal(samples.distances_m, expected_distances):
        raise ValueError("Cached route sample distances do not match _sample_distances().")


@dataclass(frozen=True)
class SurvivorRadiationRecord:
    """One survivor's static truth and pre-discovery exposure."""

    survivor_id: int
    x_m: float
    y_m: float
    found: bool
    first_discovery_distance_m: float | None
    first_discovery_fraction_of_mission: float | None
    exposure_end_distance_m: float
    ground_truth_excess_rate_uSv_h: float
    ground_truth_total_rate_uSv_h: float
    excess_exposure_distance_integral_uSv_h_m: float
    total_exposure_distance_integral_uSv_h_m: float
    first_discovery_time_s: float | None
    exposure_end_time_s: float | None
    pre_discovery_excess_dose_uSv: float | None
    pre_discovery_total_dose_uSv: float | None
    contract: RadiationQuantityContract = DEFAULT_USV_CONTRACT

    def to_dict(self) -> dict[str, object]:
        return {
            "survivor_id": self.survivor_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "found": self.found,
            "first_discovery_distance_m": self.first_discovery_distance_m,
            "first_discovery_fraction_of_mission": (
                self.first_discovery_fraction_of_mission
            ),
            "exposure_end_distance_m": self.exposure_end_distance_m,
            "radiation_quantity": self.contract.quantity,
            "radiation_rate_unit": self.contract.rate_unit,
            "radiation_dose_unit": self.contract.dose_unit,
            "ground_truth_excess_rate": self.ground_truth_excess_rate_uSv_h,
            "ground_truth_total_rate": self.ground_truth_total_rate_uSv_h,
            "excess_exposure_distance_integral": (
                self.excess_exposure_distance_integral_uSv_h_m
            ),
            "total_exposure_distance_integral": (
                self.total_exposure_distance_integral_uSv_h_m
            ),
            "distance_integral_unit": self.contract.distance_integral_unit,
            "first_discovery_time_s": self.first_discovery_time_s,
            "exposure_end_time_s": self.exposure_end_time_s,
            "pre_discovery_excess_dose": self.pre_discovery_excess_dose_uSv,
            "pre_discovery_total_dose": self.pre_discovery_total_dose_uSv,
        }


def _resolve_background(
    background_uSv_h: float | None,
    background_rate: float | None,
) -> float:
    if background_rate is None:
        if background_uSv_h is None:
            raise ValueError("A background radiation rate must be provided.")
        background_rate = background_uSv_h
    elif background_uSv_h is not None and not math.isclose(
        background_rate,
        background_uSv_h,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Generic and legacy background values disagree.")
    if not math.isfinite(background_rate) or background_rate < 0.0:
        raise ValueError("Background radiation rate must be finite and non-negative.")
    return float(background_rate)


def _sample_distances(total_distance_m: float, step_m: float) -> np.ndarray:
    if not math.isfinite(step_m) or step_m <= 0.0:
        raise ValueError("integration_step_m must be finite and positive.")
    if total_distance_m == 0.0:
        return np.asarray([0.0])
    distances = np.arange(0.0, total_distance_m, step_m, dtype=float)
    if distances.size == 0 or distances[0] != 0.0:
        distances = np.insert(distances, 0, 0.0)
    if not math.isclose(distances[-1], total_distance_m, abs_tol=1e-12):
        distances = np.append(distances, total_distance_m)
    else:
        distances[-1] = total_distance_m
    return distances


def _cumulative_trapezoid(values: np.ndarray, distances: np.ndarray) -> np.ndarray:
    if values.size != distances.size:
        raise ValueError("values and distances must have the same length.")
    if values.size == 1:
        return np.asarray([0.0])
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(distances)
    return np.concatenate((np.asarray([0.0]), np.cumsum(increments)))


def integrate_radiation_along_route(
    trajectory: ExecutedTrajectory,
    excess_truth_query: RadiationTruthQuery,
    *,
    query_height_m: float,
    background_uSv_h: float | None = None,
    background_rate: float | None = None,
    contract: RadiationQuantityContract = DEFAULT_USV_CONTRACT,
    integration_step_m: float = 1.0,
    route_samples: RouteIntegrationSamples | None = None,
) -> RouteRadiationIntegral:
    """Integrate static radiation truth over distance using trapezoidal sampling."""
    if not callable(excess_truth_query):
        raise TypeError("excess_truth_query must be callable.")
    if not math.isfinite(query_height_m) or query_height_m < 0.0:
        raise ValueError("query_height_m must be finite and non-negative.")
    background_value = _resolve_background(background_uSv_h, background_rate)
    if not isinstance(contract, RadiationQuantityContract):
        raise TypeError("contract must be a RadiationQuantityContract.")
    if route_samples is None:
        route_samples = prepare_route_integration_samples(
            trajectory,
            integration_step_m=integration_step_m,
        )
    else:
        _validate_route_integration_samples(
            route_samples,
            trajectory,
            integration_step_m,
        )
    distances = route_samples.distances_m
    sampled_coordinates = route_samples.sampled_xy_m
    excess_values = np.empty(distances.size, dtype=float)
    for index, (x_m, y_m) in enumerate(sampled_coordinates):
        value = float(excess_truth_query(float(x_m), float(y_m), query_height_m))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Radiation truth must return finite non-negative excess.")
        excess_values[index] = value
    total_values = excess_values + background_value
    return RouteRadiationIntegral(
        distances_m=distances,
        excess_rates_uSv_h=excess_values,
        total_rates_uSv_h=total_values,
        cumulative_excess_integral_uSv_h_m=_cumulative_trapezoid(
            excess_values,
            distances,
        ),
        cumulative_total_integral_uSv_h_m=_cumulative_trapezoid(
            total_values,
            distances,
        ),
        query_height_m=float(query_height_m),
        integration_step_m=float(integration_step_m),
        contract=contract,
    )


def dose_from_distance_integral(
    distance_integral_uSv_h_m: float,
    evaluation_speed_m_s: float | None,
) -> float | None:
    """Convert a distance-domain rate integral to dose for constant speed."""
    if evaluation_speed_m_s is None:
        return None
    if not math.isfinite(evaluation_speed_m_s) or evaluation_speed_m_s <= 0.0:
        raise ValueError("evaluation_speed_m_s must be finite and positive.")
    return float(distance_integral_uSv_h_m / (evaluation_speed_m_s * 3600.0))


def evaluate_survivor_radiation(
    trajectory: ExecutedTrajectory,
    survivors: tuple[Point, ...] | list[Point],
    discoveries: tuple[SurvivorDiscovery, ...],
    excess_truth_query: RadiationTruthQuery,
    *,
    query_height_m: float,
    background_uSv_h: float | None = None,
    background_rate: float | None = None,
    contract: RadiationQuantityContract = DEFAULT_USV_CONTRACT,
    evaluation_speed_m_s: float | None,
) -> tuple[SurvivorRadiationRecord, ...]:
    """Evaluate static 1 m truth until discovery or mission-end censoring."""
    if len(survivors) != len(discoveries):
        raise ValueError("survivors and discoveries must have equal length.")
    if evaluation_speed_m_s is not None and (
        not math.isfinite(evaluation_speed_m_s) or evaluation_speed_m_s <= 0.0
    ):
        raise ValueError("evaluation_speed_m_s must be finite and positive.")
    background_value = _resolve_background(background_uSv_h, background_rate)
    if not isinstance(contract, RadiationQuantityContract):
        raise TypeError("contract must be a RadiationQuantityContract.")
    records = []
    mission_distance = trajectory.total_distance_m
    for survivor, discovery in zip(survivors, discoveries, strict=True):
        excess_rate = float(
            excess_truth_query(survivor.x, survivor.y, query_height_m)
        )
        if not math.isfinite(excess_rate) or excess_rate < 0.0:
            raise ValueError("Radiation truth must return finite non-negative excess.")
        total_rate = excess_rate + background_value
        exposure_distance = discovery.exposure_end_distance_m
        excess_integral = excess_rate * exposure_distance
        total_integral = total_rate * exposure_distance
        discovery_fraction = (
            discovery.first_discovery_distance_m / mission_distance
            if discovery.found and mission_distance > 0.0
            else (0.0 if discovery.found else None)
        )
        discovery_time = (
            discovery.first_discovery_distance_m / evaluation_speed_m_s
            if discovery.found and evaluation_speed_m_s is not None
            else None
        )
        exposure_end_time = (
            exposure_distance / evaluation_speed_m_s
            if evaluation_speed_m_s is not None
            else None
        )
        records.append(
            SurvivorRadiationRecord(
                survivor_id=discovery.survivor_id,
                x_m=float(survivor.x),
                y_m=float(survivor.y),
                found=discovery.found,
                first_discovery_distance_m=discovery.first_discovery_distance_m,
                first_discovery_fraction_of_mission=discovery_fraction,
                exposure_end_distance_m=exposure_distance,
                ground_truth_excess_rate_uSv_h=excess_rate,
                ground_truth_total_rate_uSv_h=total_rate,
                excess_exposure_distance_integral_uSv_h_m=excess_integral,
                total_exposure_distance_integral_uSv_h_m=total_integral,
                first_discovery_time_s=discovery_time,
                exposure_end_time_s=exposure_end_time,
                pre_discovery_excess_dose_uSv=dose_from_distance_integral(
                    excess_integral,
                    evaluation_speed_m_s,
                ),
                pre_discovery_total_dose_uSv=dose_from_distance_integral(
                    total_integral,
                    evaluation_speed_m_s,
                ),
                contract=contract,
            )
        )
    return tuple(records)


def distribution_statistics(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    """Return stable descriptive statistics for one scalar distribution."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return {
            "count": 0,
            "sum": None,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p90": None,
            "p95": None,
        }
    if not np.isfinite(data).all():
        raise ValueError("Distribution values must be finite.")
    return {
        "count": int(data.size),
        "sum": float(data.sum()),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "min": float(data.min()),
        "max": float(data.max()),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
    }


def summarise_survivor_radiation(
    records: tuple[SurvivorRadiationRecord, ...],
) -> dict[str, object]:
    """Summarise all survivors, retaining mission-end-censored non-discoveries."""
    found_records = tuple(record for record in records if record.found)
    all_excess_integrals = [
        record.excess_exposure_distance_integral_uSv_h_m for record in records
    ]
    all_total_integrals = [
        record.total_exposure_distance_integral_uSv_h_m for record in records
    ]
    found_excess_integrals = [
        record.excess_exposure_distance_integral_uSv_h_m
        for record in found_records
    ]
    found_total_integrals = [
        record.total_exposure_distance_integral_uSv_h_m
        for record in found_records
    ]
    all_excess_doses = [
        record.pre_discovery_excess_dose_uSv
        for record in records
        if record.pre_discovery_excess_dose_uSv is not None
    ]
    all_total_doses = [
        record.pre_discovery_total_dose_uSv
        for record in records
        if record.pre_discovery_total_dose_uSv is not None
    ]
    found_excess_doses = [
        record.pre_discovery_excess_dose_uSv
        for record in found_records
        if record.pre_discovery_excess_dose_uSv is not None
    ]
    found_total_doses = [
        record.pre_discovery_total_dose_uSv
        for record in found_records
        if record.pre_discovery_total_dose_uSv is not None
    ]
    total_count = len(records)
    found_count = len(found_records)
    contract = records[0].contract if records else DEFAULT_USV_CONTRACT
    if any(record.contract != contract for record in records):
        raise ValueError("All survivor records must share one quantity contract.")
    return {
        "radiation_quantity": contract.quantity,
        "radiation_rate_unit": contract.rate_unit,
        "radiation_dose_unit": contract.dose_unit,
        "distance_integral_unit": contract.distance_integral_unit,
        "total_survivor_count": total_count,
        "found_count": found_count,
        "not_found_count": total_count - found_count,
        "found_fraction": found_count / total_count if total_count else 0.0,
        "main_population": "all_survivors_including_mission_end_censored_non_discoveries",
        "all_survivors": {
            "excess_exposure_distance_integral": distribution_statistics(
                all_excess_integrals
            ),
            "total_exposure_distance_integral": distribution_statistics(
                all_total_integrals
            ),
            "pre_discovery_excess_dose": distribution_statistics(
                all_excess_doses
            ),
            "pre_discovery_total_dose": distribution_statistics(
                all_total_doses
            ),
        },
        "found_survivors_only": {
            "excess_exposure_distance_integral": distribution_statistics(
                found_excess_integrals
            ),
            "total_exposure_distance_integral": distribution_statistics(
                found_total_integrals
            ),
            "pre_discovery_excess_dose": distribution_statistics(
                found_excess_doses
            ),
            "pre_discovery_total_dose": distribution_statistics(
                found_total_doses
            ),
        },
    }


__all__ = [
    "DEFAULT_USV_CONTRACT",
    "RadiationQuantityContract",
    "RadiationTruthQuery",
    "RouteRadiationIntegral",
    "SurvivorRadiationRecord",
    "distribution_statistics",
    "dose_from_distance_integral",
    "evaluate_survivor_radiation",
    "integrate_radiation_along_route",
    "prepare_route_integration_samples",
    "RouteIntegrationSamples",
    "summarise_survivor_radiation",
]
