from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point
from shapely.geometry import box

from sarenv.analytics.metrics import PathEvaluator
from sarenv.evaluation import (
    EvaluationConfig,
    ExecutedTrajectory,
    RadiationQuantityContract,
    SurfaceKermaTruthField,
    TrajectoryNode,
    discover_survivors,
    evaluate_native_sar_metrics,
    evaluate_survivor_radiation,
    first_discovery_distance_m,
    integrate_radiation_along_route,
    prepare_route_integration_samples,
)
from sarenv.evaluation.exposure import _sample_distances
from sarenv.radiation import GridSpec
from sarenv.radiation.surface import (
    SurfacePhotonResponseConfig,
    SurfacePhotonResponseKernel,
    UniformPolygonConfig,
    UniformPolygonSource,
    simulate_surface_source,
)


def _trajectory(x_coordinates=(0.0, 10.0)):
    nodes = []
    cumulative = 0.0
    previous = None
    for index, x_m in enumerate(x_coordinates):
        segment = 0.0 if previous is None else float(x_m - previous)
        cumulative += segment
        nodes.append(
            TrajectoryNode(
                step_index=index,
                x_m=float(x_m),
                y_m=0.0,
                mode="NORMAL",
                segment_distance_m=segment,
                cumulative_distance_m=cumulative,
            )
        )
        previous = x_m
    return ExecutedTrajectory(tuple(nodes))


def _polyline_trajectory(coordinates):
    nodes = []
    cumulative = 0.0
    previous = None
    for index, (x_m, y_m) in enumerate(coordinates):
        segment = 0.0 if previous is None else float(
            np.hypot(x_m - previous[0], y_m - previous[1])
        )
        cumulative += segment
        nodes.append(
            TrajectoryNode(
                step_index=index,
                x_m=float(x_m),
                y_m=float(y_m),
                mode="NORMAL",
                segment_distance_m=segment,
                cumulative_distance_m=cumulative,
            )
        )
        previous = (x_m, y_m)
    return ExecutedTrajectory(tuple(nodes))


def test_vectorised_sampler_matches_shapely_at_existing_sample_distances():
    trajectory = _polyline_trajectory(
        ((0.0, 0.0), (3.0, 4.0), (3.0, 4.0), (9.0, 4.0), (9.0, 1.0))
    )
    distances = _sample_distances(trajectory.total_distance_m, 1.0)

    actual = trajectory.sample_coordinates(distances)
    expected = np.asarray(
        [
            trajectory.line.interpolate(float(distance_m)).coords[0]
            for distance_m in distances
        ],
        dtype=float,
    )

    np.testing.assert_array_equal(distances, _sample_distances(14.0, 1.0))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_vectorised_sampler_preserves_clipping_and_zero_length_route():
    trajectory = _polyline_trajectory(((2.0, 3.0), (2.0, 3.0)))

    actual = trajectory.sample_coordinates(
        np.asarray([-5.0, 0.0, 10.0], dtype=float)
    )

    np.testing.assert_array_equal(
        actual,
        np.asarray(((2.0, 3.0), (2.0, 3.0), (2.0, 3.0))),
    )


def test_trajectory_shapely_geometry_is_cached():
    trajectory = _trajectory()

    assert trajectory.line is trajectory.line
    assert trajectory.coordinates is trajectory.coordinates


def test_route_integration_samples_are_reused_without_resampling(monkeypatch):
    trajectory = _polyline_trajectory(((0.0, 0.0), (3.0, 4.0), (9.0, 4.0)))
    calls = {"count": 0}
    original = ExecutedTrajectory.sample_coordinates

    def counted_sample_coordinates(self, distances_m):
        calls["count"] += 1
        return original(self, distances_m)

    monkeypatch.setattr(
        ExecutedTrajectory,
        "sample_coordinates",
        counted_sample_coordinates,
    )
    samples = prepare_route_integration_samples(
        trajectory,
        integration_step_m=1.0,
    )
    first = integrate_radiation_along_route(
        trajectory,
        lambda x_m, y_m, z_m: x_m + y_m,
        query_height_m=50.0,
        background_rate=0.2,
        route_samples=samples,
    )
    second = integrate_radiation_along_route(
        trajectory,
        lambda x_m, y_m, z_m: x_m + y_m,
        query_height_m=50.0,
        background_rate=0.2,
        route_samples=samples,
    )

    assert calls["count"] == 1
    np.testing.assert_array_equal(first.distances_m, samples.distances_m)
    np.testing.assert_array_equal(second.excess_rates_uSv_h, first.excess_rates_uSv_h)
    assert not samples.distances_m.flags.writeable
    assert not samples.sampled_xy_m.flags.writeable


def test_route_integration_cache_rejects_a_different_trajectory():
    first = _trajectory((0.0, 10.0))
    second = _polyline_trajectory(((0.0, 0.0), (6.0, 8.0)))
    samples = prepare_route_integration_samples(first, integration_step_m=1.0)

    with pytest.raises(ValueError, match="different trajectory"):
        integrate_radiation_along_route(
            second,
            lambda x_m, y_m, z_m: 1.0,
            query_height_m=50.0,
            background_rate=0.2,
            route_samples=samples,
        )


def test_first_discovery_uses_exact_segment_entry_not_only_waypoints():
    trajectory = _trajectory()

    distance = first_discovery_distance_m(
        trajectory,
        Point(5.0, 0.0),
        detection_radius_m=1.0,
    )

    assert distance == pytest.approx(4.0)


def test_survivor_inside_start_footprint_is_discovered_at_zero():
    assert first_discovery_distance_m(
        _trajectory(),
        Point(0.5, 0.0),
        detection_radius_m=1.0,
    ) == pytest.approx(0.0)


def test_unfound_survivor_exposure_is_censored_at_mission_end():
    trajectory = _trajectory()
    survivor = Point(5.0, 5.0)
    discoveries = discover_survivors(trajectory, (survivor,), 1.0)

    records = evaluate_survivor_radiation(
        trajectory,
        (survivor,),
        discoveries,
        lambda x_m, y_m, z_m: 2.0,
        query_height_m=1.0,
        background_uSv_h=0.2,
        evaluation_speed_m_s=5.0,
    )

    record = records[0]
    assert not record.found
    assert record.first_discovery_distance_m is None
    assert record.exposure_end_distance_m == pytest.approx(10.0)
    assert record.excess_exposure_distance_integral_uSv_h_m == pytest.approx(20.0)
    assert record.pre_discovery_excess_dose_uSv == pytest.approx(20.0 / (5 * 3600))


def test_first_discovery_is_independent_of_collinear_route_discretisation():
    survivor = Point(5.0, 0.0)

    coarse = first_discovery_distance_m(_trajectory((0.0, 10.0)), survivor, 1.0)
    densified = first_discovery_distance_m(
        _trajectory((0.0, 2.0, 5.0, 7.0, 10.0)),
        survivor,
        1.0,
    )

    assert coarse == pytest.approx(4.0)
    assert densified == pytest.approx(coarse)


@pytest.mark.parametrize("spacing_m", [0.5, 1.0, 2.0])
def test_uav_radiation_integral_is_stable_across_sampling_intervals(spacing_m):
    integral = integrate_radiation_along_route(
        _trajectory(),
        lambda x_m, y_m, z_m: 2.0,
        query_height_m=50.0,
        background_uSv_h=0.2,
        integration_step_m=spacing_m,
    )

    assert integral.excess_distance_integral_uSv_h_m == pytest.approx(20.0)
    assert integral.total_distance_integral_uSv_h_m == pytest.approx(22.0)


def test_constant_field_dose_matches_analytic_constant_speed_result():
    integral = integrate_radiation_along_route(
        _trajectory(),
        lambda x_m, y_m, z_m: 2.0,
        query_height_m=50.0,
        background_uSv_h=0.2,
        integration_step_m=1.0,
    )

    expected_total_dose = 22.0 / (5.0 * 3600.0)
    assert integral.total_distance_integral_uSv_h_m / (5.0 * 3600.0) == pytest.approx(
        expected_total_dose
    )


def test_uav_truth_is_queried_only_at_platform_height():
    heights = []

    def truth(x_m, y_m, z_m):
        heights.append(z_m)
        return 1.0

    integrate_radiation_along_route(
        _trajectory(),
        truth,
        query_height_m=50.0,
        background_uSv_h=0.2,
        integration_step_m=2.0,
    )

    assert heights
    assert set(heights) == {50.0}


def test_survivor_truth_is_queried_only_at_ground_reference_height():
    heights = []
    trajectory = _trajectory()
    survivor = Point(5.0, 0.0)
    discoveries = discover_survivors(trajectory, (survivor,), 1.0)

    def truth(x_m, y_m, z_m):
        heights.append(z_m)
        return 1.0

    evaluate_survivor_radiation(
        trajectory,
        (survivor,),
        discoveries,
        truth,
        query_height_m=1.0,
        background_uSv_h=0.2,
        evaluation_speed_m_s=None,
    )

    assert heights == [1.0]


def test_truth_metrics_require_explicit_truth_query_not_planner_estimator():
    query_calls = []

    def truth(x_m, y_m, z_m):
        query_calls.append((x_m, y_m, z_m))
        return 1.0

    integrate_radiation_along_route(
        _trajectory(),
        truth,
        query_height_m=50.0,
        background_uSv_h=0.2,
        integration_step_m=1.0,
    )

    assert query_calls


def test_native_sar_metrics_call_existing_path_evaluator(monkeypatch):
    called = {"value": False}

    def fake_calculate(self, paths, discount_factor):
        called["value"] = True
        assert discount_factor == pytest.approx(0.999)
        return {
            "total_likelihood_score": 0.75,
            "total_time_discounted_score": 0.5,
            "victim_detection_metrics": {
                "percentage_found": 100.0,
                "found_victim_indices": [0],
            },
            "area_covered": 0.001,
            "total_path_length": 0.01,
            "cumulative_distances": [np.asarray([0.0, 10.0])],
            "cumulative_likelihoods": [np.asarray([0.25, 0.75])],
            "cumulative_time_discounted_scores": [np.asarray([0.25, 0.5])],
        }

    monkeypatch.setattr(PathEvaluator, "calculate_all_metrics", fake_calculate)
    monkeypatch.setattr(PathEvaluator, "get_visible_cells", lambda self, x, y: {(0, 0)})
    item = SimpleNamespace(
        heatmap=np.asarray([[1.0]]),
        bounds=(0.0, -1.0, 10.0, 1.0),
        meter_per_bin=10.0,
    )
    survivors = SimpleNamespace(
        points=(Point(5.0, 0.0),),
        projected_crs="EPSG:32630",
    )

    metrics, distances, probability = evaluate_native_sar_metrics(
        _trajectory(),
        item,
        survivors,
        fov_deg=45.0,
        altitude_m=50.0,
        discount_factor=0.999,
    )

    assert called["value"]
    assert metrics["implementation"] == "sarenv.analytics.metrics.PathEvaluator"
    assert metrics["total_likelihood_score"] == pytest.approx(0.75)
    np.testing.assert_array_equal(distances, [0.0, 10.0])
    np.testing.assert_array_equal(probability, [0.25, 0.75])


def test_surface_evaluation_plane_preserves_native_unit_and_ground_result():
    grid = GridSpec.from_bounds((0.0, 0.0, 7.0, 7.0), "EPSG:32630")
    source = UniformPolygonSource(
        UniformPolygonConfig(
            source_id="surface-evaluation-height-test",
            geometry=box(2.0, 2.0, 5.0, 5.0),
            activity_density_bq_m2=1.0e6,
            crs=grid.crs,
        )
    )
    kernel = SurfacePhotonResponseKernel.create(
        SurfacePhotonResponseConfig(),
        grid,
    )
    result = simulate_surface_source(source, kernel, nominal_grid=grid)
    truth = SurfaceKermaTruthField.from_simulation(
        result,
        platform_altitude_m=50.0,
    )

    expected_ground = (
        result.collision_air_kerma_rate_patch.query_collision_air_kerma_rate(
            3.5,
            3.5,
        )
    )
    assert truth.quantity == "collision_air_kerma_rate"
    assert truth.unit == "uGy/h"
    assert truth.query_excess(3.5, 3.5, 1.0) == pytest.approx(expected_ground)
    assert 0.0 < truth.query_excess(3.5, 3.5, 50.0) < expected_ground
    with pytest.raises(ValueError, match="available only"):
        truth.query_excess(3.5, 3.5, 20.0)


def test_generic_radiation_contract_is_preserved_in_survivor_output():
    trajectory = _trajectory()
    survivor = Point(5.0, 0.0)
    discoveries = discover_survivors(trajectory, (survivor,), 1.0)
    contract = RadiationQuantityContract(
        quantity="collision_air_kerma_rate",
        rate_unit="uGy/h",
        dose_unit="uGy",
    )

    record = evaluate_survivor_radiation(
        trajectory,
        (survivor,),
        discoveries,
        lambda x_m, y_m, z_m: 2.0,
        query_height_m=1.0,
        background_rate=0.0,
        contract=contract,
        evaluation_speed_m_s=5.0,
    )[0]

    payload = record.to_dict()
    assert payload["radiation_quantity"] == "collision_air_kerma_rate"
    assert payload["radiation_rate_unit"] == "uGy/h"
    assert payload["radiation_dose_unit"] == "uGy"


def test_evaluation_config_rejects_negative_legacy_background(tmp_path):
    with pytest.raises(ValueError, match="background_uSv_h"):
        EvaluationConfig(
            trajectory_path=tmp_path / "route.csv",
            dataset_directory=tmp_path,
            survivors_path=tmp_path / "survivors.json",
            scenario_parameters_path=tmp_path / "scenario.json",
            output_directory=tmp_path / "output",
            platform_altitude_m=50.0,
            survivor_reference_height_m=1.0,
            fov_deg=45.0,
            background_uSv_h=-0.1,
        )
