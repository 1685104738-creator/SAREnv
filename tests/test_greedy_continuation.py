import numpy as np
import pytest

from sarenv.analytics.paths import (
    generate_greedy_continuation_route,
    generate_greedy_path,
    greedy_visible_cells,
)


BOUNDS = (0.0, 0.0, 90.0, 90.0)
MAP_SHAPE = (3, 3)


def test_continuation_matches_characterised_original_first_step():
    probability_map = np.ones(MAP_SHAPE, dtype=float)
    probability_map[1, 2] = 9.0
    test_only_fov_deg = 1.0
    test_only_altitude_m = 1.0
    detection_radius_m = test_only_altitude_m * np.tan(
        np.radians(test_only_fov_deg / 2.0)
    )
    initial_observed = greedy_visible_cells(
        1,
        1,
        map_shape=MAP_SHAPE,
        bounds=BOUNDS,
        detection_radius_m=detection_radius_m,
    )
    original = generate_greedy_path(
        45.0,
        45.0,
        1,
        probability_map,
        BOUNDS,
        100.0,
        fov_deg=test_only_fov_deg,
        altitude=test_only_altitude_m,
        budget=30.0,
    )[0]

    continuation = generate_greedy_continuation_route(
        current_position_m=(45.0, 45.0),
        observed_cells=initial_observed,
        search_center=(45.0, 45.0),
        remaining_budget_m=30.0,
        probability_map=probability_map,
        bounds=BOUNDS,
        max_radius=100.0,
        fov_deg=test_only_fov_deg,
        altitude=test_only_altitude_m,
    )

    assert list(original.coords) == [(45.0, 45.0), (75.0, 45.0)]
    assert list(continuation.coords) == list(original.coords)


def test_continuation_inherits_observed_probability_without_mutating_input():
    probability_map = np.full(MAP_SHAPE, 0.01)
    probability_map[1, 2] = 0.5
    probability_map[2, 1] = 0.4
    observed = {(1, 1), (1, 2)}
    observed_before = set(observed)

    continuation = generate_greedy_continuation_route(
        current_position_m=(45.0, 45.0),
        observed_cells=observed,
        search_center=(45.0, 45.0),
        remaining_budget_m=30.0,
        probability_map=probability_map,
        bounds=BOUNDS,
        max_radius=100.0,
        fov_deg=0.0,
        altitude=50.0,
    )

    assert list(continuation.coords)[:2] == [(45.0, 45.0), (45.0, 75.0)]
    assert observed == observed_before


def test_continuation_starts_at_actual_position_and_respects_remaining_budget():
    probability_map = np.full(MAP_SHAPE, 0.01)
    probability_map[2, 2] = 1.0
    current_position = (45.0, 45.0)
    test_only_remaining_budget_m = 40.0

    continuation = generate_greedy_continuation_route(
        current_position_m=current_position,
        observed_cells={(1, 1)},
        search_center=(45.0, 45.0),
        remaining_budget_m=test_only_remaining_budget_m,
        probability_map=probability_map,
        bounds=BOUNDS,
        max_radius=100.0,
        fov_deg=0.0,
        altitude=50.0,
    )

    assert continuation.coords[0] == current_position
    assert continuation.length == pytest.approx(test_only_remaining_budget_m)
    assert continuation.length <= test_only_remaining_budget_m


def test_continuation_uses_changed_current_position_but_fixed_search_center():
    probability_map = np.full(MAP_SHAPE, 0.01)
    probability_map[0, 1] = 0.5

    continuation = generate_greedy_continuation_route(
        current_position_m=(15.0, 15.0),
        observed_cells={(0, 0)},
        search_center=(45.0, 45.0),
        remaining_budget_m=30.0,
        probability_map=probability_map,
        bounds=BOUNDS,
        max_radius=100.0,
        fov_deg=0.0,
        altitude=50.0,
    )

    assert list(continuation.coords)[:2] == [(15.0, 15.0), (45.0, 15.0)]


def test_non_positive_remaining_budget_produces_no_continuation_route():
    route = generate_greedy_continuation_route(
        current_position_m=(45.0, 45.0),
        observed_cells={(1, 1)},
        search_center=(45.0, 45.0),
        remaining_budget_m=0.0,
        probability_map=np.ones(MAP_SHAPE),
        bounds=BOUNDS,
        max_radius=100.0,
        fov_deg=0.0,
        altitude=50.0,
    )

    assert route.is_empty
