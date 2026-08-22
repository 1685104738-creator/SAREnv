import numpy as np

from sarenv.analytics.paths import (
    generate_greedy_path,
    greedy_visible_cells,
    score_greedy_neighbours,
)


BOUNDS = (0.0, 0.0, 90.0, 90.0)
MAP_SHAPE = (3, 3)


def test_extracted_scoring_preserves_characterised_positive_first_step():
    probability_map = np.ones(MAP_SHAPE, dtype=float)
    probability_map[1, 2] = 9.0
    observed = greedy_visible_cells(
        1,
        1,
        map_shape=MAP_SHAPE,
        bounds=BOUNDS,
        detection_radius_m=1.0 * np.tan(np.radians(1.0 / 2.0)),
    )

    candidates = score_greedy_neighbours(
        (1, 1),
        observed,
        probability_map=probability_map,
        bounds=BOUNDS,
        search_center=(45.0, 45.0),
        max_radius=100.0,
        detection_radius_m=1.0 * np.tan(np.radians(1.0 / 2.0)),
    )
    helper_choice = max(candidates, key=lambda item: item[1])[0]
    route = generate_greedy_path(
        45.0,
        45.0,
        1,
        probability_map,
        BOUNDS,
        100.0,
        fov_deg=1.0,
        altitude=1.0,
    )[0]

    assert helper_choice == (1, 2)
    assert list(route.coords)[:2] == [(45.0, 45.0), (75.0, 45.0)]


def test_neighbour_scoring_returns_only_eight_connected_cells():
    probability_map = np.ones((5, 5), dtype=float)
    candidates = score_greedy_neighbours(
        (2, 2),
        set(),
        probability_map=probability_map,
        bounds=(0.0, 0.0, 150.0, 150.0),
        search_center=(75.0, 75.0),
        max_radius=200.0,
        detection_radius_m=0.0,
    )

    positions = {position for position, _ in candidates}
    assert len(positions) == 8
    assert positions == {
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
    }
