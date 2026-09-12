# sarenv/analytics/paths.py
"""
Collection of coverage path generation algorithms for drones.
"""
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring


GREEDY_NEIGHBOUR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _greedy_grid_geometry(
    map_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[int, int, float, float, float, float]:
    height, width = map_shape
    minx, miny, maxx, maxy = bounds
    dx = (maxx - minx) / width
    dy = (maxy - miny) / height
    return height, width, dx, dy, minx + dx / 2, miny + dy / 2


def greedy_grid_position_to_world(
    row: int,
    col: int,
    *,
    map_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return the SAR cell centre using Original Greedy's coordinate mapping."""
    _, _, dx, dy, x_offset, y_offset = _greedy_grid_geometry(map_shape, bounds)
    return x_offset + col * dx, y_offset + row * dy


def greedy_grid_cell_bounds(
    row: int,
    col: int,
    *,
    map_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return the pixel-edge world bounds for one SAR Greedy cell."""
    height, width, dx, dy, _, _ = _greedy_grid_geometry(map_shape, bounds)
    if not 0 <= row < height or not 0 <= col < width:
        raise IndexError("SAR row/col lies outside map_shape.")
    minx, miny, _, _ = bounds
    cell_minx = minx + col * dx
    cell_miny = miny + row * dy
    return (
        cell_minx,
        cell_miny,
        cell_minx + dx,
        cell_miny + dy,
    )


def greedy_searchable_cells(
    *,
    map_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    search_center: tuple[float, float],
    max_radius: float,
) -> set[tuple[int, int]]:
    """Return cells whose centres satisfy Original Greedy's radius rule."""
    height, width, dx, dy, x_offset, y_offset = _greedy_grid_geometry(
        map_shape,
        bounds,
    )
    center_x, center_y = search_center
    radius_squared = max_radius * max_radius
    return {
        (row, col)
        for row in range(height)
        for col in range(width)
        if (x_offset + col * dx - center_x) ** 2
        + (y_offset + row * dy - center_y) ** 2
        < radius_squared
    }


def native_greedy_step_allowance(
    map_shape: tuple[int, int],
    *,
    num_drones: int = 1,
) -> int:
    """Return Original Greedy's resolution-derived maximum move count."""
    if num_drones <= 0:
        raise ValueError("num_drones must be positive.")
    height, width = map_shape
    if height <= 0 or width <= 0:
        raise ValueError("map_shape dimensions must be positive.")
    return height * width // num_drones


def greedy_world_to_grid_position(
    x_m: float,
    y_m: float,
    *,
    map_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[int, int]:
    """Map a world position as Original Greedy maps its starting position."""
    height, width, dx, dy, _, _ = _greedy_grid_geometry(map_shape, bounds)
    minx, miny, _, _ = bounds
    col = np.clip(int((x_m - minx) / dx), 0, width - 1)
    row = np.clip(int((y_m - miny) / dy), 0, height - 1)
    return int(row), int(col)


def _greedy_visible_cells_with_geometry(
    row: int,
    col: int,
    *,
    height: int,
    width: int,
    dx: float,
    dy: float,
    x_offset: float,
    y_offset: float,
    detection_radius_m: float,
) -> set[tuple[int, int]]:
    detection_radius_cells_x = int(np.ceil(detection_radius_m / dx))
    detection_radius_cells_y = int(np.ceil(detection_radius_m / dy))
    world_x = x_offset + col * dx
    world_y = y_offset + row * dy
    visible_cells = set()
    for visible_row in range(
        max(0, row - detection_radius_cells_y),
        min(height, row + detection_radius_cells_y + 1),
    ):
        for visible_col in range(
            max(0, col - detection_radius_cells_x),
            min(width, col + detection_radius_cells_x + 1),
        ):
            cell_x = x_offset + visible_col * dx
            cell_y = y_offset + visible_row * dy
            distance = np.sqrt(
                (cell_x - world_x) ** 2 + (cell_y - world_y) ** 2
            )
            if distance <= detection_radius_m:
                visible_cells.add((visible_row, visible_col))
    return visible_cells


def greedy_visible_cells(
    row: int,
    col: int,
    *,
    map_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    detection_radius_m: float,
) -> set[tuple[int, int]]:
    """Return SAR cells visible from one executed or candidate grid position."""
    height, width, dx, dy, x_offset, y_offset = _greedy_grid_geometry(
        map_shape, bounds
    )
    return _greedy_visible_cells_with_geometry(
        row,
        col,
        height=height,
        width=width,
        dx=dx,
        dy=dy,
        x_offset=x_offset,
        y_offset=y_offset,
        detection_radius_m=detection_radius_m,
    )


def score_greedy_neighbours(
    current_position: tuple[int, int],
    observed_cells: set[tuple[int, int]],
    *,
    probability_map: np.ndarray,
    bounds: tuple[float, float, float, float],
    search_center: tuple[float, float],
    max_radius: float,
    detection_radius_m: float,
) -> list[tuple[tuple[int, int], float]]:
    """Score Original Greedy's valid 8-connected neighbours without selecting."""
    height, width, dx, dy, x_offset, y_offset = _greedy_grid_geometry(
        probability_map.shape, bounds
    )
    center_x, center_y = search_center
    max_radius_sq = max_radius * max_radius
    current_row, current_col = current_position
    candidates = []
    for row_offset, col_offset in GREEDY_NEIGHBOUR_OFFSETS:
        row = current_row + row_offset
        col = current_col + col_offset
        if row < 0 or row >= height or col < 0 or col >= width:
            continue

        world_x = x_offset + col * dx
        world_y = y_offset + row * dy
        distance_squared = (
            (world_x - center_x) ** 2 + (world_y - center_y) ** 2
        )
        if distance_squared >= max_radius_sq:
            continue

        visible_cells = _greedy_visible_cells_with_geometry(
            row,
            col,
            height=height,
            width=width,
            dx=dx,
            dy=dy,
            x_offset=x_offset,
            y_offset=y_offset,
            detection_radius_m=detection_radius_m,
        )
        new_cells = visible_cells - observed_cells
        score = sum(probability_map[r, c] for r, c in new_cells)
        candidates.append(((row, col), score))
    return candidates


def select_original_greedy_candidate(
    candidates: list[tuple[tuple[int, int], float]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    """Apply Original Greedy's score maximum and zero-score random fallback."""
    if not candidates:
        return None
    best_position, maximum_score = max(candidates, key=lambda item: item[1])
    if maximum_score <= 0:
        random_index = rng.choice(len(candidates))
        return candidates[random_index][0]
    return best_position


def generate_greedy_continuation_route(
    *,
    current_position_m: tuple[float, float],
    observed_cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
    search_center: tuple[float, float],
    remaining_budget_m: float | None = None,
    remaining_steps: int | None = None,
    probability_map: np.ndarray,
    bounds: tuple[float, float, float, float],
    max_radius: float,
    fov_deg: float,
    altitude: float,
    rng: np.random.Generator | None = None,
) -> LineString:
    """Generate one Original Greedy continuation from actually executed state."""
    if remaining_budget_m is None and remaining_steps is None:
        raise ValueError("A remaining distance budget or step allowance is required.")
    if remaining_budget_m is not None and remaining_budget_m <= 0:
        return LineString()
    if remaining_steps is not None:
        if isinstance(remaining_steps, bool) or not isinstance(remaining_steps, int):
            raise TypeError("remaining_steps must be an integer.")
        if remaining_steps <= 0:
            return LineString()
    if rng is None:
        rng = np.random.default_rng()

    detection_radius_m = altitude * np.tan(np.radians(fov_deg / 2))
    current_grid_position = greedy_world_to_grid_position(
        *current_position_m,
        map_shape=probability_map.shape,
        bounds=bounds,
    )
    working_observed_cells = set(observed_cells)
    planned_grid_positions = [current_grid_position]
    planned_world_positions = [
        (float(current_position_m[0]), float(current_position_m[1]))
    ]
    _, _, dx, _, _, _ = _greedy_grid_geometry(probability_map.shape, bounds)
    limits = []
    if remaining_budget_m is not None:
        limits.append(int(remaining_budget_m // dx))
    if remaining_steps is not None:
        limits.append(remaining_steps)
    max_iterations = min(limits)
    searchable_cells = greedy_searchable_cells(
        map_shape=probability_map.shape,
        bounds=bounds,
        search_center=search_center,
        max_radius=max_radius,
    )

    iteration = 0
    while iteration < max_iterations:
        if searchable_cells.issubset(working_observed_cells):
            break
        iteration += 1
        candidates = score_greedy_neighbours(
            current_grid_position,
            working_observed_cells,
            probability_map=probability_map,
            bounds=bounds,
            search_center=search_center,
            max_radius=max_radius,
            detection_radius_m=detection_radius_m,
        )
        next_grid_position = select_original_greedy_candidate(candidates, rng)
        if next_grid_position is None:
            break

        current_grid_position = next_grid_position
        planned_grid_positions.append(current_grid_position)
        planned_world_positions.append(
            greedy_grid_position_to_world(
                *current_grid_position,
                map_shape=probability_map.shape,
                bounds=bounds,
            )
        )
        working_observed_cells.update(
            greedy_visible_cells(
                *current_grid_position,
                map_shape=probability_map.shape,
                bounds=bounds,
                detection_radius_m=detection_radius_m,
            )
        )

    if len(planned_grid_positions) <= 1:
        return LineString()
    route = LineString(planned_world_positions)
    if remaining_budget_m is not None:
        return restrict_path_length(route, remaining_budget_m)
    return route

def split_path_for_drones(path: LineString, num_drones: int) -> list[LineString]:
    if num_drones <= 1 or path.is_empty or path.length == 0:
        return [path]
    segments = []
    segment_length = path.length / num_drones
    for i in range(num_drones):
        segments.append(substring(path, i * segment_length, (i + 1) * segment_length))
    return segments

# Add **kwargs to accept and ignore unused arguments
def generate_spiral_path(center_x: float, center_y: float, max_radius: float, fov_deg: float, altitude: float, overlap: float, num_drones: int, path_point_spacing_m: float, **kwargs) -> list[LineString]:
    budget = kwargs.get('budget')
    loop_spacing = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    a = loop_spacing / (2 * np.pi)
    num_rotations = max_radius / loop_spacing
    theta_max = num_rotations * 2 * np.pi
    approx_path_length = 0.5 * a * (theta_max * np.sqrt(1 + theta_max**2) + np.log(theta_max + np.sqrt(1 + theta_max**2))) if theta_max > 0 else 0
    num_points = int(approx_path_length / path_point_spacing_m) if path_point_spacing_m > 0 else 2000
    theta = np.linspace(0, theta_max, max(2, num_points))
    radius = np.clip(a * theta, 0, max_radius)
    full_path = LineString(zip(center_x + radius * np.cos(theta), center_y + radius * np.sin(theta), strict=True))
    paths = split_path_for_drones(full_path, num_drones)
    
    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)
    
    return paths

# Add **kwargs here as well for consistency
def generate_concentric_circles_path(center_x: float, center_y: float, max_radius: float, fov_deg: float, altitude: float, overlap: float, num_drones: int, path_point_spacing_m: float, transition_distance_m: float, **kwargs) -> list[LineString]:
    budget = kwargs.get('budget')
    radius_increment = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    path_points, current_radius = [], radius_increment
    while current_radius <= max_radius:
        transition_angle_rad = transition_distance_m / current_radius if current_radius > 0 else np.radians(45)
        arc_length = current_radius * (2 * np.pi - transition_angle_rad)
        num_points_circle = max(2, int(arc_length / path_point_spacing_m))
        theta = np.linspace(0, 2 * np.pi - transition_angle_rad, num_points_circle)
        path_points.extend(zip(center_x + current_radius * np.cos(theta), center_y + current_radius * np.sin(theta), strict=True))
        next_radius = current_radius + radius_increment
        if next_radius <= max_radius:
            path_points.append((center_x + next_radius, center_y))
        else:
            final_theta_points = max(2, int((current_radius * transition_angle_rad) / path_point_spacing_m))
            final_theta = np.linspace(2 * np.pi - transition_angle_rad, 2 * np.pi, final_theta_points)
            path_points.extend(zip(center_x + current_radius * np.cos(final_theta), center_y + current_radius * np.sin(final_theta), strict=True))
        current_radius = next_radius
    full_path = LineString(path_points) if path_points else LineString()
    paths = split_path_for_drones(full_path, num_drones)
    
    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)
    
    return paths

# Add **kwargs here too to handle arguments like 'transition_distance_m'
def generate_pizza_zigzag_path(center_x: float, center_y: float, max_radius: float, num_drones: int, fov_deg: float, altitude: float, overlap: float, path_point_spacing_m: float, border_gap_m: float, **kwargs) -> list[LineString]:
    budget = kwargs.get('budget')
    paths, section_angle_rad = [], 2 * np.pi / num_drones
    pass_width = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    for i in range(num_drones):
        base_start_angle, base_end_angle = i * section_angle_rad, (i + 1) * section_angle_rad
        points, radius, direction = [(center_x, center_y)], pass_width, 1
        while radius <= max_radius:
            angular_offset_rad = border_gap_m / radius if radius > 0 else 0
            start_angle, end_angle = base_start_angle + angular_offset_rad, base_end_angle - angular_offset_rad
            if start_angle >= end_angle:
                radius += pass_width
                continue
            arc_length = radius * (end_angle - start_angle)
            num_arc_points = max(2, int(arc_length / path_point_spacing_m))
            current_arc_angles = np.linspace(start_angle, end_angle, num_arc_points) if direction == 1 else np.linspace(end_angle, start_angle, num_arc_points)
            points.extend(zip(center_x + radius * np.cos(current_arc_angles), center_y + radius * np.sin(current_arc_angles), strict=True))
            radius += pass_width
            direction *= -1
        if len(points) > 1:
            paths.append(LineString(points))
    
    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)
    
    return paths

def generate_greedy_path(center_x: float, center_y: float, num_drones: int, probability_map: np.ndarray, bounds: tuple, max_radius: float, **kwargs) -> list[LineString]:
    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds
    rng = kwargs.get("rng")
    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    if maxx <= minx or maxy <= miny:
        return [LineString() for _ in range(num_drones)]

    # Extract view model parameters
    fov_deg = kwargs.get('fov_deg')
    altitude = kwargs.get('altitude')
    detection_radius = altitude * np.tan(np.radians(fov_deg / 2))

    # Pre-compute coordinate mappings for efficiency
    dx = (maxx - minx) / width
    dy = ((maxy - miny) / height )
    x_offset = minx + dx / 2
    y_offset = miny + dy / 2

    # Convert the real-world starting coordinates to grid indices
    start_col = np.clip(int((center_x - minx) / dx), 0, width - 1)
    start_row = np.clip(int((center_y - miny) / dy), 0, height - 1)
    start_pos = (start_row, start_col)

    # Track globally observed cells across all drones
    globally_observed_cells = set()

    # Initialize drone positions efficiently
    current_positions = [start_pos]
    for i in range(1, num_drones):
        angle = 2 * np.pi * i / num_drones
        offset_r = min(2, height // 10)
        offset_c = min(2, width // 10)
        new_r = np.clip(start_pos[0] + int(offset_r * np.sin(angle)), 0, height - 1)
        new_c = np.clip(start_pos[1] + int(offset_c * np.cos(angle)), 0, width - 1)
        current_positions.append((new_r, new_c))

    # Store paths as lists of (row, col) indices
    paths = [[] for _ in range(num_drones)]
    for i, pos in enumerate(current_positions):
        paths[i].append(pos)
        # Add visible cells from initial positions to globally observed
        visible_cells = greedy_visible_cells(
            pos[0],
            pos[1],
            map_shape=probability_map.shape,
            bounds=bounds,
            detection_radius_m=detection_radius,
        )
        globally_observed_cells.update(visible_cells)

    max_iterations = native_greedy_step_allowance(
        probability_map.shape,
        num_drones=num_drones,
    )
    
    # Estimate max_iterations based on budget (in meters) and minimum move length
    budget = kwargs.get('budget')
    if budget is not None and budget > 0:
        # Each move is at least dx meters (cell width)
        max_iterations = (budget // dx) // num_drones
    max_steps = kwargs.get("max_steps")
    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be an integer.")
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative.")
        max_iterations = min(max_iterations, max_steps)
    searchable_cells = greedy_searchable_cells(
        map_shape=probability_map.shape,
        bounds=bounds,
        search_center=(center_x, center_y),
        max_radius=max_radius,
    )

    iteration = 0
    while iteration < max_iterations:
        if searchable_cells.issubset(globally_observed_cells):
            break
        iteration += 1
        for i in range(num_drones):
            valid_neighbors = score_greedy_neighbours(
                current_positions[i],
                globally_observed_cells,
                probability_map=probability_map,
                bounds=bounds,
                search_center=(center_x, center_y),
                max_radius=max_radius,
                detection_radius_m=detection_radius,
            )

            # Choose next position based on scores
            if valid_neighbors:
                best_neighbor = select_original_greedy_candidate(
                    valid_neighbors,
                    rng,
                )
                current_positions[i] = best_neighbor
                
                paths[i].append(best_neighbor)
                
                # Update globally observed cells with newly visible cells
                visible_cells = greedy_visible_cells(
                    best_neighbor[0],
                    best_neighbor[1],
                    map_shape=probability_map.shape,
                    bounds=bounds,
                    detection_radius_m=detection_radius,
                )
                globally_observed_cells.update(visible_cells)

    # Convert grid paths back to real-world coordinate paths
    line_paths = []
    for drone_path_indices in paths:
        if len(drone_path_indices) > 1:
            # Convert (row, col) indices directly to world coordinates
            line_paths.append(LineString([(x_offset + c * dx, y_offset + r * dy) for r, c in drone_path_indices]))
        else:
            line_paths.append(LineString())

    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(line_paths, budget / num_drones)
    else:
        paths = line_paths
    
    return paths

def generate_random_walk_path(
    center_x: float,
    center_y: float,
    num_drones: int,
    probability_map,
    **kwargs
) -> list[LineString]:
    return generate_greedy_path(
        center_x=center_x,
        center_y=center_y,
        num_drones=num_drones,
        probability_map=np.zeros_like(probability_map),
        **kwargs
    )

def restrict_path_length(line: LineString, max_length: float) -> LineString:
    if isinstance(line, list):
        return [restrict_path_length(path, max_length) for path in line]
    if line.is_empty or max_length is None or max_length <= 0 or line.length <= max_length:
        return line
    return substring(line, 0, max_length)
