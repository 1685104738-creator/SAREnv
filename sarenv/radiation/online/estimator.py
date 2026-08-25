"""Fixed-cost local incremental radiation estimation on a 1 m grid."""

from __future__ import annotations

import math

import numpy as np

from ..common.grid import GridSpec
from .measurement import (
    RadiationEstimate,
    RadiationMeasurement,
    RadiationRegionEstimate,
)


RADIATION_UPDATE_RADIUS_M = 45.0
RADIATION_KERNEL_SIGMA_M = 15.0


class IncrementalRadiationGrid:
    """Maintain a Gaussian-weighted mean and cumulative weight per cell."""

    def __init__(
        self,
        grid: GridSpec,
        *,
        quantity: str,
        unit: str,
        update_radius_m: float = RADIATION_UPDATE_RADIUS_M,
        kernel_sigma_m: float = RADIATION_KERNEL_SIGMA_M,
        value_reference_height_m: float | None = None,
    ) -> None:
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        if not quantity:
            raise ValueError("quantity must be a non-empty string.")
        if not unit:
            raise ValueError("unit must be a non-empty string.")
        if not math.isfinite(update_radius_m) or update_radius_m <= 0.0:
            raise ValueError("update_radius_m must be finite and positive.")
        if not math.isfinite(kernel_sigma_m) or kernel_sigma_m <= 0.0:
            raise ValueError("kernel_sigma_m must be finite and positive.")
        if value_reference_height_m is not None and (
            not math.isfinite(value_reference_height_m)
            or value_reference_height_m < 0.0
        ):
            raise ValueError(
                "value_reference_height_m must be finite and non-negative."
            )

        self.grid = grid
        self.quantity = quantity
        self.unit = unit
        self.update_radius_m = float(update_radius_m)
        self.kernel_sigma_m = float(kernel_sigma_m)
        self.value_reference_height_m = value_reference_height_m
        self._mean = np.zeros(grid.shape, dtype=np.float64)
        self._weight_sum = np.zeros(grid.shape, dtype=np.float64)

    @property
    def mean(self) -> np.ndarray:
        """Return a numeric read-only view; unsupported cells contain zero."""
        view = self._mean.view()
        view.flags.writeable = False
        return view

    @property
    def weight_sum(self) -> np.ndarray:
        """Return a read-only view of cumulative Gaussian support."""
        view = self._weight_sum.view()
        view.flags.writeable = False
        return view

    @property
    def observed_mask(self) -> np.ndarray:
        """Return a boolean mask derived from positive cumulative support."""
        return self._weight_sum > 0.0

    def update(self, measurement: RadiationMeasurement) -> int:
        """Assimilate one measurement into only its circular local ROI."""
        if not isinstance(measurement, RadiationMeasurement):
            raise TypeError("measurement must be a RadiationMeasurement.")
        if measurement.quantity != self.quantity or measurement.unit != self.unit:
            raise ValueError(
                "Measurement quantity/unit does not match the estimator contract."
            )
        if self.value_reference_height_m is not None and not math.isclose(
            measurement.value_reference_height_m,
            self.value_reference_height_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Measurement value-reference height does not match the estimator contract."
            )

        bounds = self._local_index_bounds(measurement.x_m, measurement.y_m)
        if bounds is None:
            return 0
        min_row, max_row, min_col, max_col = bounds

        rows = np.arange(min_row, max_row + 1, dtype=np.int64)
        cols = np.arange(min_col, max_col + 1, dtype=np.int64)
        x_coordinates = self.grid.minx + (
            cols.astype(np.float64) + 0.5
        ) * self.grid.resolution_m
        y_coordinates = self.grid.miny + (
            rows.astype(np.float64) + 0.5
        ) * self.grid.resolution_m
        dx_m, dy_m = np.meshgrid(
            x_coordinates - measurement.x_m,
            y_coordinates - measurement.y_m,
        )
        distance_squared = dx_m * dx_m + dy_m * dy_m
        local_mask = distance_squared <= self.update_radius_m**2
        if not np.any(local_mask):
            return 0

        local_weights = np.exp(
            -0.5 * distance_squared / (self.kernel_sigma_m**2)
        )
        mean_roi = self._mean[min_row : max_row + 1, min_col : max_col + 1]
        weight_roi = self._weight_sum[
            min_row : max_row + 1, min_col : max_col + 1
        ]

        old_weights = weight_roi[local_mask]
        new_weights = local_weights[local_mask]
        old_means = mean_roi[local_mask]
        first_observation = old_weights == 0.0
        updated_means = np.empty_like(old_means)
        updated_means[first_observation] = measurement.value
        repeated = ~first_observation
        updated_means[repeated] = (
            new_weights[repeated] * measurement.value
            + old_weights[repeated] * old_means[repeated]
        ) / (new_weights[repeated] + old_weights[repeated])

        mean_roi[local_mask] = updated_means
        weight_roi[local_mask] = old_weights + new_weights
        return int(np.count_nonzero(local_mask))

    def estimate_at(self, x_m: float, y_m: float) -> RadiationEstimate:
        """Return a numeric estimate; zero support remains separately observable."""
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError("Estimate query coordinates must be finite.")
        if not self.grid.contains(x_m, y_m):
            return self._zero_estimate(x_m, y_m)

        row, col = self.grid.world_to_row_col(x_m, y_m)
        weight = float(self._weight_sum[row, col])
        return RadiationEstimate(
            x_m=float(x_m),
            y_m=float(y_m),
            value=float(self._mean[row, col]),
            weight_sum=weight,
            quantity=self.quantity,
            unit=self.unit,
        )

    def _zero_estimate(self, x_m: float, y_m: float) -> RadiationEstimate:
        return RadiationEstimate(
            x_m=float(x_m),
            y_m=float(y_m),
            value=0.0,
            weight_sum=0.0,
            quantity=self.quantity,
            unit=self.unit,
        )

    def max_estimate_in_bounds(
        self,
        bounds: tuple[float, float, float, float],
    ) -> RadiationRegionEstimate:
        """Return the MAX over every radiation cell overlapping the region."""
        if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("bounds must contain four finite values.")
        minx, miny, maxx, maxy = (float(value) for value in bounds)
        if maxx <= minx or maxy <= miny:
            raise ValueError("bounds must have positive width and height.")

        resolution = self.grid.resolution_m
        min_col = math.floor((minx - self.grid.minx) / resolution)
        max_col = math.ceil((maxx - self.grid.minx) / resolution) - 1
        min_row = math.floor((miny - self.grid.miny) / resolution)
        max_row = math.ceil((maxy - self.grid.miny) / resolution) - 1
        min_col = max(0, min_col)
        max_col = min(self.grid.width - 1, max_col)
        min_row = max(0, min_row)
        max_row = min(self.grid.height - 1, max_row)

        if min_col > max_col or min_row > max_row:
            return RadiationRegionEstimate(
                bounds=(minx, miny, maxx, maxy),
                value=0.0,
                max_weight_sum=0.0,
                supported_cell_count=0,
                total_cell_count=0,
                quantity=self.quantity,
                unit=self.unit,
            )

        mean_roi = self._mean[min_row : max_row + 1, min_col : max_col + 1]
        weight_roi = self._weight_sum[
            min_row : max_row + 1,
            min_col : max_col + 1,
        ]
        return RadiationRegionEstimate(
            bounds=(minx, miny, maxx, maxy),
            value=float(np.max(mean_roi)),
            max_weight_sum=float(np.max(weight_roi)),
            supported_cell_count=int(np.count_nonzero(weight_roi > 0.0)),
            total_cell_count=int(mean_roi.size),
            quantity=self.quantity,
            unit=self.unit,
        )

    def _local_index_bounds(
        self, x_m: float, y_m: float
    ) -> tuple[int, int, int, int] | None:
        resolution = self.grid.resolution_m
        min_col = math.ceil(
            (x_m - self.update_radius_m - self.grid.minx) / resolution - 0.5
        )
        max_col = math.floor(
            (x_m + self.update_radius_m - self.grid.minx) / resolution - 0.5
        )
        min_row = math.ceil(
            (y_m - self.update_radius_m - self.grid.miny) / resolution - 0.5
        )
        max_row = math.floor(
            (y_m + self.update_radius_m - self.grid.miny) / resolution - 0.5
        )

        min_col = max(0, min_col)
        max_col = min(self.grid.width - 1, max_col)
        min_row = max(0, min_row)
        max_row = min(self.grid.height - 1, max_row)
        if min_col > max_col or min_row > max_row:
            return None
        return min_row, max_row, min_col, max_col


__all__ = [
    "IncrementalRadiationGrid",
    "RADIATION_KERNEL_SIGMA_M",
    "RADIATION_UPDATE_RADIUS_M",
]
