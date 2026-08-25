"""Data contracts shared by online radiation sensing and estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, init=False)
class RadiationMeasurement:
    """One scalar observation with separate platform and value-reference heights."""

    x_m: float
    y_m: float
    platform_altitude_m: float
    value_reference_height_m: float
    simulated_time_s: float
    value: float
    quantity: str
    unit: str

    def __init__(
        self,
        *,
        x_m: float,
        y_m: float,
        simulated_time_s: float,
        value: float,
        quantity: str,
        unit: str,
        platform_altitude_m: float | None = None,
        value_reference_height_m: float | None = None,
        altitude_m: float | None = None,
    ) -> None:
        """Build a measurement; ``altitude_m`` is a legacy platform alias."""
        if platform_altitude_m is None:
            if altitude_m is None:
                raise TypeError("platform_altitude_m must be provided.")
            platform_altitude_m = altitude_m
        elif altitude_m is not None and not math.isclose(
            platform_altitude_m,
            altitude_m,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("platform_altitude_m and altitude_m disagree.")
        if value_reference_height_m is None:
            value_reference_height_m = platform_altitude_m

        object.__setattr__(self, "x_m", float(x_m))
        object.__setattr__(self, "y_m", float(y_m))
        object.__setattr__(self, "platform_altitude_m", float(platform_altitude_m))
        object.__setattr__(
            self,
            "value_reference_height_m",
            float(value_reference_height_m),
        )
        object.__setattr__(self, "simulated_time_s", float(simulated_time_s))
        object.__setattr__(self, "value", float(value))
        object.__setattr__(self, "quantity", str(quantity))
        object.__setattr__(self, "unit", str(unit))
        self.__post_init__()

    def __post_init__(self) -> None:
        numeric = {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "platform_altitude_m": self.platform_altitude_m,
            "value_reference_height_m": self.value_reference_height_m,
            "simulated_time_s": self.simulated_time_s,
            "value": self.value,
        }
        for name, value in numeric.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.platform_altitude_m < 0.0:
            raise ValueError("platform_altitude_m must be non-negative.")
        if self.value_reference_height_m < 0.0:
            raise ValueError("value_reference_height_m must be non-negative.")
        if self.simulated_time_s < 0.0:
            raise ValueError("simulated_time_s must be non-negative.")
        if self.value < 0.0:
            raise ValueError("Radiation measurement value must be non-negative.")
        if not self.quantity:
            raise ValueError("quantity must be a non-empty string.")
        if not self.unit:
            raise ValueError("unit must be a non-empty string.")

    @property
    def altitude_m(self) -> float:
        """Return platform altitude for compatibility with earlier callers."""
        return self.platform_altitude_m


@dataclass(frozen=True)
class RadiationEstimate:
    """A numeric planner-safe excess estimate plus independent support weight."""

    x_m: float
    y_m: float
    value: float
    weight_sum: float
    quantity: str
    unit: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.x_m) or not math.isfinite(self.y_m):
            raise ValueError("Estimate coordinates must be finite.")
        if not math.isfinite(self.weight_sum) or self.weight_sum < 0.0:
            raise ValueError("weight_sum must be finite and non-negative.")
        if not math.isfinite(self.value) or self.value < 0.0:
            raise ValueError("Estimate value must be finite and non-negative.")
        if not self.quantity:
            raise ValueError("quantity must be a non-empty string.")
        if not self.unit:
            raise ValueError("unit must be a non-empty string.")

    @property
    def observed(self) -> bool:
        """Return whether this location has any estimator support."""
        return self.weight_sum > 0.0


@dataclass(frozen=True)
class RadiationRegionEstimate:
    """Maximum numeric estimate and support summary for one world-space region."""

    bounds: tuple[float, float, float, float]
    value: float
    max_weight_sum: float
    supported_cell_count: int
    total_cell_count: int
    quantity: str
    unit: str

    def __post_init__(self) -> None:
        if len(self.bounds) != 4 or not all(math.isfinite(v) for v in self.bounds):
            raise ValueError("Region bounds must contain four finite values.")
        minx, miny, maxx, maxy = self.bounds
        if maxx <= minx or maxy <= miny:
            raise ValueError("Region bounds must have positive width and height.")
        if not math.isfinite(self.value) or self.value < 0.0:
            raise ValueError("Region estimate must be finite and non-negative.")
        if not math.isfinite(self.max_weight_sum) or self.max_weight_sum < 0.0:
            raise ValueError("max_weight_sum must be finite and non-negative.")
        if self.supported_cell_count < 0 or self.total_cell_count < 0:
            raise ValueError("Region cell counts must be non-negative.")
        if self.supported_cell_count > self.total_cell_count:
            raise ValueError("Supported cells cannot exceed total overlapping cells.")
        if not self.quantity or not self.unit:
            raise ValueError("Region estimate quantity and unit must be non-empty.")

    @property
    def observed(self) -> bool:
        return self.supported_cell_count > 0

    @property
    def support_fraction(self) -> float:
        if self.total_cell_count == 0:
            return 0.0
        return self.supported_cell_count / self.total_cell_count


__all__ = [
    "RadiationEstimate",
    "RadiationMeasurement",
    "RadiationRegionEstimate",
]
