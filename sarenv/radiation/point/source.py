"""Analytic synthetic point-source truth model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..common.grid import GridSpec
from ..common.metadata import RADIATION_GENERATOR_VERSION
from .config import PointSourceConfig


@dataclass(frozen=True)
class PointSource:
    """
    A deterministic analytic point source returning excess dose rate only.

    When ``z_m`` is omitted, queries use the reference plane at
    ``source_height_m + reference_distance_m``. This is a ground-truth model,
    not a UAV detector response or radiation-transport calculation.
    """

    config: PointSourceConfig

    def query_excess_dose_rate(
        self, x_m: float, y_m: float, z_m: float | None = None
    ) -> float:
        """Return the softened inverse-square-inspired excess in uSv/h."""
        values = (x_m, y_m)
        if not np.isfinite(values).all():
            raise ValueError("Point-source query coordinates must be finite.")
        query_z = (
            self.config.source_height_m + self.config.reference_distance_m
            if z_m is None
            else float(z_m)
        )
        if not np.isfinite(query_z):
            raise ValueError("z_m must be finite when provided.")

        dx = float(x_m) - self.config.x_m
        dy = float(y_m) - self.config.y_m
        dz = query_z - self.config.source_height_m
        distance_squared = dx * dx + dy * dy + dz * dz
        distance_m = float(np.sqrt(distance_squared))
        if (
            self.config.cutoff_radius_m is not None
            and distance_m > self.config.cutoff_radius_m
        ):
            return 0.0

        core_squared = self.config.core_radius_m**2
        reference_squared = self.config.reference_distance_m**2
        scale = (reference_squared + core_squared) / (
            distance_squared + core_squared
        )
        return float(self.config.reference_excess_uSv_h * scale)

    def rasterize(self, grid: GridSpec, z_m: float | None = None) -> np.ndarray:
        """Rasterize excess values on a local 1 m grid for visualisation/tests."""
        x_coordinates, y_coordinates = grid.cell_center_mesh()
        query_z = (
            self.config.source_height_m + self.config.reference_distance_m
            if z_m is None
            else float(z_m)
        )
        if not np.isfinite(query_z):
            raise ValueError("z_m must be finite when provided.")
        distance_squared = (
            (x_coordinates - self.config.x_m) ** 2
            + (y_coordinates - self.config.y_m) ** 2
            + (query_z - self.config.source_height_m) ** 2
        )
        core_squared = self.config.core_radius_m**2
        reference_squared = self.config.reference_distance_m**2
        field = self.config.reference_excess_uSv_h * (
            reference_squared + core_squared
        ) / (distance_squared + core_squared)
        if self.config.cutoff_radius_m is not None:
            field = np.where(
                np.sqrt(distance_squared) <= self.config.cutoff_radius_m,
                field,
                0.0,
            )
        return np.asarray(field, dtype=float)

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-ready point-source metadata."""
        return {
            "schema_version": 1,
            "model_type": "analytic_point_source",
            "scenario_type": "point_only",
            "propagation_model": self.config.model_name,
            "quantity": "synthetic_excess_gamma_dose_rate",
            "background_included": False,
            "generator_version": RADIATION_GENERATOR_VERSION,
            **asdict(self.config),
            "default_query_plane_z_m": (
                self.config.source_height_m
                + self.config.reference_distance_m
            ),
            "formula": (
                "E(r)=E_ref*(reference_distance_m^2+core_radius_m^2)/"
                "(r^2+core_radius_m^2)"
            ),
            "cutoff_rule": (
                "zero_when_3d_distance_exceeds_cutoff_radius_m"
                if self.config.cutoff_radius_m is not None
                else "no_cutoff"
            ),
        }


__all__ = ["PointSource"]
