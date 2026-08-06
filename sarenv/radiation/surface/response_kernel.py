"""Normalised inverse-square-inspired benchmark surface response kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BenchmarkSurfaceResponseConfig


@dataclass(frozen=True)
class BenchmarkSurfaceResponseKernel:
    """Deterministic benchmark kernel; not a physical dose-response model."""

    config: BenchmarkSurfaceResponseConfig
    values: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=float)
        object.__setattr__(self, "values", array)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("Surface response kernel must be a square 2D array.")
        if array.shape[0] % 2 != 1:
            raise ValueError("Surface response kernel shape must be odd.")
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("Surface response kernel must be finite/non-negative.")
        if not np.isclose(array.sum(), 1.0, atol=1e-12):
            raise ValueError("Normalised surface response kernel must sum to one.")

    @classmethod
    def create(
        cls, config: BenchmarkSurfaceResponseConfig
    ) -> "BenchmarkSurfaceResponseKernel":
        """Build ``K(r)=1/(r^2+r0^2)`` with circular cutoff and unit sum."""
        radius_cells = int(round(config.cutoff_radius_m / config.resolution_m))
        offsets = np.arange(-radius_cells, radius_cells + 1, dtype=float)
        x_offsets, y_offsets = np.meshgrid(offsets, offsets)
        radius_squared = (
            (x_offsets * config.resolution_m) ** 2
            + (y_offsets * config.resolution_m) ** 2
        )
        cutoff_squared = config.cutoff_radius_m**2
        values = np.where(
            radius_squared <= cutoff_squared + 1e-12,
            1.0 / (radius_squared + config.core_radius_m**2),
            0.0,
        )
        kernel_sum = float(values.sum())
        if not np.isfinite(kernel_sum) or kernel_sum <= 0:
            raise ValueError("Surface response kernel has invalid total weight.")
        values /= kernel_sum
        return cls(config=config, values=values)

    @property
    def center_index(self) -> tuple[int, int]:
        """Return the central row/column index."""
        center = self.values.shape[0] // 2
        return center, center

    @property
    def radius_cells(self) -> int:
        return self.values.shape[0] // 2

    def to_metadata(self) -> dict[str, object]:
        """Return the explicit benchmark-kernel definition."""
        return {
            "kernel_class": type(self).__name__,
            "kernel_model": self.config.model_name,
            "kernel_formula": "K(r)=1/(r^2+core_radius_m^2)",
            "core_radius_m": self.config.core_radius_m,
            "cutoff_radius_m": self.config.cutoff_radius_m,
            "resolution_m": self.config.resolution_m,
            "normalise_kernel": self.config.normalise_kernel,
            "kernel_shape": list(self.values.shape),
            "kernel_center_index": list(self.center_index),
            "kernel_sum": float(self.values.sum()),
            "kernel_cutoff_geometry": "circular",
            "physical_model": False,
        }


__all__ = ["BenchmarkSurfaceResponseKernel"]
