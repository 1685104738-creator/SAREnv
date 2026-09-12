"""Cs-137 photon-fluence to collision-air-kerma conversion."""

from __future__ import annotations

import numpy as np

from .config import (
    GAMMA_ENERGY_MEV,
    GY_TO_UGY,
    MASS_ENERGY_ABSORPTION_COEFF_AIR_M2_KG,
    MEV_TO_J,
    SECONDS_PER_HOUR,
)


def photon_fluence_to_collision_air_kerma_rate(
    photon_fluence_rate_ph_m2_s: float | np.ndarray,
) -> float | np.ndarray:
    """Convert Cs-137 photon fluence rate to collision air kerma in uGy/h.

    Gamma yield is intentionally absent here because it has already been
    applied when activity is converted to photon fluence by the response
    kernel. The conversion uses photon energy and the dry-air mass
    energy-absorption coefficient directly.
    """
    fluence_rate = np.asarray(
        photon_fluence_rate_ph_m2_s,
        dtype=np.float64,
    )
    if not np.isfinite(fluence_rate).all():
        raise ValueError("Photon fluence rate must contain only finite values.")
    if (fluence_rate < 0.0).any():
        raise ValueError("Photon fluence rate must be non-negative.")

    collision_air_kerma_rate_uGy_h = (
        fluence_rate
        * GAMMA_ENERGY_MEV
        * MEV_TO_J
        * MASS_ENERGY_ABSORPTION_COEFF_AIR_M2_KG
        * SECONDS_PER_HOUR
        * GY_TO_UGY
    )
    if fluence_rate.ndim == 0:
        return float(collision_air_kerma_rate_uGy_h)
    return np.asarray(collision_air_kerma_rate_uGy_h, dtype=np.float64)


__all__ = ["photon_fluence_to_collision_air_kerma_rate"]
