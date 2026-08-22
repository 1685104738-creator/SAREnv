"""Active pre-planner radiation scenario selection."""

from __future__ import annotations

from enum import Enum


class RadiationScenarioType(str, Enum):
    """Radiation modes currently supported as independent scenarios."""

    NO_RADIATION = "no_radiation"
    POINT_ONLY = "point_only"
    SURFACE_ONLY = "surface_only"


def resolve_radiation_scenario(
    value: str | RadiationScenarioType,
) -> RadiationScenarioType:
    """Resolve one active scenario and clearly reject mixed modes."""
    if isinstance(value, RadiationScenarioType):
        return value
    try:
        return RadiationScenarioType(value)
    except ValueError as exc:
        raise ValueError(
            "Unsupported radiation scenario. Active options are: "
            "no_radiation, point_only, surface_only. Point + Surface is not "
            "supported by the current active pipeline."
        ) from exc


__all__ = ["RadiationScenarioType", "resolve_radiation_scenario"]
