"""Deprecated wrapper for the removed heatmap-aligned radiation layer API."""

from __future__ import annotations

import warnings


_MIGRATION_MESSAGE = (
    "generate_radiation_layer(dataset_path, RadiationConfig) has been removed "
    "because it inferred a radiation grid from heatmap.npy. Build an independent "
    "1 m UniformPolygonSource or ZonedPolygonSource and call "
    "sarenv.radiation.simulate_surface_source(), then "
    "save_surface_simulation()."
)


def generate_radiation_layer(*args, **kwargs):
    """Reject the old coupled API and point callers to the local-patch workflow."""
    del args, kwargs
    warnings.warn(_MIGRATION_MESSAGE, DeprecationWarning, stacklevel=2)
    raise NotImplementedError(_MIGRATION_MESSAGE)


__all__ = ["generate_radiation_layer"]
