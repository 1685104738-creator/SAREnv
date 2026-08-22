"""Contract tests for the online radiation pipeline scaffolding."""

import numpy as np
import pytest

from sarenv.radiation.common.grid import GridSpec
from sarenv.radiation.online import (
    NoiseFreeRadiationSensor,
    RadiationEstimate,
    RadiationMeasurement,
)
from sarenv.radiation.point import PointSource, PointSourceConfig
from sarenv.radiation.surface import CollisionAirKermaRatePatch


CRS = "EPSG:32630"


def test_measurement_requires_explicit_position_time_and_metadata() -> None:
    measurement = RadiationMeasurement(
        x_m=10.0,
        y_m=20.0,
        altitude_m=50.0,
        simulated_time_s=3.0,
        value=2.5,
        quantity="synthetic_excess_gamma_dose_rate",
        unit="uSv/h",
    )

    assert measurement.altitude_m == 50.0
    assert measurement.simulated_time_s == 3.0


def test_unknown_estimate_is_not_encoded_as_zero_radiation() -> None:
    estimate = RadiationEstimate(
        x_m=10.0,
        y_m=20.0,
        value=None,
        weight_sum=0.0,
        quantity="synthetic_excess_gamma_dose_rate",
        unit="uSv/h",
    )

    assert not estimate.observed
    assert estimate.value is None


def test_known_estimate_requires_positive_support() -> None:
    with pytest.raises(ValueError, match="positive weight_sum"):
        RadiationEstimate(
            x_m=10.0,
            y_m=20.0,
            value=0.0,
            weight_sum=0.0,
            quantity="synthetic_excess_gamma_dose_rate",
            unit="uSv/h",
        )


def test_noise_free_sensor_queries_point_truth_at_explicit_altitude() -> None:
    source = PointSource(
        PointSourceConfig(
            source_id="point",
            x_m=10.0,
            y_m=20.0,
            source_height_m=0.0,
            reference_excess_uSv_h=8.0,
            crs=CRS,
        )
    )
    sensor = NoiseFreeRadiationSensor(
        source.query_excess_dose_rate,
        quantity="synthetic_excess_gamma_dose_rate",
        unit="uSv/h",
    )

    measurement = sensor.measure(
        x_m=10.0,
        y_m=20.0,
        altitude_m=50.0,
        simulated_time_s=12.5,
    )

    assert measurement.value == pytest.approx(
        source.query_excess_dose_rate(10.0, 20.0, 50.0)
    )
    assert measurement.unit == "uSv/h"
    assert measurement.simulated_time_s == 12.5


def test_noise_free_sensor_preserves_surface_kerma_quantity_and_unit() -> None:
    grid = GridSpec.from_bounds((0.0, 0.0, 1.0, 1.0), CRS)
    patch = CollisionAirKermaRatePatch(
        collision_air_kerma_rate_uGy_h=np.array([[3.5]]),
        grid=grid,
    )
    sensor = NoiseFreeRadiationSensor(
        lambda x_m, y_m, altitude_m: patch.query_collision_air_kerma_rate(
            x_m, y_m
        ),
        quantity="collision_air_kerma_rate",
        unit="uGy/h",
    )

    measurement = sensor.measure(0.5, 0.5, 50.0, 1.0)

    assert measurement.value == 3.5
    assert measurement.quantity == "collision_air_kerma_rate"
    assert measurement.unit == "uGy/h"
