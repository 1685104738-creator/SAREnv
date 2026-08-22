import pytest

from sarenv.radiation.online.measurement import RadiationMeasurement
from sarenv.radiation.online.trigger import (
    AboveBackgroundCriterion,
    ConsecutiveRadiationTrigger,
    RADIATION_CONFIRMATION_SAMPLES,
)


QUANTITY = "synthetic_total_gamma_dose_rate"
UNIT = "uSv/h"


def _measurement(value: float, *, unit: str = UNIT) -> RadiationMeasurement:
    return RadiationMeasurement(
        x_m=10.0,
        y_m=20.0,
        altitude_m=50.0,
        simulated_time_s=1.0,
        value=value,
        quantity=QUANTITY,
        unit=unit,
    )


def _trigger() -> ConsecutiveRadiationTrigger:
    return ConsecutiveRadiationTrigger(
        AboveBackgroundCriterion(0.2, quantity=QUANTITY, unit=UNIT)
    )


def test_v1_confirmation_baseline_is_three_samples():
    assert RADIATION_CONFIRMATION_SAMPLES == 3
    trigger = _trigger()

    assert not trigger.observe(_measurement(0.21))
    assert not trigger.observe(_measurement(0.21))
    assert trigger.observe(_measurement(0.21))
    assert trigger.confirmed


def test_background_or_lower_measurement_breaks_consecutive_run():
    trigger = _trigger()

    assert not trigger.observe(_measurement(0.21))
    assert not trigger.observe(_measurement(0.2))
    assert trigger.consecutive_count == 0
    assert not trigger.observe(_measurement(0.21))
    assert not trigger.observe(_measurement(0.21))
    assert trigger.observe(_measurement(0.21))


def test_confirmation_is_latched_until_explicit_reset():
    trigger = _trigger()
    for _ in range(3):
        trigger.observe(_measurement(0.21))

    assert trigger.observe(_measurement(0.0))
    trigger.reset()
    assert not trigger.confirmed
    assert trigger.consecutive_count == 0


def test_background_contract_rejects_incompatible_measurement_unit():
    trigger = _trigger()

    with pytest.raises(ValueError, match="quantity/unit"):
        trigger.observe(_measurement(0.21, unit="uGy/h"))

    for _ in range(3):
        trigger.observe(_measurement(0.21))
    with pytest.raises(ValueError, match="quantity/unit"):
        trigger.observe(_measurement(0.21, unit="uGy/h"))
