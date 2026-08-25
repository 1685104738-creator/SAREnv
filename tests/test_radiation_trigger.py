import pytest

from sarenv.radiation.online.measurement import RadiationMeasurement
from sarenv.radiation.online.trigger import (
    AdaptiveHysteresisRadiationTrigger,
    AboveBackgroundCriterion,
    ConsecutiveRadiationTrigger,
    HysteresisRadiationTrigger,
    NOMINAL_HYSTERESIS_MARGIN,
    NOISE_FREE_CONFIRMATION_SAMPLES,
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


def test_noise_free_hysteresis_uses_one_sample_and_ten_percent_margin():
    trigger = HysteresisRadiationTrigger(
        enter_threshold=10.0,
        quantity=QUANTITY,
        unit=UNIT,
    )

    assert NOMINAL_HYSTERESIS_MARGIN == 0.10
    assert NOISE_FREE_CONFIRMATION_SAMPLES == 1
    assert trigger.exit_threshold == pytest.approx(9.0)
    assert trigger.observe(_measurement(10.0))
    assert trigger.last_transition == "enter"
    assert trigger.observe(_measurement(9.5))
    assert trigger.last_transition is None
    assert not trigger.observe(_measurement(9.0))
    assert trigger.last_transition == "exit"


def test_hysteresis_band_holds_the_current_mode_without_bounce():
    trigger = HysteresisRadiationTrigger(
        enter_threshold=10.0,
        quantity=QUANTITY,
        unit=UNIT,
    )

    assert not trigger.observe(_measurement(9.5))
    assert trigger.last_transition is None
    assert trigger.observe(_measurement(11.0))
    assert trigger.observe(_measurement(9.5))
    assert trigger.observe(_measurement(9.1))
    assert not trigger.observe(_measurement(8.9))
    assert not trigger.observe(_measurement(9.5))

    for _ in range(3):
        trigger.observe(_measurement(0.21))
    with pytest.raises(ValueError, match="quantity/unit"):
        trigger.observe(_measurement(0.21, unit="uGy/h"))


def _adaptive_trigger(multiplier: float = 0.1):
    return AdaptiveHysteresisRadiationTrigger(
        base_hazard_reference=10.0,
        initial_entry_multiplier=multiplier,
        quantity=QUANTITY,
        unit=UNIT,
    )


def _complete_adaptive_episode(trigger: AdaptiveHysteresisRadiationTrigger) -> None:
    assert trigger.observe(_measurement(trigger.current_entry_threshold))
    assert trigger.last_transition == "enter"
    assert not trigger.observe(_measurement(trigger.current_exit_threshold))
    assert trigger.last_transition == "exit"
    trigger.reset_after_episode()


def test_adaptive_tenth_multiplier_progresses_once_per_completed_episode():
    trigger = _adaptive_trigger(0.1)

    observed_thresholds = []
    for _ in range(3):
        observed_thresholds.append(trigger.current_entry_threshold)
        _complete_adaptive_episode(trigger)

    assert observed_thresholds == pytest.approx([1.0, 2.0, 3.0])
    assert trigger.current_entry_threshold == pytest.approx(4.0)
    assert trigger.threshold_increment_count == 3
    assert trigger.completed_episode_count == 3


def test_adaptive_threshold_never_exceeds_base_hazard_reference():
    trigger = _adaptive_trigger(0.1)

    for _ in range(15):
        _complete_adaptive_episode(trigger)
        assert trigger.current_entry_threshold <= trigger.base_hazard_reference

    assert trigger.current_entry_threshold == pytest.approx(10.0)
    assert trigger.threshold_increment_count == 9


def test_adaptive_multiplier_one_remains_at_base_reference():
    trigger = _adaptive_trigger(1.0)

    for _ in range(3):
        assert trigger.current_entry_threshold == pytest.approx(10.0)
        assert trigger.current_exit_threshold == pytest.approx(9.0)
        _complete_adaptive_episode(trigger)

    assert trigger.current_entry_threshold == pytest.approx(10.0)
    assert trigger.threshold_increment_count == 0


def test_adaptive_threshold_is_constant_inside_one_episode():
    trigger = _adaptive_trigger(0.1)
    assert trigger.observe(_measurement(1.0))

    for value in (20.0, 5.0, 1.0):
        assert trigger.observe(_measurement(value))
        assert trigger.current_entry_threshold == pytest.approx(1.0)
        assert trigger.current_exit_threshold == pytest.approx(0.9)
        assert trigger.threshold_increment_count == 0


def test_adaptive_threshold_increases_only_after_exit():
    trigger = _adaptive_trigger(0.1)

    assert not trigger.observe(_measurement(0.5))
    assert trigger.current_entry_threshold == pytest.approx(1.0)
    assert trigger.observe(_measurement(1.0))
    assert trigger.current_entry_threshold == pytest.approx(1.0)
    assert trigger.observe(_measurement(0.95))
    assert trigger.current_entry_threshold == pytest.approx(1.0)
    assert not trigger.observe(_measurement(0.9))
    assert trigger.current_entry_threshold == pytest.approx(2.0)
    assert trigger.threshold_increment_count == 1


def test_adaptive_exit_is_always_ten_percent_below_current_enter():
    trigger = _adaptive_trigger(0.1)

    for _ in range(12):
        assert trigger.current_exit_threshold == pytest.approx(
            0.9 * trigger.current_entry_threshold
        )
        _complete_adaptive_episode(trigger)


def test_adaptive_new_mission_reset_restores_initial_threshold():
    trigger = _adaptive_trigger(0.1)
    _complete_adaptive_episode(trigger)
    _complete_adaptive_episode(trigger)
    assert trigger.current_entry_threshold == pytest.approx(3.0)

    trigger.reset()

    assert trigger.current_entry_threshold == pytest.approx(1.0)
    assert trigger.current_exit_threshold == pytest.approx(0.9)
    assert trigger.current_episode_index == 1
    assert trigger.completed_episode_count == 0
    assert trigger.threshold_increment_count == 0
    assert not trigger.confirmed
