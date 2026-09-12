import inspect

import pytest

from sarenv.core.mission import MissionClock


def test_mission_clock_advances_from_distance_and_explicit_test_speed():
    test_only_speed_m_s = 5.0
    clock = MissionClock()

    current_time = clock.advance_travel(10.0, uav_speed_m_s=test_only_speed_m_s)

    assert current_time == pytest.approx(2.0)
    assert clock.time_s == pytest.approx(2.0)


def test_mission_clock_has_no_production_speed_default():
    parameter = inspect.signature(MissionClock.advance_travel).parameters[
        "uav_speed_m_s"
    ]

    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="uav_speed_m_s"):
        MissionClock().advance_travel(10.0)


@pytest.mark.parametrize("speed", [0.0, -1.0, float("inf")])
def test_mission_clock_rejects_invalid_speed(speed):
    with pytest.raises(ValueError, match="uav_speed_m_s"):
        MissionClock().advance_travel(10.0, uav_speed_m_s=speed)
