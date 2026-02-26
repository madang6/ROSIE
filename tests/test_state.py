"""
Tests for rosie.state — VehicleState, ControlObjective, ControlCommand.
"""

from __future__ import annotations

import time

import numpy as np

from rosie.state import VehicleState, ControlObjective, ControlCommand


class TestVehicleState:
    def test_default_state_has_none_fields(self):
        s = VehicleState()
        assert s.position is None
        assert s.orientation is None
        assert s.linear_velocity is None
        assert not s.position_valid()
        assert not s.velocity_valid()

    def test_position_valid_when_set(self):
        s = VehicleState(position=np.array([1.0, 2.0, 3.0]))
        assert s.position_valid()

    def test_velocity_valid_when_set(self):
        s = VehicleState(linear_velocity=np.array([0.5, 0.0, 0.0]))
        assert s.velocity_valid()

    def test_age_returns_inf_when_no_stamp(self):
        s = VehicleState()
        assert s.age() == float("inf")

    def test_age_returns_positive(self):
        s = VehicleState(stamp=time.monotonic() - 0.5)
        assert 0.4 < s.age() < 1.0

    def test_as_dict_serialises_arrays(self):
        s = VehicleState(
            stamp=time.monotonic(),
            position=np.array([1.0, 2.0, 3.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        d = s.as_dict()
        assert d["position"] == [1.0, 2.0, 3.0]
        assert d["orientation"] == [1.0, 0.0, 0.0, 0.0]
        assert d["linear_velocity"] is None
        assert isinstance(d["age_sec"], float)

    def test_as_dict_includes_image_topics(self):
        s = VehicleState()
        s.images["/camera/rgb"] = np.zeros(10, dtype=np.uint8)
        d = s.as_dict()
        assert "/camera/rgb" in d["image_topics"]


class TestControlObjective:
    def test_default_is_idle(self):
        obj = ControlObjective()
        assert obj.mode == "idle"

    def test_as_dict_round_trip(self):
        obj = ControlObjective(
            mode="position",
            target_position=np.array([5.0, 10.0, 0.0]),
            target_yaw=1.57,
        )
        d = obj.as_dict()
        assert d["mode"] == "position"
        assert d["target_position"] == [5.0, 10.0, 0.0]
        assert d["target_yaw"] == 1.57

    def test_velocity_objective(self):
        obj = ControlObjective(
            mode="velocity",
            target_velocity=np.array([1.0, 0.0, 0.0]),
            target_yaw_rate=0.5,
        )
        d = obj.as_dict()
        assert d["target_velocity"] == [1.0, 0.0, 0.0]
        assert d["target_yaw_rate"] == 0.5

    def test_trajectory_objective(self):
        obj = ControlObjective(
            mode="trajectory",
            waypoints=[np.array([1, 2, 0]), np.array([3, 4, 0])],
        )
        d = obj.as_dict()
        assert len(d["waypoints"]) == 2


class TestControlCommand:
    def test_default_is_zero(self):
        cmd = ControlCommand()
        np.testing.assert_array_equal(cmd.linear_velocity, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(cmd.angular_velocity, [0.0, 0.0, 0.0])
        assert cmd.position is None
