"""
Tests for rosie.heart_tools — agent tools that interact with the Heart.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt

from rosie.state import VehicleState, ControlObjective
from rosie.controllers.passthrough import PassthroughController
from rosie.controllers.pid_position import PIDPositionController
import rosie.heart_tools as ht


class _FakeHeart:
    """Minimal Heart stand-in for testing tools."""

    def __init__(self):
        self._state = VehicleState(
            stamp=time.monotonic(),
            position=np.array([1.0, 2.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.array([0.5, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
        )
        self._objective = ControlObjective(mode="idle")
        self._controllers = {
            "passthrough": PassthroughController(),
            "pid_position": PIDPositionController(),
        }
        self._active_controller = self._controllers["passthrough"]
        self._cmd_vel_topic = "/cmd_vel"
        self._trajectory_setpoint_topic = None

    def get_state(self):
        return self._state

    def get_objective(self):
        return self._objective

    def set_objective(self, obj):
        self._objective = obj

    def switch_controller(self, name):
        if name in self._controllers:
            self._active_controller = self._controllers[name]
            return True
        return False

    def get_controller_names(self):
        return list(self._controllers.keys())

    def get_status(self):
        return {
            "controller": self._active_controller.name,
            "available_controllers": self.get_controller_names(),
            "objective": self._objective.as_dict(),
            "state": self._state.as_dict(),
            "cmd_vel_topic": self._cmd_vel_topic,
            "trajectory_setpoint_topic": self._trajectory_setpoint_topic,
        }


# Fixture: register/deregister the fake heart for each test
import pytest


@pytest.fixture(autouse=True)
def _fake_heart():
    heart = _FakeHeart()
    ht.set_heart(heart)
    yield heart
    ht.set_heart(None)


class TestHeartGetState:
    def test_returns_position(self):
        result = json.loads(ht.heart_get_state())
        assert result["position"] == [1.0, 2.0, 0.0]

    def test_returns_velocity(self):
        result = json.loads(ht.heart_get_state())
        assert result["linear_velocity"] == [0.5, 0.0, 0.0]


class TestHeartGetStatus:
    def test_includes_controller(self):
        result = json.loads(ht.heart_get_status())
        assert result["controller"] == "passthrough"
        assert "pid_position" in result["available_controllers"]

    def test_includes_objective(self):
        result = json.loads(ht.heart_get_status())
        assert result["objective"]["mode"] == "idle"

    def test_includes_topics(self):
        result = json.loads(ht.heart_get_status())
        assert result["cmd_vel_topic"] == "/cmd_vel"


class TestHeartSetVelocity:
    def test_sets_velocity_objective(self, _fake_heart):
        result = json.loads(ht.heart_set_velocity(vx=1.0, yaw_rate=0.5))
        assert result["status"] == "velocity_set"
        assert result["velocity"] == [1.0, 0.0, 0.0]

        obj = _fake_heart.get_objective()
        assert obj.mode == "velocity"
        npt.assert_array_equal(obj.target_velocity, [1.0, 0.0, 0.0])

    def test_switches_to_passthrough(self, _fake_heart):
        _fake_heart.switch_controller("pid_position")
        ht.heart_set_velocity(vx=1.0)
        assert _fake_heart._active_controller.name == "passthrough"


class TestHeartGoToPosition:
    def test_sets_position_objective(self, _fake_heart):
        result = json.loads(ht.heart_go_to_position(x=10.0, y=5.0))
        assert result["status"] == "navigating"
        assert result["target"] == [10.0, 5.0, 0.0]

        obj = _fake_heart.get_objective()
        assert obj.mode == "position"
        npt.assert_array_equal(obj.target_position, [10.0, 5.0, 0.0])

    def test_switches_to_pid(self, _fake_heart):
        ht.heart_go_to_position(x=1.0, y=2.0)
        assert _fake_heart._active_controller.name == "pid_position"

    def test_with_yaw(self, _fake_heart):
        result = json.loads(ht.heart_go_to_position(x=1.0, y=2.0, yaw=1.57))
        assert result["yaw"] == 1.57


class TestHeartStop:
    def test_sets_idle(self, _fake_heart):
        _fake_heart.set_objective(ControlObjective(mode="velocity"))
        result = json.loads(ht.heart_stop())
        assert result["status"] == "stopped"
        assert _fake_heart.get_objective().mode == "idle"


class TestHeartSwitchController:
    def test_valid_switch(self, _fake_heart):
        result = json.loads(ht.heart_switch_controller("pid_position"))
        assert result["status"] == "switched"

    def test_invalid_switch(self, _fake_heart):
        result = json.loads(ht.heart_switch_controller("nonexistent"))
        assert "error" in result
        assert "available" in result


class TestHeartSetTrajectory:
    def test_sets_waypoints(self, _fake_heart):
        wps = [[1, 2, 0], [3, 4, 0], [5, 6, 0]]
        result = json.loads(ht.heart_set_trajectory(wps))
        assert result["status"] == "trajectory_set"
        assert result["waypoint_count"] == 3

        obj = _fake_heart.get_objective()
        assert obj.mode == "trajectory"
        assert len(obj.waypoints) == 3


class TestHeartNotRunning:
    def test_raises_without_heart(self):
        ht.set_heart(None)
        with pytest.raises(RuntimeError, match="Heart node is not running"):
            ht.heart_get_state()
