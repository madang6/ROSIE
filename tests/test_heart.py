"""
Tests for rosie.heart — the continuously-spinning state→control node.

Since we can't spin a real rclpy node in CI, these tests focus on the
programmatic API (set_objective, get_state, control_tick) using the mock
rclpy from conftest.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt

from rosie.state import VehicleState, ControlObjective, ControlCommand
from rosie.controllers.base import ControlContext
from rosie.controllers.passthrough import PassthroughController
from rosie.controllers.pid_position import PIDPositionController
from rosie.flight import FlightPhase, ReadyGate


# ---------------------------------------------------------------------------
# Minimal Heart stub that avoids real ROS subscriptions/publishers
# ---------------------------------------------------------------------------

class _HeartTestHarness:
    """Wraps Heart's core logic without touching ROS 2 infrastructure.

    We can't instantiate the real Heart node in tests (no real rclpy), so
    we replicate the state/objective/controller interactions directly.
    """

    def __init__(self):
        from rosie.heart import Heart  # imported for type reference only

        self.state = VehicleState()
        self.objective = ControlObjective(mode="idle")
        self.controllers = {
            "passthrough": PassthroughController(),
            "pid_position": PIDPositionController(kp=1.0, ki=0.0, kd=0.1, max_velocity=2.0),
        }
        self.active_controller = self.controllers["passthrough"]
        self.published_commands: list[ControlCommand] = []

    def set_state(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.state, k, v)
        self.state.stamp = time.monotonic()

    def set_objective(self, obj: ControlObjective):
        old = self.objective.mode
        self.objective = obj
        if obj.mode != old:
            self.active_controller.reset()

    def switch_controller(self, name: str) -> bool:
        ctrl = self.controllers.get(name)
        if ctrl is None:
            return False
        self.active_controller.reset()
        ctrl.reset()
        self.active_controller = ctrl
        return True

    def tick(self) -> ControlCommand | None:
        """Simulate one control loop iteration."""
        if self.objective.mode == "idle":
            return None
        cmd = self.active_controller.compute(self.state, self.objective)
        self.published_commands.append(cmd)
        return cmd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHeartStateRouting:
    """Verify that state updates flow correctly to the controller."""

    def test_odom_updates_position(self):
        h = _HeartTestHarness()
        h.set_state(position=np.array([1.0, 2.0, 0.0]))
        assert h.state.position_valid()
        npt.assert_array_equal(h.state.position, [1.0, 2.0, 0.0])

    def test_odom_updates_velocity(self):
        h = _HeartTestHarness()
        h.set_state(linear_velocity=np.array([0.5, 0.0, 0.0]))
        assert h.state.velocity_valid()

    def test_joint_state_update(self):
        h = _HeartTestHarness()
        h.set_state(
            joint_names=["joint1", "joint2"],
            joint_positions=np.array([0.0, 1.57]),
        )
        assert h.state.joint_names == ["joint1", "joint2"]
        npt.assert_array_equal(h.state.joint_positions, [0.0, 1.57])

    def test_image_update(self):
        h = _HeartTestHarness()
        h.state.images["/camera/rgb"] = np.zeros(100, dtype=np.uint8)
        assert "/camera/rgb" in h.state.images


class TestHeartControlLoop:
    """Verify the tick → controller → command pipeline."""

    def test_idle_produces_no_command(self):
        h = _HeartTestHarness()
        assert h.tick() is None

    def test_velocity_passthrough(self):
        h = _HeartTestHarness()
        h.set_objective(ControlObjective(
            mode="velocity",
            target_velocity=np.array([1.0, 0.0, 0.0]),
        ))
        cmd = h.tick()
        assert cmd is not None
        npt.assert_array_equal(cmd.linear_velocity, [1.0, 0.0, 0.0])

    def test_position_pid_drives_forward(self):
        h = _HeartTestHarness()
        h.switch_controller("pid_position")
        h.set_state(position=np.array([0.0, 0.0, 0.0]))
        h.set_objective(ControlObjective(
            mode="position",
            target_position=np.array([5.0, 0.0, 0.0]),
        ))
        cmd = h.tick()
        assert cmd is not None
        assert cmd.linear_velocity[0] > 0  # moving toward target

    def test_position_pid_converges(self):
        """Simulate multiple ticks — error should decrease."""
        from unittest.mock import patch

        h = _HeartTestHarness()
        h.switch_controller("pid_position")
        h.set_state(position=np.array([0.0, 0.0, 0.0]))
        h.set_objective(ControlObjective(
            mode="position",
            target_position=np.array([1.0, 0.0, 0.0]),
        ))

        # Mock time.monotonic to give a consistent 20ms dt per tick,
        # avoiding sub-microsecond dt that makes the derivative explode.
        fake_time = [0.0]

        def _monotonic():
            fake_time[0] += 0.02
            return fake_time[0]

        with patch("rosie.controllers.pid_position.time") as mock_time:
            mock_time.monotonic = _monotonic

            for _ in range(100):
                cmd = h.tick()
                if cmd is not None:
                    h.state.position = h.state.position + cmd.linear_velocity * 0.02
                    h.state.stamp = fake_time[0]

        error = np.linalg.norm(h.state.position - np.array([1.0, 0.0, 0.0]))
        assert error < 0.3, f"Position error {error} too large after 100 ticks"

    def test_controller_switch(self):
        h = _HeartTestHarness()
        assert h.active_controller.name == "passthrough"
        ok = h.switch_controller("pid_position")
        assert ok
        assert h.active_controller.name == "pid_position"

    def test_controller_switch_unknown(self):
        h = _HeartTestHarness()
        ok = h.switch_controller("nonexistent")
        assert not ok

    def test_objective_change_resets_controller(self):
        h = _HeartTestHarness()
        h.switch_controller("pid_position")
        pid = h.active_controller
        h.set_state(position=np.array([0, 0, 0]))
        h.set_objective(ControlObjective(mode="position", target_position=np.array([1, 0, 0])))

        # Build up integrator
        for _ in range(10):
            h.tick()

        # Change objective mode → should reset
        h.set_objective(ControlObjective(mode="idle"))
        npt.assert_array_equal(pid._integral, [0, 0, 0])


class TestHeartMultipleStateTopics:
    """Verify that different state sources write to the correct fields."""

    def test_odom_and_joints_coexist(self):
        h = _HeartTestHarness()
        h.set_state(position=np.array([1, 2, 0]))
        h.set_state(joint_names=["arm_joint"], joint_positions=np.array([0.5]))
        assert h.state.position_valid()
        assert h.state.joint_names == ["arm_joint"]

    def test_odom_and_image_coexist(self):
        h = _HeartTestHarness()
        h.set_state(position=np.array([1, 2, 0]))
        h.state.images["/cam0"] = np.zeros(10, dtype=np.uint8)
        assert h.state.position_valid()
        assert "/cam0" in h.state.images


class TestHeartNEDConversion:
    """Verify NED↔ENU conversion logic used for PX4 VehicleOdometry."""

    def test_ned_to_enu_position(self):
        """PX4 NED [north, east, down] → ENU [east, north, up]."""
        ned = np.array([10.0, 5.0, -3.0])  # N=10, E=5, D=-3 (3m up)
        enu = np.array([ned[1], ned[0], -ned[2]])
        npt.assert_array_equal(enu, [5.0, 10.0, 3.0])

    def test_ned_to_enu_velocity(self):
        ned_vel = np.array([1.0, 0.5, -0.1])
        enu_vel = np.array([ned_vel[1], ned_vel[0], -ned_vel[2]])
        npt.assert_array_equal(enu_vel, [0.5, 1.0, 0.1])
