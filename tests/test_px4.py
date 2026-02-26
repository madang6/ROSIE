"""
Tests for PX4 offboard protocol support in the Heart.

Tests cover:
  - OffboardControlMode publishing every tick
  - VehicleCommand for arm/disarm/offboard mode
  - TrajectorySetpoint velocity-mode (NaN position)
  - TrajectorySetpoint position-mode (NaN velocity)
  - Full engage sequence
  - PX4 agent tools
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import numpy.testing as npt
import pytest

from rosie.state import VehicleState, ControlObjective, ControlCommand
from rosie.controllers.passthrough import PassthroughController
from rosie.controllers.pid_position import PIDPositionController
import rosie.heart_tools as ht


# ---------------------------------------------------------------------------
# PX4 heart test harness — mirrors the Heart's PX4 behaviour without rclpy
# ---------------------------------------------------------------------------


class _PX4HeartHarness:
    """Simulates the Heart's PX4 offboard protocol for testing."""

    def __init__(self):
        self.state = VehicleState()
        self.objective = ControlObjective(mode="idle")
        self.controllers = {
            "passthrough": PassthroughController(),
            "pid_position": PIDPositionController(kp=1.0, ki=0.0, kd=0.1),
        }
        self.active_controller = self.controllers["passthrough"]

        # PX4 protocol state
        self.px4_armed = False
        self.px4_offboard_mode = False
        self.px4_offboard_heartbeat_count = 0

        # Published messages (for assertions)
        self.offboard_ctrl_msgs: list[dict] = []
        self.vehicle_cmds: list[dict] = []
        self.trajectory_setpoints: list[dict] = []

    def set_objective(self, obj: ControlObjective):
        self.objective = obj

    def switch_controller(self, name: str) -> bool:
        ctrl = self.controllers.get(name)
        if ctrl is None:
            return False
        self.active_controller = ctrl
        return True

    def px4_arm(self):
        self.vehicle_cmds.append({"command": 400, "param1": 1.0})
        self.px4_armed = True

    def px4_disarm(self):
        self.vehicle_cmds.append({"command": 400, "param1": 0.0})
        self.px4_armed = False

    def px4_set_offboard_mode(self):
        self.vehicle_cmds.append({"command": 176, "param1": 1.0, "param2": 6.0})
        self.px4_offboard_mode = True

    def px4_engage(self):
        self.px4_set_offboard_mode()
        self.px4_arm()

    def tick(self) -> dict | None:
        """Simulate one Heart control tick with PX4 protocol."""
        if self.objective.mode == "idle":
            return None

        cmd = self.active_controller.compute(self.state, self.objective)

        # Offboard heartbeat
        has_position = cmd.position is not None
        ocm = {
            "position": has_position,
            "velocity": not has_position,
            "acceleration": False,
            "attitude": False,
            "body_rate": False,
        }
        self.offboard_ctrl_msgs.append(ocm)
        self.px4_offboard_heartbeat_count += 1

        # TrajectorySetpoint — mimics _publish_px4_trajectory_setpoint
        tsp = {}
        if cmd.position is not None:
            # ENU → NED
            tsp["position"] = [
                float(cmd.position[1]),
                float(cmd.position[0]),
                float(-cmd.position[2]),
            ]
            tsp["velocity"] = [float("nan")] * 3
            tsp["yaw"] = float(cmd.yaw) if cmd.yaw is not None else float("nan")
        else:
            # Velocity mode — ENU → NED
            tsp["velocity"] = [
                float(cmd.linear_velocity[1]),
                float(cmd.linear_velocity[0]),
                float(-cmd.linear_velocity[2]),
            ]
            tsp["position"] = [float("nan")] * 3
            tsp["yaw"] = float("nan")
            if cmd.angular_velocity is not None and len(cmd.angular_velocity) >= 3:
                tsp["yawspeed"] = float(cmd.angular_velocity[2])

        self.trajectory_setpoints.append(tsp)
        return tsp

    def get_controller_names(self):
        return list(self.controllers.keys())

    def get_state(self):
        return self.state

    def get_objective(self):
        return self.objective

    def get_status(self):
        return {
            "controller": self.active_controller.name,
            "available_controllers": self.get_controller_names(),
            "objective": self.objective.as_dict(),
            "state": self.state.as_dict(),
            "cmd_vel_topic": None,
            "trajectory_setpoint_topic": "/fmu/in/trajectory_setpoint",
            "px4": {
                "offboard_enabled": True,
                "armed": self.px4_armed,
                "offboard_mode": self.px4_offboard_mode,
                "heartbeat_count": self.px4_offboard_heartbeat_count,
            },
        }


# ---------------------------------------------------------------------------
# Offboard heartbeat tests
# ---------------------------------------------------------------------------


class TestPX4OffboardHeartbeat:
    def test_heartbeat_published_every_tick(self):
        h = _PX4HeartHarness()
        h.state = VehicleState(stamp=time.monotonic(), position=np.zeros(3))
        h.set_objective(ControlObjective(
            mode="velocity",
            target_velocity=np.array([1.0, 0.0, 0.0]),
        ))

        for _ in range(10):
            h.tick()

        assert len(h.offboard_ctrl_msgs) == 10
        assert h.px4_offboard_heartbeat_count == 10

    def test_heartbeat_velocity_mode_flags(self):
        h = _PX4HeartHarness()
        h.set_objective(ControlObjective(
            mode="velocity",
            target_velocity=np.array([1.0, 0.0, 0.0]),
        ))
        h.tick()

        ocm = h.offboard_ctrl_msgs[0]
        assert ocm["velocity"] is True
        assert ocm["position"] is False

    def test_heartbeat_position_mode_flags(self):
        h = _PX4HeartHarness()
        h.set_objective(ControlObjective(
            mode="position",
            target_position=np.array([5.0, 0.0, 0.0]),
        ))
        # Passthrough in position mode gives cmd.position
        h.tick()

        ocm = h.offboard_ctrl_msgs[0]
        assert ocm["position"] is True
        assert ocm["velocity"] is False

    def test_idle_produces_no_heartbeat(self):
        h = _PX4HeartHarness()
        h.set_objective(ControlObjective(mode="idle"))
        result = h.tick()
        assert result is None
        assert len(h.offboard_ctrl_msgs) == 0


# ---------------------------------------------------------------------------
# TrajectorySetpoint tests
# ---------------------------------------------------------------------------


class TestPX4TrajectorySetpoint:
    def test_velocity_mode_has_nan_position(self):
        h = _PX4HeartHarness()
        h.set_objective(ControlObjective(
            mode="velocity",
            target_velocity=np.array([1.0, 0.0, 0.0]),  # ENU: east 1 m/s
        ))
        tsp = h.tick()

        # Position should be NaN in velocity mode
        assert all(np.isnan(p) for p in tsp["position"])
        # Velocity: ENU [1,0,0] → NED [0,1,0]  (east→north swap)
        assert tsp["velocity"][0] == 0.0  # north (from vy=0)
        assert tsp["velocity"][1] == 1.0  # east (from vx=1)
        assert tsp["velocity"][2] == 0.0  # down (from -vz=0)

    def test_position_mode_has_nan_velocity(self):
        h = _PX4HeartHarness()
        h.set_objective(ControlObjective(
            mode="position",
            target_position=np.array([5.0, 10.0, -3.0]),  # ENU
            target_yaw=1.57,
        ))
        tsp = h.tick()

        # Velocity should be NaN in position mode
        assert all(np.isnan(v) for v in tsp["velocity"])
        # Position: ENU [5,10,-3] → NED [10,5,3]
        assert tsp["position"][0] == 10.0   # north (from y=10)
        assert tsp["position"][1] == 5.0    # east (from x=5)
        assert tsp["position"][2] == 3.0    # down (from -z=3)
        assert tsp["yaw"] == 1.57

    def test_velocity_yaw_rate(self):
        h = _PX4HeartHarness()
        h.set_objective(ControlObjective(
            mode="velocity",
            target_velocity=np.array([0.0, 0.0, 0.0]),
            target_yaw_rate=0.5,
        ))
        tsp = h.tick()
        assert tsp["yawspeed"] == 0.5

    def test_position_no_yaw_is_nan(self):
        h = _PX4HeartHarness()
        h.set_objective(ControlObjective(
            mode="position",
            target_position=np.array([0.0, 0.0, 0.0]),
        ))
        tsp = h.tick()
        assert np.isnan(tsp["yaw"])


# ---------------------------------------------------------------------------
# Arm / Disarm / Offboard mode
# ---------------------------------------------------------------------------


class TestPX4VehicleCommands:
    def test_arm_command(self):
        h = _PX4HeartHarness()
        h.px4_arm()
        assert h.px4_armed is True
        assert h.vehicle_cmds[-1] == {"command": 400, "param1": 1.0}

    def test_disarm_command(self):
        h = _PX4HeartHarness()
        h.px4_arm()
        h.px4_disarm()
        assert h.px4_armed is False
        assert h.vehicle_cmds[-1] == {"command": 400, "param1": 0.0}

    def test_offboard_mode_command(self):
        h = _PX4HeartHarness()
        h.px4_set_offboard_mode()
        assert h.px4_offboard_mode is True
        assert h.vehicle_cmds[-1] == {"command": 176, "param1": 1.0, "param2": 6.0}

    def test_engage_sends_both_commands(self):
        h = _PX4HeartHarness()
        h.px4_engage()
        assert h.px4_armed is True
        assert h.px4_offboard_mode is True
        assert len(h.vehicle_cmds) == 2
        # First: offboard mode, second: arm
        assert h.vehicle_cmds[0]["command"] == 176
        assert h.vehicle_cmds[1]["command"] == 400


# ---------------------------------------------------------------------------
# Status includes PX4 fields
# ---------------------------------------------------------------------------


class TestPX4Status:
    def test_status_includes_px4_section(self):
        h = _PX4HeartHarness()
        status = h.get_status()
        assert "px4" in status
        assert status["px4"]["offboard_enabled"] is True
        assert status["px4"]["armed"] is False

    def test_status_after_engage(self):
        h = _PX4HeartHarness()
        h.px4_engage()
        status = h.get_status()
        assert status["px4"]["armed"] is True
        assert status["px4"]["offboard_mode"] is True


# ---------------------------------------------------------------------------
# PX4 agent tools
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _px4_heart():
    heart = _PX4HeartHarness()
    ht.set_heart(heart)
    yield heart
    ht.set_heart(None)


class TestPX4Tools:
    def test_px4_arm(self, _px4_heart):
        result = json.loads(ht.px4_arm())
        assert result["status"] == "armed"
        assert _px4_heart.px4_armed is True

    def test_px4_disarm(self, _px4_heart):
        ht.px4_arm()
        result = json.loads(ht.px4_disarm())
        assert result["status"] == "disarmed"
        assert _px4_heart.px4_armed is False

    def test_px4_offboard_mode(self, _px4_heart):
        result = json.loads(ht.px4_offboard_mode())
        assert result["status"] == "offboard_mode_set"
        assert _px4_heart.px4_offboard_mode is True

    def test_px4_engage(self, _px4_heart):
        result = json.loads(ht.px4_engage())
        assert result["status"] == "engaged"
        assert _px4_heart.px4_armed is True
        assert _px4_heart.px4_offboard_mode is True

    def test_status_shows_px4(self, _px4_heart):
        ht.px4_engage()
        result = json.loads(ht.heart_get_status())
        assert result["px4"]["armed"] is True


# ---------------------------------------------------------------------------
# Full PX4 offboard flight sequence (integration-style)
# ---------------------------------------------------------------------------


class TestPX4OffboardSequence:
    """Simulate the typical PX4 offboard workflow end-to-end."""

    def test_full_offboard_sequence(self, _px4_heart):
        h = _px4_heart
        h.state = VehicleState(
            stamp=time.monotonic(),
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )

        # Step 1: Start streaming setpoints (hover in place)
        h.set_objective(ControlObjective(
            mode="velocity",
            target_velocity=np.array([0.0, 0.0, 0.0]),
        ))

        # Stream for a few ticks to build up heartbeats
        for _ in range(10):
            h.tick()
        assert h.px4_offboard_heartbeat_count == 10

        # Step 2: Engage (offboard + arm)
        h.px4_engage()
        assert h.px4_armed
        assert h.px4_offboard_mode

        # Step 3: Command to fly to a position
        h.switch_controller("passthrough")
        h.set_objective(ControlObjective(
            mode="position",
            target_position=np.array([5.0, 5.0, -10.0]),  # 10m up in ENU
            target_yaw=0.0,
        ))
        tsp = h.tick()

        # Verify NED conversion: ENU [5, 5, -10] → NED [5, 5, 10]
        assert tsp["position"] == [5.0, 5.0, 10.0]
        assert tsp["yaw"] == 0.0

        # Step 4: Stop and disarm
        h.set_objective(ControlObjective(mode="idle"))
        assert h.tick() is None
        h.px4_disarm()
        assert not h.px4_armed
