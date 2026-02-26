"""
Tests for rosie.controllers — passthrough, PID, learned.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from rosie.state import VehicleState, ControlObjective, ControlCommand
from rosie.controllers.passthrough import PassthroughController
from rosie.controllers.pid_position import PIDPositionController, _quat_to_yaw, _wrap_angle
from rosie.controllers.learned import (
    LearnedController,
    default_obs_builder,
    default_action_interpreter,
)


# ── Passthrough ──────────────────────────────────────────────────────────────


class TestPassthroughController:
    def test_name(self):
        assert PassthroughController().name == "passthrough"

    def test_velocity_mode(self):
        ctrl = PassthroughController()
        state = VehicleState()
        obj = ControlObjective(
            mode="velocity",
            target_velocity=np.array([1.0, 0.0, 0.0]),
            target_yaw_rate=0.5,
        )
        cmd = ctrl.compute(state, obj)
        npt.assert_array_equal(cmd.linear_velocity, [1.0, 0.0, 0.0])
        assert cmd.angular_velocity[2] == 0.5

    def test_position_mode(self):
        ctrl = PassthroughController()
        state = VehicleState()
        obj = ControlObjective(
            mode="position",
            target_position=np.array([10.0, 5.0, 0.0]),
            target_yaw=1.57,
        )
        cmd = ctrl.compute(state, obj)
        npt.assert_array_equal(cmd.position, [10.0, 5.0, 0.0])
        assert cmd.yaw == 1.57

    def test_idle_returns_zero(self):
        ctrl = PassthroughController()
        cmd = ctrl.compute(VehicleState(), ControlObjective(mode="idle"))
        npt.assert_array_equal(cmd.linear_velocity, [0, 0, 0])


# ── PID Position ─────────────────────────────────────────────────────────────


class TestPIDPositionController:
    def test_name(self):
        assert PIDPositionController().name == "pid_position"

    def test_drives_toward_target(self):
        ctrl = PIDPositionController(kp=1.0, ki=0.0, kd=0.0, max_velocity=5.0)
        state = VehicleState(position=np.array([0.0, 0.0, 0.0]))
        obj = ControlObjective(mode="position", target_position=np.array([3.0, 0.0, 0.0]))

        cmd = ctrl.compute(state, obj)
        # Velocity should be positive in x
        assert cmd.linear_velocity[0] > 0
        assert cmd.linear_velocity[1] == 0.0

    def test_stops_at_target(self):
        ctrl = PIDPositionController(kp=1.0, ki=0.0, kd=0.0)
        state = VehicleState(position=np.array([5.0, 5.0, 0.0]))
        obj = ControlObjective(mode="position", target_position=np.array([5.0, 5.0, 0.0]))

        cmd = ctrl.compute(state, obj)
        npt.assert_allclose(cmd.linear_velocity, [0, 0, 0], atol=1e-6)

    def test_velocity_clamped(self):
        ctrl = PIDPositionController(kp=100.0, ki=0.0, kd=0.0, max_velocity=1.0)
        state = VehicleState(position=np.array([0.0, 0.0, 0.0]))
        obj = ControlObjective(mode="position", target_position=np.array([100.0, 0.0, 0.0]))

        cmd = ctrl.compute(state, obj)
        speed = np.linalg.norm(cmd.linear_velocity)
        assert speed <= 1.0 + 1e-6

    def test_reset_clears_integrator(self):
        ctrl = PIDPositionController(ki=1.0)
        state = VehicleState(position=np.array([0.0, 0.0, 0.0]))
        obj = ControlObjective(mode="position", target_position=np.array([1.0, 0.0, 0.0]))

        # Run a few iterations to build up integrator
        for _ in range(5):
            ctrl.compute(state, obj)

        ctrl.reset()
        npt.assert_array_equal(ctrl._integral, [0, 0, 0])

    def test_no_command_without_state(self):
        ctrl = PIDPositionController()
        state = VehicleState()  # no position
        obj = ControlObjective(mode="position", target_position=np.array([1, 0, 0]))
        cmd = ctrl.compute(state, obj)
        npt.assert_array_equal(cmd.linear_velocity, [0, 0, 0])

    def test_yaw_control(self):
        ctrl = PIDPositionController()
        state = VehicleState(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # yaw = 0
        )
        obj = ControlObjective(
            mode="position",
            target_position=np.array([0.0, 0.0, 0.0]),
            target_yaw=1.0,  # turn to 1 rad
        )
        cmd = ctrl.compute(state, obj)
        assert cmd.angular_velocity[2] > 0  # should turn positive


class TestYawHelpers:
    def test_identity_quat_is_zero_yaw(self):
        assert abs(_quat_to_yaw(np.array([1, 0, 0, 0]))) < 1e-6

    def test_wrap_angle(self):
        assert abs(_wrap_angle(0.0)) < 1e-6
        assert abs(_wrap_angle(2 * np.pi)) < 1e-6
        assert abs(_wrap_angle(-np.pi) - (-np.pi)) < 1e-6


# ── Learned ──────────────────────────────────────────────────────────────────


class TestLearnedController:
    def test_callable_model(self):
        """A simple callable that returns a fixed action."""
        def dummy_model(obs: np.ndarray) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5])

        ctrl = LearnedController(model=dummy_model)
        assert ctrl.name == "learned"

        state = VehicleState(position=np.array([0, 0, 0]))
        obj = ControlObjective(mode="learned")
        cmd = ctrl.compute(state, obj)
        assert cmd.linear_velocity[0] == 1.0
        assert cmd.angular_velocity[2] == 0.5

    def test_custom_obs_builder(self):
        calls = []

        def custom_obs(state, obj):
            calls.append((state, obj))
            return np.zeros(6)

        def dummy_model(obs):
            return np.zeros(6)

        ctrl = LearnedController(model=dummy_model, obs_builder=custom_obs)
        ctrl.compute(VehicleState(), ControlObjective())
        assert len(calls) == 1

    def test_custom_action_interpreter(self):
        def dummy_model(obs):
            return np.array([42.0])

        def custom_interp(action):
            cmd = ControlCommand()
            cmd.linear_velocity = np.array([action[0], 0, 0])
            return cmd

        ctrl = LearnedController(model=dummy_model, action_interpreter=custom_interp)
        cmd = ctrl.compute(VehicleState(), ControlObjective())
        assert cmd.linear_velocity[0] == 42.0


class TestDefaultObsBuilder:
    def test_output_shape(self):
        state = VehicleState(position=np.array([1, 2, 3]))
        obj = ControlObjective(target_position=np.array([4, 5, 6]))
        obs = default_obs_builder(state, obj)
        assert obs.shape == (19,)
        assert obs.dtype == np.float32

    def test_zeros_when_empty(self):
        obs = default_obs_builder(VehicleState(), ControlObjective())
        assert obs.shape == (19,)
        npt.assert_array_equal(obs[:3], [0, 0, 0])


class TestDefaultActionInterpreter:
    def test_six_dim_action(self):
        action = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        cmd = default_action_interpreter(action)
        npt.assert_array_equal(cmd.linear_velocity, [1, 2, 3])
        npt.assert_array_equal(cmd.angular_velocity, [0.1, 0.2, 0.3])

    def test_three_dim_action(self):
        action = np.array([1.0, 0.0, 0.0])
        cmd = default_action_interpreter(action)
        npt.assert_array_equal(cmd.linear_velocity, [1, 0, 0])
