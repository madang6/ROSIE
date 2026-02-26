"""
PID position controller — feedback-driven go-to-position.

Computes velocity commands to drive the robot toward a target position
using independent PID loops on each axis.  Suitable for ground robots,
drones (via velocity setpoints), and simple manipulator end-effector
positioning.
"""

from __future__ import annotations

import time

import numpy as np

from rosie.controllers.base import Controller
from rosie.state import VehicleState, ControlObjective, ControlCommand


class PIDPositionController(Controller):
    """3-axis PID position controller with configurable gains."""

    def __init__(
        self,
        kp: np.ndarray | float = 1.0,
        ki: np.ndarray | float = 0.0,
        kd: np.ndarray | float = 0.1,
        max_velocity: float = 1.0,
    ) -> None:
        self._kp = np.atleast_1d(np.broadcast_to(np.asarray(kp, dtype=float), (3,))).copy()
        self._ki = np.atleast_1d(np.broadcast_to(np.asarray(ki, dtype=float), (3,))).copy()
        self._kd = np.atleast_1d(np.broadcast_to(np.asarray(kd, dtype=float), (3,))).copy()
        self._max_vel = max_velocity

        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._prev_time: float | None = None

    @property
    def name(self) -> str:
        return "pid_position"

    def reset(self) -> None:
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._prev_time = None

    def compute(
        self,
        state: VehicleState,
        objective: ControlObjective,
    ) -> ControlCommand:
        cmd = ControlCommand()

        if objective.mode != "position" or objective.target_position is None:
            return cmd
        if not state.position_valid():
            return cmd

        now = time.monotonic()
        dt = (now - self._prev_time) if self._prev_time is not None else 0.02
        dt = np.clip(dt, 1e-4, 0.5)  # guard against jumps
        self._prev_time = now

        error = objective.target_position - state.position

        # PID
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        self._prev_error = error.copy()

        vel = self._kp * error + self._ki * self._integral + self._kd * derivative

        # Clamp magnitude
        speed = np.linalg.norm(vel)
        if speed > self._max_vel:
            vel = vel * (self._max_vel / speed)

        cmd.linear_velocity = vel

        # Yaw control (simple P)
        if objective.target_yaw is not None and state.orientation is not None:
            current_yaw = _quat_to_yaw(state.orientation)
            yaw_error = _wrap_angle(objective.target_yaw - current_yaw)
            cmd.angular_velocity = np.array([0.0, 0.0, 2.0 * yaw_error])

        return cmd


def _quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw from a [w, x, y, z] quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)
