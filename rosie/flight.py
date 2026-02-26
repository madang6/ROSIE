"""
Flight state machine and recorder for the ROSIE Heart.

FlightPhase mirrors the lifecycle that FlightCommand uses:
  INIT → READY → ACTIVE → HOLD → LAND

FlightRecorder logs state, commands, and timing at each control tick.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from dataclasses import dataclass, field

import numpy as np

from rosie.state import VehicleState, ControlCommand

log = logging.getLogger("rosie.flight")


class FlightPhase(Enum):
    """Flight lifecycle phases — modelled after FlightCommand's StateMachine."""

    INIT = 0      # Waiting for first state estimate
    READY = 1     # Have state, waiting for attitude/altitude match
    ACTIVE = 2    # Executing objective (trajectory, velocity, learned)
    HOLD = 3      # Position/velocity hold (hover)
    LAND = 4      # Landing sequence


@dataclass
class ReadyGate:
    """Pre-flight checks that must pass before transitioning READY → ACTIVE.

    Mirrors FlightCommand's q0/z0 tolerance checks.
    """

    # Target initial attitude (quaternion [w, x, y, z]) — None = don't check
    target_orientation: np.ndarray | None = None
    orientation_tolerance: float = 0.3   # quaternion norm distance

    # Target initial altitude (z in ENU, metres) — None = don't check
    target_altitude: float | None = None
    altitude_tolerance: float = 0.5      # metres

    # Minimum number of state updates before allowing ACTIVE
    min_state_updates: int = 10

    def check(self, state: VehicleState, update_count: int) -> tuple[bool, str]:
        """Return (passed, reason) — reason explains what's blocking."""
        if update_count < self.min_state_updates:
            return False, f"waiting for state updates ({update_count}/{self.min_state_updates})"

        if not state.position_valid():
            return False, "no position estimate"

        if self.target_altitude is not None:
            alt_error = abs(state.position[2] - self.target_altitude)
            if alt_error > self.altitude_tolerance:
                return False, f"altitude error {alt_error:.2f}m > {self.altitude_tolerance}m"

        if self.target_orientation is not None and state.orientation is not None:
            q_error = np.linalg.norm(state.orientation - self.target_orientation)
            if q_error > self.orientation_tolerance:
                return False, f"orientation error {q_error:.3f} > {self.orientation_tolerance}"

        return True, "ready"


class FlightRecorder:
    """Logs state and command data at each control tick for post-flight analysis.

    Data is stored in pre-allocated numpy arrays for performance, then
    can be exported to a dict for saving.
    """

    def __init__(self, max_ticks: int = 100_000) -> None:
        self._max = max_ticks
        self._k = 0

        # Pre-allocate arrays
        self.timestamps = np.zeros(max_ticks)
        self.positions = np.zeros((max_ticks, 3))
        self.orientations = np.zeros((max_ticks, 4))
        self.linear_velocities = np.zeros((max_ticks, 3))
        self.angular_velocities = np.zeros((max_ticks, 3))

        # Commands
        self.cmd_linear_velocities = np.zeros((max_ticks, 3))
        self.cmd_angular_velocities = np.zeros((max_ticks, 3))
        self.cmd_thrusts = np.zeros(max_ticks)
        self.cmd_body_rates = np.zeros((max_ticks, 3))

        # Timing
        self.loop_times = np.zeros(max_ticks)
        self.phases = np.zeros(max_ticks, dtype=np.int32)

    @property
    def tick_count(self) -> int:
        return self._k

    def record(
        self,
        state: VehicleState,
        cmd: ControlCommand | None,
        phase: FlightPhase,
        loop_time: float = 0.0,
    ) -> None:
        """Record one tick of data."""
        if self._k >= self._max:
            return

        k = self._k
        self.timestamps[k] = state.stamp
        if state.position is not None:
            self.positions[k] = state.position
        if state.orientation is not None:
            self.orientations[k] = state.orientation
        if state.linear_velocity is not None:
            self.linear_velocities[k] = state.linear_velocity
        if state.angular_velocity is not None:
            self.angular_velocities[k] = state.angular_velocity

        if cmd is not None:
            self.cmd_linear_velocities[k] = cmd.linear_velocity
            self.cmd_angular_velocities[k] = cmd.angular_velocity
            if cmd.thrust is not None:
                self.cmd_thrusts[k] = cmd.thrust
            if cmd.body_rates is not None:
                self.cmd_body_rates[k] = cmd.body_rates

        self.loop_times[k] = loop_time
        self.phases[k] = phase.value
        self._k += 1

    def as_dict(self) -> dict:
        """Export recorded data as a trimmed dict (only used ticks)."""
        k = self._k
        return {
            "tick_count": k,
            "timestamps": self.timestamps[:k].tolist(),
            "positions": self.positions[:k].tolist(),
            "orientations": self.orientations[:k].tolist(),
            "linear_velocities": self.linear_velocities[:k].tolist(),
            "angular_velocities": self.angular_velocities[:k].tolist(),
            "cmd_linear_velocities": self.cmd_linear_velocities[:k].tolist(),
            "cmd_angular_velocities": self.cmd_angular_velocities[:k].tolist(),
            "cmd_thrusts": self.cmd_thrusts[:k].tolist(),
            "cmd_body_rates": self.cmd_body_rates[:k].tolist(),
            "loop_times": self.loop_times[:k].tolist(),
            "phases": self.phases[:k].tolist(),
        }

    def summary(self) -> dict:
        """Return summary statistics for the agent."""
        k = self._k
        if k == 0:
            return {"tick_count": 0}

        valid_loop = self.loop_times[:k]
        valid_loop = valid_loop[valid_loop > 0]

        return {
            "tick_count": k,
            "duration_sec": round(float(self.timestamps[k - 1] - self.timestamps[0]), 2),
            "mean_loop_ms": round(float(np.mean(valid_loop) * 1000), 2) if len(valid_loop) > 0 else 0,
            "max_loop_ms": round(float(np.max(valid_loop) * 1000), 2) if len(valid_loop) > 0 else 0,
            "mean_hz": round(float(1.0 / np.mean(valid_loop)), 1) if len(valid_loop) > 0 and np.mean(valid_loop) > 0 else 0,
        }
