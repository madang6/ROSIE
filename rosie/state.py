"""
Canonical vehicle state representation.

The Heart node normalises every supported state-estimation source into this
single dataclass so that controllers never need to know whether the upstream
is PX4 VehicleOdometry, nav_msgs/Odometry, or something else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VehicleState:
    """Fused, canonical state estimate — always in a right-handed ENU frame.

    Fields are populated incrementally as topics arrive.  A ``None`` value
    means that particular measurement has not yet been received.
    """

    # Timestamp (seconds, monotonic clock at reception)
    stamp: float = 0.0

    # --- Pose ---
    position: np.ndarray | None = None          # [x, y, z]  metres
    orientation: np.ndarray | None = None        # quaternion [w, x, y, z]

    # --- Twist ---
    linear_velocity: np.ndarray | None = None    # [vx, vy, vz]  m/s
    angular_velocity: np.ndarray | None = None   # [wx, wy, wz]  rad/s

    # --- Joints (manipulators) ---
    joint_names: list[str] = field(default_factory=list)
    joint_positions: np.ndarray | None = None    # rad or m
    joint_velocities: np.ndarray | None = None   # rad/s or m/s

    # --- Images (cameras) ---
    images: dict[str, np.ndarray] = field(default_factory=dict)

    def age(self) -> float:
        """Seconds since the last state update."""
        return time.monotonic() - self.stamp if self.stamp else float("inf")

    def position_valid(self) -> bool:
        return self.position is not None

    def velocity_valid(self) -> bool:
        return self.linear_velocity is not None

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot (for the agent tools)."""
        def _arr(a: np.ndarray | None) -> list | None:
            return a.tolist() if a is not None else None

        return {
            "stamp": self.stamp,
            "age_sec": round(self.age(), 4),
            "position": _arr(self.position),
            "orientation": _arr(self.orientation),
            "linear_velocity": _arr(self.linear_velocity),
            "angular_velocity": _arr(self.angular_velocity),
            "joint_names": self.joint_names,
            "joint_positions": _arr(self.joint_positions),
            "joint_velocities": _arr(self.joint_velocities),
            "image_topics": list(self.images.keys()),
        }


@dataclass
class ControlObjective:
    """What the agent (or user) wants the robot to do.

    The Heart feeds this + the current VehicleState to the active controller
    every control tick.
    """

    # One of: "idle", "position", "velocity", "trajectory", "learned"
    mode: str = "idle"

    # Position hold / go-to
    target_position: np.ndarray | None = None       # [x, y, z]
    target_yaw: float | None = None                  # rad

    # Velocity command
    target_velocity: np.ndarray | None = None        # [vx, vy, vz]
    target_yaw_rate: float | None = None             # rad/s

    # Waypoint trajectory
    waypoints: list[np.ndarray] = field(default_factory=list)  # list of [x,y,z]

    # Learned controller model identifier
    model_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        def _arr(a: np.ndarray | None) -> list | None:
            return a.tolist() if a is not None else None

        return {
            "mode": self.mode,
            "target_position": _arr(self.target_position),
            "target_yaw": self.target_yaw,
            "target_velocity": _arr(self.target_velocity),
            "target_yaw_rate": self.target_yaw_rate,
            "waypoints": [w.tolist() for w in self.waypoints],
            "model_name": self.model_name,
        }


@dataclass
class ControlCommand:
    """Output of a controller — what actually gets published.

    The Heart translates this into the platform-specific message (Twist,
    TrajectorySetpoint, JointTrajectory, etc.).
    """

    # Velocity-level command (most universal)
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Position-level command (PX4 TrajectorySetpoint, MoveIt)
    position: np.ndarray | None = None
    yaw: float | None = None

    # Raw joint commands (manipulators)
    joint_positions: np.ndarray | None = None
    joint_velocities: np.ndarray | None = None
