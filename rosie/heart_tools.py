"""
Agent tools for interacting with the Heart node.

These tools give the LLM high-level control over the Heart's state
estimation + control loop, without needing to know about raw topic
publishing.
"""

from __future__ import annotations

import json

import numpy as np

from rosie.heart import Heart
from rosie.state import ControlObjective

# The Heart instance is set at startup by the server/agent.
_heart: Heart | None = None


def set_heart(heart: Heart) -> None:
    """Register the Heart instance that tools will operate on."""
    global _heart
    _heart = heart


def _require_heart() -> Heart:
    if _heart is None:
        raise RuntimeError("Heart node is not running.")
    return _heart


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def heart_get_state() -> str:
    """Get the current fused vehicle state from the Heart.

    Returns position, velocity, orientation, joint states, and which image
    topics are active.  Use this instead of reading individual state topics.
    """
    heart = _require_heart()
    return json.dumps(heart.get_state().as_dict(), default=str)


def heart_get_status() -> str:
    """Get the full Heart status: active controller, objective, state, and
    configured topics.
    """
    heart = _require_heart()
    return json.dumps(heart.get_status(), default=str)


def heart_set_velocity(vx: float = 0.0, vy: float = 0.0, vz: float = 0.0, yaw_rate: float = 0.0) -> str:
    """Command the robot to move at a given velocity.

    Sets a velocity objective that the Heart sustains until changed.
    Unlike ros2_publish_repeated, this is *persistent* — the Heart keeps
    publishing until you set a new objective or call heart_stop.
    """
    heart = _require_heart()
    obj = ControlObjective(
        mode="velocity",
        target_velocity=np.array([vx, vy, vz]),
        target_yaw_rate=yaw_rate,
    )
    heart.set_objective(obj)
    heart.switch_controller("passthrough")
    return json.dumps({"status": "velocity_set", "velocity": [vx, vy, vz], "yaw_rate": yaw_rate})


def heart_go_to_position(x: float, y: float, z: float = 0.0, yaw: float | None = None) -> str:
    """Command the robot to move to a target position using PID control.

    The Heart's PID controller will continuously drive the robot toward
    the target, reading state feedback and adjusting commands each tick.
    """
    heart = _require_heart()
    obj = ControlObjective(
        mode="position",
        target_position=np.array([x, y, z]),
        target_yaw=yaw,
    )
    heart.set_objective(obj)
    heart.switch_controller("pid_position")
    return json.dumps({"status": "navigating", "target": [x, y, z], "yaw": yaw})


def heart_stop() -> str:
    """Stop the robot by setting the objective to idle.

    The Heart will stop publishing commands, and the robot should halt
    (assuming it has its own failsafe for missing commands).
    """
    heart = _require_heart()
    heart.set_objective(ControlObjective(mode="idle"))
    return json.dumps({"status": "stopped"})


def heart_switch_controller(controller_name: str) -> str:
    """Switch the active controller.

    Available controllers can be found via heart_get_status.
    """
    heart = _require_heart()
    ok = heart.switch_controller(controller_name)
    if ok:
        return json.dumps({"status": "switched", "controller": controller_name})
    available = heart.get_controller_names()
    return json.dumps({"error": f"Unknown controller '{controller_name}'", "available": available})


def heart_set_trajectory(waypoints: list[list[float]]) -> str:
    """Set a multi-waypoint trajectory for the robot to follow.

    Each waypoint is [x, y, z].  The active controller will drive through
    them sequentially.
    """
    heart = _require_heart()
    wps = [np.array(wp) for wp in waypoints]
    obj = ControlObjective(
        mode="trajectory",
        waypoints=wps,
        target_position=wps[0] if wps else None,
    )
    heart.set_objective(obj)
    heart.switch_controller("pid_position")
    return json.dumps({"status": "trajectory_set", "waypoint_count": len(wps)})
