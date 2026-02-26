"""
Agent tools for interacting with the Heart node.

These tools give the LLM high-level control over the Heart's state
estimation + control loop, without needing to know about raw topic
publishing.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np

from rosie.heart import Heart
from rosie.state import ControlObjective

log = logging.getLogger("rosie.heart_tools")

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


# ---------------------------------------------------------------------------
# PX4-specific tools
# ---------------------------------------------------------------------------


def px4_arm() -> str:
    """Arm the PX4 flight controller.

    Must be called after px4_engage or px4_offboard_mode.  The drone
    will not move until armed.
    """
    heart = _require_heart()
    heart.px4_arm()
    return json.dumps({"status": "armed"})


def px4_disarm() -> str:
    """Disarm the PX4 flight controller.

    Motors will stop.  Only call this when the drone is landed.
    """
    heart = _require_heart()
    heart.px4_disarm()
    return json.dumps({"status": "disarmed"})


def px4_offboard_mode() -> str:
    """Switch PX4 to OFFBOARD flight mode.

    The Heart must already be streaming OffboardControlMode (which it
    does automatically every tick when any objective is set).  After this
    call, PX4 will accept TrajectorySetpoint commands from the Heart.
    """
    heart = _require_heart()
    heart.px4_set_offboard_mode()
    return json.dumps({"status": "offboard_mode_set"})


def px4_engage() -> str:
    """Full offboard engagement: switch to OFFBOARD mode + ARM.

    This is the single command to go from idle to flying.  After this,
    set a velocity or position objective to move the drone.  Typical
    PX4 offboard sequence:

    1. heart_set_velocity(vz=0)   — start streaming setpoints
    2. px4_engage()                — switch mode + arm
    3. heart_go_to_position(...)   — fly somewhere
    4. heart_stop() + px4_disarm() — land and stop
    """
    heart = _require_heart()
    heart.px4_engage()
    return json.dumps({"status": "engaged", "armed": True, "offboard_mode": True})


# ---------------------------------------------------------------------------
# Configuration tools
# ---------------------------------------------------------------------------


def heart_configure(
    odom_topic: str | None = None,
    vehicle_odom_topic: str | None = None,
    image_topics: list[str] | None = None,
    joint_state_topic: str | None = None,
    cmd_vel_topic: str | None = None,
    trajectory_setpoint_topic: str | None = None,
    vehicle_rates_topic: str | None = None,
    px4_offboard: bool | None = None,
) -> str:
    """Reconfigure the Heart's topic subscriptions and publishers at runtime.

    Call this after discovering the robot's ROS 2 graph to connect the
    Heart to the correct state/command topics.  All parameters are optional;
    pass only the topics you want to subscribe/publish to (None = disabled).

    This resets the Heart's internal state and flight phase back to INIT.
    """
    heart = _require_heart()
    config = heart.reconfigure(
        odom_topic=odom_topic,
        vehicle_odom_topic=vehicle_odom_topic,
        image_topics=image_topics,
        joint_state_topic=joint_state_topic,
        cmd_vel_topic=cmd_vel_topic,
        trajectory_setpoint_topic=trajectory_setpoint_topic,
        vehicle_rates_topic=vehicle_rates_topic,
        px4_offboard=px4_offboard,
    )
    return json.dumps({"status": "reconfigured", "config": config})


# ---------------------------------------------------------------------------
# Auto-discovery and bootstrap
# ---------------------------------------------------------------------------

# Known topic patterns → what they mean
_TOPIC_SIGNATURES: list[tuple[str, str, str]] = [
    # (regex pattern, msg type substring, config key)
    (r"/fmu/out/vehicle_odometry", "VehicleOdometry", "vehicle_odom_topic"),
    (r"/odom\b", "Odometry", "odom_topic"),
    (r"/joint_states\b", "JointState", "joint_state_topic"),
]

_IMAGE_PATTERN = re.compile(r".*/image.*/compressed|.*/image_raw/compressed", re.IGNORECASE)

# PX4 control topics
_PX4_CONTROL_PATTERNS = {
    "trajectory_setpoint_topic": re.compile(r"/fmu/in/trajectory_setpoint"),
    "vehicle_rates_topic": re.compile(r"/fmu/in/vehicle_rates_setpoint"),
}


def _parse_topic_list(raw: str) -> list[tuple[str, str]]:
    """Parse ``ros2 topic list -t`` output into [(topic, type), ...]."""
    result = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "/topic_name [msg/Type]" or "/topic_name [msg/Type1, msg/Type2]"
        match = re.match(r"^(\S+)\s+\[(.+)]$", line)
        if match:
            result.append((match.group(1), match.group(2).split(",")[0].strip()))
        else:
            # Might be just the topic name without brackets
            parts = line.split()
            if parts:
                result.append((parts[0], ""))
    return result


def _identify_robot(topics: list[tuple[str, str]]) -> dict:
    """Analyse a topic list and return a Heart configuration dict.

    Returns a dict with keys matching Heart.reconfigure() kwargs, plus
    metadata about what was detected.
    """
    topic_names = {t for t, _ in topics}
    topic_map = {t: typ for t, typ in topics}

    config: dict = {
        "odom_topic": None,
        "vehicle_odom_topic": None,
        "image_topics": [],
        "joint_state_topic": None,
        "cmd_vel_topic": None,
        "trajectory_setpoint_topic": None,
        "vehicle_rates_topic": None,
        "px4_offboard": False,
    }
    detected: list[str] = []  # human-readable descriptions

    # --- State topics ---
    for topic, typ in topics:
        # PX4 vehicle odometry
        if re.search(r"/fmu/out/vehicle_odometry", topic):
            config["vehicle_odom_topic"] = topic
            detected.append(f"PX4 odometry: {topic}")
        # Standard nav_msgs/Odometry
        elif "Odometry" in typ and config["odom_topic"] is None:
            config["odom_topic"] = topic
            detected.append(f"Odometry: {topic}")
        # Joint states
        elif "JointState" in typ:
            config["joint_state_topic"] = topic
            detected.append(f"Joint states: {topic}")
        # Compressed images
        elif _IMAGE_PATTERN.match(topic):
            config["image_topics"].append(topic)
            detected.append(f"Camera: {topic}")

    # --- Control output topics ---
    for topic, typ in topics:
        # cmd_vel
        if re.search(r"/cmd_vel\b", topic) and "Twist" in typ:
            config["cmd_vel_topic"] = topic
            detected.append(f"Velocity command: {topic}")
        # PX4 control inputs
        for key, pattern in _PX4_CONTROL_PATTERNS.items():
            if pattern.search(topic):
                config[key] = topic
                detected.append(f"PX4 control: {topic}")

    # --- Infer PX4 offboard ---
    if config["trajectory_setpoint_topic"] or config["vehicle_rates_topic"]:
        config["px4_offboard"] = True
        detected.append("PX4 offboard mode enabled")

    # --- Infer robot type ---
    robot_type = "unknown"
    if config["px4_offboard"] or config["vehicle_odom_topic"]:
        robot_type = "px4_drone"
    elif config["joint_state_topic"] and not config["odom_topic"]:
        robot_type = "manipulator"
    elif config["joint_state_topic"] and config["odom_topic"]:
        robot_type = "mobile_manipulator"
    elif config["odom_topic"]:
        robot_type = "ground_robot"

    return {
        "config": config,
        "robot_type": robot_type,
        "detected": detected,
    }


def heart_auto_configure() -> str:
    """Automatically discover the robot and configure the Heart.

    Scans the ROS 2 topic graph, identifies the robot type from known
    topic patterns (PX4 drone, ground robot, manipulator, etc.), and
    reconfigures the Heart's subscriptions and publishers to match.

    Call this once at startup before issuing any control commands.
    """
    from rosie.ros2_tools import ros2_list_topics

    heart = _require_heart()

    # 1. Scan the topic graph
    raw = ros2_list_topics()
    if raw.startswith("[error]"):
        return json.dumps({"error": f"Failed to list topics: {raw}"})

    topics = _parse_topic_list(raw)
    if not topics:
        return json.dumps({"error": "No topics found on the ROS 2 graph."})

    # 2. Identify robot type and build config
    result = _identify_robot(topics)
    config = result["config"]

    # 3. Apply to Heart
    heart.reconfigure(
        odom_topic=config["odom_topic"],
        vehicle_odom_topic=config["vehicle_odom_topic"],
        image_topics=config["image_topics"] or None,
        joint_state_topic=config["joint_state_topic"],
        cmd_vel_topic=config["cmd_vel_topic"],
        trajectory_setpoint_topic=config["trajectory_setpoint_topic"],
        vehicle_rates_topic=config["vehicle_rates_topic"],
        px4_offboard=config["px4_offboard"],
    )

    log.info(
        "Auto-configured Heart for %s: %s",
        result["robot_type"],
        result["detected"],
    )

    return json.dumps({
        "status": "auto_configured",
        "robot_type": result["robot_type"],
        "detected": result["detected"],
        "config": config,
        "topic_count": len(topics),
    })
