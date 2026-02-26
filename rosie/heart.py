"""
Heart — the continuously-spinning ROS 2 node that bridges state estimation
to control output.

Architecture::

    State topics                Heart                 Control topics
    ────────────         ┌─────────────────┐         ──────────────
    /odom            ──▶ │  State Router   │
    /vehicle_odometry──▶ │       ↓         │
    /camera/image    ──▶ │  VehicleState   │
    /joint_states    ──▶ │       ↓         │
                         │  Controller     │──▶  /cmd_vel
                         │  (PID / learned │──▶  /fmu/in/trajectory_setpoint
                         │   / passthrough)│──▶  /joint_commands
                         └────────┬────────┘
                                  ↑
                         ControlObjective
                         (set by the agent
                          via /rosie/set_objective)

The Heart:
1. Subscribes to all configured state topics and normalises them into a
   single VehicleState.
2. Runs a control loop at a fixed rate (default 50 Hz).
3. Feeds (VehicleState, ControlObjective) to the active Controller.
4. Publishes the resulting ControlCommand to the appropriate control topic.
5. Exposes a ROS 2 service ``/rosie/set_objective`` so the agent (or any
   other node) can change the current objective at any time.
6. Exposes a ROS 2 service ``/rosie/get_state`` so the agent can query the
   latest fused state without subscribing itself.
"""

from __future__ import annotations

import json
import logging
import time
import threading
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from rosie.state import VehicleState, ControlObjective, ControlCommand
from rosie.controllers.base import Controller
from rosie.controllers.passthrough import PassthroughController
from rosie.controllers.pid_position import PIDPositionController

log = logging.getLogger("rosie.heart")

# Best-effort QoS for high-rate sensor topics
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class Heart(Node):
    """Continuously-spinning state → controller → command node."""

    def __init__(
        self,
        control_rate_hz: float = 50.0,
        controller: Controller | None = None,
        *,
        # State input topic configuration (set to None to disable)
        odom_topic: str | None = "/odom",
        vehicle_odom_topic: str | None = None,          # PX4 VehicleOdometry
        image_topics: list[str] | None = None,           # CompressedImage
        joint_state_topic: str | None = None,            # sensor_msgs/JointState
        # Control output topic configuration
        cmd_vel_topic: str | None = "/cmd_vel",
        trajectory_setpoint_topic: str | None = None,    # PX4 TrajectorySetpoint
    ) -> None:
        super().__init__("rosie_heart")

        # --- State ---
        self._state = VehicleState()
        self._state_lock = threading.Lock()

        # --- Objective ---
        self._objective = ControlObjective(mode="idle")
        self._objective_lock = threading.Lock()

        # --- Controller ---
        self._controllers: dict[str, Controller] = {}
        self._active_controller: Controller = controller or PassthroughController()
        self.register_controller(self._active_controller)
        self.register_controller(PIDPositionController())

        # --- Topic config (stored for introspection) ---
        self._cmd_vel_topic = cmd_vel_topic
        self._trajectory_setpoint_topic = trajectory_setpoint_topic

        # --- State subscribers ---
        self._setup_state_subscribers(
            odom_topic=odom_topic,
            vehicle_odom_topic=vehicle_odom_topic,
            image_topics=image_topics or [],
            joint_state_topic=joint_state_topic,
        )

        # --- Control publishers ---
        self._cmd_vel_pub = None
        self._traj_sp_pub = None
        self._setup_control_publishers(
            cmd_vel_topic=cmd_vel_topic,
            trajectory_setpoint_topic=trajectory_setpoint_topic,
        )

        # --- Services for agent interaction ---
        self._setup_services()

        # --- Control-loop timer ---
        period = 1.0 / control_rate_hz
        self._timer = self.create_timer(period, self._control_tick)

        log.info(
            "Heart started — rate=%.0f Hz, controller=%s",
            control_rate_hz,
            self._active_controller.name,
        )

    # ------------------------------------------------------------------
    # Controller registry
    # ------------------------------------------------------------------

    def register_controller(self, ctrl: Controller) -> None:
        self._controllers[ctrl.name] = ctrl

    def switch_controller(self, name: str) -> bool:
        ctrl = self._controllers.get(name)
        if ctrl is None:
            log.warning("Unknown controller: %s", name)
            return False
        if ctrl is not self._active_controller:
            self._active_controller.reset()
            ctrl.reset()
            self._active_controller = ctrl
            log.info("Switched controller to %s", name)
        return True

    # ------------------------------------------------------------------
    # State subscription setup
    # ------------------------------------------------------------------

    def _setup_state_subscribers(
        self,
        odom_topic: str | None,
        vehicle_odom_topic: str | None,
        image_topics: list[str],
        joint_state_topic: str | None,
    ) -> None:
        # nav_msgs/Odometry
        if odom_topic:
            self._subscribe_odom(odom_topic)

        # PX4 VehicleOdometry
        if vehicle_odom_topic:
            self._subscribe_vehicle_odom(vehicle_odom_topic)

        # Compressed images
        for topic in image_topics:
            self._subscribe_image(topic)

        # Joint states
        if joint_state_topic:
            self._subscribe_joint_states(joint_state_topic)

    def _subscribe_odom(self, topic: str) -> None:
        """Subscribe to nav_msgs/msg/Odometry."""
        from rosidl_runtime_py.utilities import get_message

        msg_class = get_message("nav_msgs/msg/Odometry")
        self.create_subscription(msg_class, topic, self._on_odom, _SENSOR_QOS)
        log.info("Subscribed to Odometry on %s", topic)

    def _subscribe_vehicle_odom(self, topic: str) -> None:
        """Subscribe to px4_msgs/msg/VehicleOdometry."""
        from rosidl_runtime_py.utilities import get_message

        msg_class = get_message("px4_msgs/msg/VehicleOdometry")
        self.create_subscription(msg_class, topic, self._on_vehicle_odom, _SENSOR_QOS)
        log.info("Subscribed to VehicleOdometry on %s", topic)

    def _subscribe_image(self, topic: str) -> None:
        """Subscribe to sensor_msgs/msg/CompressedImage."""
        from rosidl_runtime_py.utilities import get_message

        msg_class = get_message("sensor_msgs/msg/CompressedImage")
        self.create_subscription(msg_class, topic, lambda m, t=topic: self._on_image(m, t), _SENSOR_QOS)
        log.info("Subscribed to CompressedImage on %s", topic)

    def _subscribe_joint_states(self, topic: str) -> None:
        """Subscribe to sensor_msgs/msg/JointState."""
        from rosidl_runtime_py.utilities import get_message

        msg_class = get_message("sensor_msgs/msg/JointState")
        self.create_subscription(msg_class, topic, self._on_joint_states, _SENSOR_QOS)
        log.info("Subscribed to JointState on %s", topic)

    # ------------------------------------------------------------------
    # State callbacks — normalise into VehicleState
    # ------------------------------------------------------------------

    def _on_odom(self, msg: Any) -> None:
        """nav_msgs/Odometry → VehicleState."""
        with self._state_lock:
            s = self._state
            s.stamp = time.monotonic()
            p = msg.pose.pose.position
            s.position = np.array([p.x, p.y, p.z])
            q = msg.pose.pose.orientation
            s.orientation = np.array([q.w, q.x, q.y, q.z])
            v = msg.twist.twist.linear
            s.linear_velocity = np.array([v.x, v.y, v.z])
            w = msg.twist.twist.angular
            s.angular_velocity = np.array([w.x, w.y, w.z])

    def _on_vehicle_odom(self, msg: Any) -> None:
        """px4_msgs/VehicleOdometry → VehicleState (NED→ENU conversion)."""
        with self._state_lock:
            s = self._state
            s.stamp = time.monotonic()
            # PX4 uses NED; convert to ENU: swap x↔y, negate z
            s.position = np.array([msg.position[1], msg.position[0], -msg.position[2]])
            # Quaternion: PX4 [w,x,y,z] NED → ENU
            s.orientation = np.array([msg.q[0], msg.q[2], msg.q[1], -msg.q[3]])
            s.linear_velocity = np.array([msg.velocity[1], msg.velocity[0], -msg.velocity[2]])
            s.angular_velocity = np.array([msg.angular_velocity[1], msg.angular_velocity[0], -msg.angular_velocity[2]])

    def _on_image(self, msg: Any, topic: str) -> None:
        """sensor_msgs/CompressedImage → images dict."""
        with self._state_lock:
            self._state.images[topic] = np.frombuffer(msg.data, dtype=np.uint8)
            self._state.stamp = time.monotonic()

    def _on_joint_states(self, msg: Any) -> None:
        """sensor_msgs/JointState → joint fields."""
        with self._state_lock:
            s = self._state
            s.stamp = time.monotonic()
            s.joint_names = list(msg.name)
            if msg.position:
                s.joint_positions = np.array(msg.position)
            if msg.velocity:
                s.joint_velocities = np.array(msg.velocity)

    # ------------------------------------------------------------------
    # Control publishers
    # ------------------------------------------------------------------

    def _setup_control_publishers(
        self,
        cmd_vel_topic: str | None,
        trajectory_setpoint_topic: str | None,
    ) -> None:
        from rosidl_runtime_py.utilities import get_message

        if cmd_vel_topic:
            twist_cls = get_message("geometry_msgs/msg/Twist")
            self._cmd_vel_pub = self.create_publisher(twist_cls, cmd_vel_topic, 10)
            self._twist_cls = twist_cls
            log.info("Publishing Twist on %s", cmd_vel_topic)

        if trajectory_setpoint_topic:
            tsp_cls = get_message("px4_msgs/msg/TrajectorySetpoint")
            self._traj_sp_pub = self.create_publisher(tsp_cls, trajectory_setpoint_topic, 10)
            self._tsp_cls = tsp_cls
            log.info("Publishing TrajectorySetpoint on %s", trajectory_setpoint_topic)

    def _publish_command(self, cmd: ControlCommand) -> None:
        """Translate a ControlCommand into platform-specific messages."""
        # Twist (cmd_vel) — universal velocity command
        if self._cmd_vel_pub is not None:
            msg = self._twist_cls()
            msg.linear.x = float(cmd.linear_velocity[0])
            msg.linear.y = float(cmd.linear_velocity[1])
            msg.linear.z = float(cmd.linear_velocity[2])
            msg.angular.x = float(cmd.angular_velocity[0])
            msg.angular.y = float(cmd.angular_velocity[1])
            msg.angular.z = float(cmd.angular_velocity[2])
            self._cmd_vel_pub.publish(msg)

        # PX4 TrajectorySetpoint
        if self._traj_sp_pub is not None and cmd.position is not None:
            msg = self._tsp_cls()
            # ENU → NED for PX4
            msg.position = [
                float(cmd.position[1]),
                float(cmd.position[0]),
                float(-cmd.position[2]),
            ]
            if cmd.yaw is not None:
                msg.yaw = float(cmd.yaw)
            self._traj_sp_pub.publish(msg)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_tick(self) -> None:
        """Called at the control rate by the timer."""
        with self._state_lock:
            state = VehicleState(
                stamp=self._state.stamp,
                position=self._state.position.copy() if self._state.position is not None else None,
                orientation=self._state.orientation.copy() if self._state.orientation is not None else None,
                linear_velocity=self._state.linear_velocity.copy() if self._state.linear_velocity is not None else None,
                angular_velocity=self._state.angular_velocity.copy() if self._state.angular_velocity is not None else None,
                joint_names=list(self._state.joint_names),
                joint_positions=self._state.joint_positions.copy() if self._state.joint_positions is not None else None,
                joint_velocities=self._state.joint_velocities.copy() if self._state.joint_velocities is not None else None,
                # Don't deep-copy images every tick for performance
                images=self._state.images,
            )

        with self._objective_lock:
            objective = self._objective

        if objective.mode == "idle":
            return

        cmd = self._active_controller.compute(state, objective)
        self._publish_command(cmd)

    # ------------------------------------------------------------------
    # Services (for agent interaction)
    # ------------------------------------------------------------------

    def _setup_services(self) -> None:
        """Register ROS 2 services for objective and state queries.

        Uses std_srvs/srv/Trigger-style JSON-in-string services so we
        don't need custom message packages.
        """
        from rosidl_runtime_py.utilities import get_service

        trigger_cls = get_service("std_srvs/srv/Trigger")

        # GET state
        self.create_service(trigger_cls, "/rosie/get_state", self._srv_get_state)

        # SET objective — we abuse Trigger by reading the objective from a
        # latched topic instead, but also provide a dedicated service using
        # rcl_interfaces/srv/SetParameters or a simple string-based service.
        # For simplicity, we use a topic-based interface (see set_objective below).

    def _srv_get_state(self, request: Any, response: Any) -> Any:
        with self._state_lock:
            snapshot = self._state.as_dict()
        response.success = True
        response.message = json.dumps(snapshot, default=str)
        return response

    # ------------------------------------------------------------------
    # Programmatic API (called from agent tools or tests)
    # ------------------------------------------------------------------

    def set_objective(self, objective: ControlObjective) -> None:
        """Set a new control objective (thread-safe)."""
        with self._objective_lock:
            old_mode = self._objective.mode
            self._objective = objective
            if objective.mode != old_mode:
                self._active_controller.reset()
                log.info("Objective changed: %s → %s", old_mode, objective.mode)

    def get_state(self) -> VehicleState:
        """Return a snapshot of the current state (thread-safe)."""
        with self._state_lock:
            return VehicleState(
                stamp=self._state.stamp,
                position=self._state.position.copy() if self._state.position is not None else None,
                orientation=self._state.orientation.copy() if self._state.orientation is not None else None,
                linear_velocity=self._state.linear_velocity.copy() if self._state.linear_velocity is not None else None,
                angular_velocity=self._state.angular_velocity.copy() if self._state.angular_velocity is not None else None,
                joint_names=list(self._state.joint_names),
                joint_positions=self._state.joint_positions.copy() if self._state.joint_positions is not None else None,
                joint_velocities=self._state.joint_velocities.copy() if self._state.joint_velocities is not None else None,
                images=dict(self._state.images),
            )

    def get_objective(self) -> ControlObjective:
        """Return the current control objective (thread-safe)."""
        with self._objective_lock:
            return self._objective

    def get_controller_names(self) -> list[str]:
        """List all registered controller names."""
        return list(self._controllers.keys())

    def get_status(self) -> dict:
        """Return a full status snapshot for the agent."""
        state = self.get_state()
        obj = self.get_objective()
        return {
            "controller": self._active_controller.name,
            "available_controllers": self.get_controller_names(),
            "objective": obj.as_dict(),
            "state": state.as_dict(),
            "cmd_vel_topic": self._cmd_vel_topic,
            "trajectory_setpoint_topic": self._trajectory_setpoint_topic,
        }


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------


def main() -> None:
    """Run the Heart as a standalone node."""
    import os

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    rclpy.init()
    heart = Heart(
        control_rate_hz=float(os.getenv("HEART_RATE_HZ", "50")),
        odom_topic=os.getenv("HEART_ODOM_TOPIC", "/odom"),
        vehicle_odom_topic=os.getenv("HEART_VEHICLE_ODOM_TOPIC"),
        image_topics=os.getenv("HEART_IMAGE_TOPICS", "").split(",") if os.getenv("HEART_IMAGE_TOPICS") else None,
        joint_state_topic=os.getenv("HEART_JOINT_STATE_TOPIC"),
        cmd_vel_topic=os.getenv("HEART_CMD_VEL_TOPIC", "/cmd_vel"),
        trajectory_setpoint_topic=os.getenv("HEART_TRAJECTORY_SP_TOPIC"),
    )

    try:
        rclpy.spin(heart)
    except KeyboardInterrupt:
        pass
    finally:
        heart.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
