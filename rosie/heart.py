"""
Heart — the continuously-spinning ROS 2 node that bridges state estimation
to control output, with a flight state machine and full PX4 protocol.

Architecture::

    State topics                Heart                     Control topics
    ────────────         ┌───────────────────────┐       ──────────────
    /odom            ──▶ │  State Router         │
    /vehicle_odometry──▶ │       ↓               │
    /camera/image    ──▶ │  VehicleState         │
    /joint_states    ──▶ │       ↓               │
                         │  FlightPhase SM       │
                         │  (INIT→READY→ACTIVE→  │
                         │   HOLD→LAND)          │
                         │       ↓               │
                         │  Controller           │──▶  /cmd_vel (Twist)
                         │  (PID / learned /     │──▶  /fmu/in/trajectory_setpoint
                         │   passthrough / MPC)  │──▶  /fmu/in/vehicle_rates_setpoint
                         └────────┬──────────────┘
                                  ↑
                         ControlObjective + ControlContext
                         (set by the agent via tools)

The Heart:
1. Subscribes to all configured state topics and normalises them into a
   single VehicleState.
2. Manages a FlightPhase state machine with ready-gate checks.
3. Runs a control loop at a fixed rate (default 50 Hz).
4. Feeds (VehicleState, ControlObjective, ControlContext) to the active
   Controller — context includes trajectory time and previous input.
5. Publishes the resulting ControlCommand as the appropriate platform
   message — Twist, TrajectorySetpoint, or VehicleRatesSetpoint.
6. Manages PX4 offboard protocol: OffboardControlMode heartbeat with
   per-phase flags (body_rate, velocity, position).
7. Records state/command data via FlightRecorder.
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

from rosie.state import VehicleState, ControlObjective, ControlCommand, OffboardFlags
from rosie.controllers.base import Controller, ControlContext
from rosie.controllers.passthrough import PassthroughController
from rosie.controllers.pid_position import PIDPositionController
from rosie.flight import FlightPhase, ReadyGate, FlightRecorder

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
        vehicle_rates_topic: str | None = None,          # PX4 VehicleRatesSetpoint
        # PX4 offboard protocol
        px4_offboard: bool = False,
        # Flight state machine
        ready_gate: ReadyGate | None = None,
    ) -> None:
        super().__init__("rosie_heart")

        # --- State ---
        self._state = VehicleState()
        self._state_lock = threading.Lock()
        self._state_update_count = 0

        # --- Objective ---
        self._objective = ControlObjective(mode="idle")
        self._objective_lock = threading.Lock()

        # --- Controller ---
        self._controllers: dict[str, Controller] = {}
        self._active_controller: Controller = controller or PassthroughController()
        self.register_controller(self._active_controller)
        self.register_controller(PIDPositionController())

        # --- Control context ---
        self._objective_start_time: float = 0.0
        self._prev_command: ControlCommand | None = None

        # --- Topic config (stored for introspection) ---
        self._cmd_vel_topic = cmd_vel_topic
        self._trajectory_setpoint_topic = trajectory_setpoint_topic
        self._vehicle_rates_topic = vehicle_rates_topic

        # --- PX4 offboard protocol ---
        self._px4_offboard = px4_offboard or (trajectory_setpoint_topic is not None) or (vehicle_rates_topic is not None)
        self._px4_armed = False
        self._px4_offboard_mode = False
        self._px4_offboard_heartbeat_count = 0

        # --- Flight state machine ---
        self._flight_phase = FlightPhase.INIT
        self._ready_gate = ready_gate or ReadyGate()

        # --- Flight recorder ---
        self._recorder = FlightRecorder()

        # --- Tracked subscriptions & publishers (for reconfigure) ---
        self._state_subs: list = []
        self._cmd_vel_pub = None
        self._traj_sp_pub = None
        self._rates_sp_pub = None
        self._offboard_ctrl_pub = None
        self._vehicle_cmd_pub = None

        # --- State subscribers ---
        self._setup_state_subscribers(
            odom_topic=odom_topic,
            vehicle_odom_topic=vehicle_odom_topic,
            image_topics=image_topics or [],
            joint_state_topic=joint_state_topic,
        )

        # --- Control publishers ---
        self._setup_control_publishers(
            cmd_vel_topic=cmd_vel_topic,
            trajectory_setpoint_topic=trajectory_setpoint_topic,
            vehicle_rates_topic=vehicle_rates_topic,
        )

        # --- Services for agent interaction ---
        self._setup_services()

        # --- Control-loop timer ---
        self._control_rate_hz = control_rate_hz
        period = 1.0 / control_rate_hz
        self._timer = self.create_timer(period, self._control_tick)

        log.info(
            "Heart started — rate=%.0f Hz, controller=%s, px4_offboard=%s",
            control_rate_hz,
            self._active_controller.name,
            self._px4_offboard,
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
    # Runtime reconfiguration
    # ------------------------------------------------------------------

    def reconfigure(
        self,
        *,
        odom_topic: str | None = None,
        vehicle_odom_topic: str | None = None,
        image_topics: list[str] | None = None,
        joint_state_topic: str | None = None,
        cmd_vel_topic: str | None = None,
        trajectory_setpoint_topic: str | None = None,
        vehicle_rates_topic: str | None = None,
        px4_offboard: bool | None = None,
    ) -> dict:
        """Reconfigure subscriptions and publishers at runtime.

        Tears down existing state subscriptions and control publishers,
        then creates new ones based on the provided topic configuration.
        Resets vehicle state and flight phase back to INIT so the ready-
        gate must pass again before commanding.

        Returns a summary of the new configuration.
        """
        # --- Tear down old state subscriptions ---
        for sub in self._state_subs:
            self.destroy_subscription(sub)
        self._state_subs.clear()

        # --- Tear down old control publishers ---
        for pub in [
            self._cmd_vel_pub,
            self._traj_sp_pub,
            self._rates_sp_pub,
            self._offboard_ctrl_pub,
            self._vehicle_cmd_pub,
        ]:
            if pub is not None:
                self.destroy_publisher(pub)
        self._cmd_vel_pub = None
        self._traj_sp_pub = None
        self._rates_sp_pub = None
        self._offboard_ctrl_pub = None
        self._vehicle_cmd_pub = None

        # --- Reset state ---
        with self._state_lock:
            self._state = VehicleState()
            self._state_update_count = 0
        with self._objective_lock:
            self._objective = ControlObjective(mode="idle")
        self._flight_phase = FlightPhase.INIT
        self._prev_command = None

        # --- Update topic config ---
        self._cmd_vel_topic = cmd_vel_topic
        self._trajectory_setpoint_topic = trajectory_setpoint_topic
        self._vehicle_rates_topic = vehicle_rates_topic

        # --- Update PX4 offboard flag ---
        if px4_offboard is not None:
            self._px4_offboard = px4_offboard
        else:
            self._px4_offboard = (
                (trajectory_setpoint_topic is not None)
                or (vehicle_rates_topic is not None)
            )
        self._px4_armed = False
        self._px4_offboard_mode = False
        self._px4_offboard_heartbeat_count = 0

        # --- Create new subscriptions ---
        self._setup_state_subscribers(
            odom_topic=odom_topic,
            vehicle_odom_topic=vehicle_odom_topic,
            image_topics=image_topics or [],
            joint_state_topic=joint_state_topic,
        )

        # --- Create new publishers ---
        self._setup_control_publishers(
            cmd_vel_topic=cmd_vel_topic,
            trajectory_setpoint_topic=trajectory_setpoint_topic,
            vehicle_rates_topic=vehicle_rates_topic,
        )

        config = {
            "odom_topic": odom_topic,
            "vehicle_odom_topic": vehicle_odom_topic,
            "image_topics": image_topics or [],
            "joint_state_topic": joint_state_topic,
            "cmd_vel_topic": cmd_vel_topic,
            "trajectory_setpoint_topic": trajectory_setpoint_topic,
            "vehicle_rates_topic": vehicle_rates_topic,
            "px4_offboard": self._px4_offboard,
        }
        log.info("Heart reconfigured: %s", config)
        return config

    # ------------------------------------------------------------------
    # Flight state machine
    # ------------------------------------------------------------------

    @property
    def flight_phase(self) -> FlightPhase:
        return self._flight_phase

    def _advance_flight_phase(self, state: VehicleState) -> None:
        """Check and advance the flight state machine."""
        phase = self._flight_phase

        if phase == FlightPhase.INIT:
            # Wait for first valid state estimate
            if state.position_valid():
                self._flight_phase = FlightPhase.READY
                log.info("Flight phase: INIT → READY (state estimate received)")

        elif phase == FlightPhase.READY:
            # Check ready-gate conditions
            passed, reason = self._ready_gate.check(state, self._state_update_count)
            if passed:
                self._flight_phase = FlightPhase.ACTIVE
                self._objective_start_time = time.monotonic()
                log.info("Flight phase: READY → ACTIVE (%s)", reason)

    def set_flight_phase(self, phase: FlightPhase) -> None:
        """Manually set the flight phase (for agent/test control)."""
        old = self._flight_phase
        self._flight_phase = phase
        if phase != old:
            log.info("Flight phase: %s → %s (manual)", old.name, phase.name)
            if phase == FlightPhase.ACTIVE:
                self._objective_start_time = time.monotonic()

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
        if odom_topic:
            self._subscribe_odom(odom_topic)
        if vehicle_odom_topic:
            self._subscribe_vehicle_odom(vehicle_odom_topic)
        for topic in image_topics:
            self._subscribe_image(topic)
        if joint_state_topic:
            self._subscribe_joint_states(joint_state_topic)

    def _subscribe_odom(self, topic: str) -> None:
        from rosidl_runtime_py.utilities import get_message
        msg_class = get_message("nav_msgs/msg/Odometry")
        sub = self.create_subscription(msg_class, topic, self._on_odom, _SENSOR_QOS)
        self._state_subs.append(sub)
        log.info("Subscribed to Odometry on %s", topic)

    def _subscribe_vehicle_odom(self, topic: str) -> None:
        from rosidl_runtime_py.utilities import get_message
        msg_class = get_message("px4_msgs/msg/VehicleOdometry")
        sub = self.create_subscription(msg_class, topic, self._on_vehicle_odom, _SENSOR_QOS)
        self._state_subs.append(sub)
        log.info("Subscribed to VehicleOdometry on %s", topic)

    def _subscribe_image(self, topic: str) -> None:
        from rosidl_runtime_py.utilities import get_message
        msg_class = get_message("sensor_msgs/msg/CompressedImage")
        sub = self.create_subscription(msg_class, topic, lambda m, t=topic: self._on_image(m, t), _SENSOR_QOS)
        self._state_subs.append(sub)
        log.info("Subscribed to CompressedImage on %s", topic)

    def _subscribe_joint_states(self, topic: str) -> None:
        from rosidl_runtime_py.utilities import get_message
        msg_class = get_message("sensor_msgs/msg/JointState")
        sub = self.create_subscription(msg_class, topic, self._on_joint_states, _SENSOR_QOS)
        self._state_subs.append(sub)
        log.info("Subscribed to JointState on %s", topic)

    # ------------------------------------------------------------------
    # State callbacks — normalise into VehicleState
    # ------------------------------------------------------------------

    def _on_odom(self, msg: Any) -> None:
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
            self._state_update_count += 1

    def _on_vehicle_odom(self, msg: Any) -> None:
        with self._state_lock:
            s = self._state
            s.stamp = time.monotonic()
            # PX4 NED → ENU: swap x↔y, negate z
            s.position = np.array([msg.position[1], msg.position[0], -msg.position[2]])
            s.orientation = np.array([msg.q[0], msg.q[2], msg.q[1], -msg.q[3]])
            s.linear_velocity = np.array([msg.velocity[1], msg.velocity[0], -msg.velocity[2]])
            s.angular_velocity = np.array([msg.angular_velocity[1], msg.angular_velocity[0], -msg.angular_velocity[2]])
            self._state_update_count += 1

    def _on_image(self, msg: Any, topic: str) -> None:
        with self._state_lock:
            self._state.images[topic] = np.frombuffer(msg.data, dtype=np.uint8)
            self._state.stamp = time.monotonic()

    def _on_joint_states(self, msg: Any) -> None:
        with self._state_lock:
            s = self._state
            s.stamp = time.monotonic()
            s.joint_names = list(msg.name)
            if msg.position:
                s.joint_positions = np.array(msg.position)
            if msg.velocity:
                s.joint_velocities = np.array(msg.velocity)
            self._state_update_count += 1

    # ------------------------------------------------------------------
    # Control publishers
    # ------------------------------------------------------------------

    def _setup_control_publishers(
        self,
        cmd_vel_topic: str | None,
        trajectory_setpoint_topic: str | None,
        vehicle_rates_topic: str | None = None,
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

        if vehicle_rates_topic:
            vrs_cls = get_message("px4_msgs/msg/VehicleRatesSetpoint")
            self._rates_sp_pub = self.create_publisher(vrs_cls, vehicle_rates_topic, 10)
            self._vrs_cls = vrs_cls
            log.info("Publishing VehicleRatesSetpoint on %s", vehicle_rates_topic)

        if self._px4_offboard:
            self._setup_px4_publishers()

    def _setup_px4_publishers(self) -> None:
        from rosidl_runtime_py.utilities import get_message

        ocm_cls = get_message("px4_msgs/msg/OffboardControlMode")
        self._offboard_ctrl_pub = self.create_publisher(
            ocm_cls, "/fmu/in/offboard_control_mode", 10,
        )
        self._ocm_cls = ocm_cls

        vcmd_cls = get_message("px4_msgs/msg/VehicleCommand")
        self._vehicle_cmd_pub = self.create_publisher(
            vcmd_cls, "/fmu/in/vehicle_command", 10,
        )
        self._vcmd_cls = vcmd_cls

        log.info("PX4 offboard publishers ready (OffboardControlMode + VehicleCommand)")

    # ------------------------------------------------------------------
    # PX4 offboard protocol
    # ------------------------------------------------------------------

    def _publish_px4_offboard_heartbeat(self, cmd: ControlCommand) -> None:
        """Publish OffboardControlMode every tick — PX4 requires >=2 Hz.

        Uses explicit offboard_flags from the command if set, otherwise
        infers from the command fields.
        """
        if self._offboard_ctrl_pub is None:
            return

        msg = self._ocm_cls()
        msg.timestamp = self._px4_timestamp()

        if cmd.offboard_flags is not None:
            # Controller explicitly set the flags (e.g. MPC body-rate)
            flags = cmd.offboard_flags
            msg.position = flags.position
            msg.velocity = flags.velocity
            msg.acceleration = flags.acceleration
            msg.attitude = flags.attitude
            msg.body_rate = flags.body_rate
        else:
            # Infer from command fields
            has_body_rate = cmd.body_rates is not None
            has_position = cmd.position is not None and not has_body_rate
            has_velocity = not has_position and not has_body_rate

            msg.position = has_position
            msg.velocity = has_velocity
            msg.acceleration = False
            msg.attitude = False
            msg.body_rate = has_body_rate

        self._offboard_ctrl_pub.publish(msg)
        self._px4_offboard_heartbeat_count += 1

    def _publish_px4_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        if self._vehicle_cmd_pub is None:
            return

        msg = self._vcmd_cls()
        msg.timestamp = self._px4_timestamp()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self._vehicle_cmd_pub.publish(msg)

    def px4_arm(self) -> None:
        self._publish_px4_vehicle_command(400, param1=1.0)
        self._px4_armed = True
        log.info("PX4: ARM command sent")

    def px4_disarm(self) -> None:
        self._publish_px4_vehicle_command(400, param1=0.0)
        self._px4_armed = False
        log.info("PX4: DISARM command sent")

    def px4_set_offboard_mode(self) -> None:
        self._publish_px4_vehicle_command(176, param1=1.0, param2=6.0)
        self._px4_offboard_mode = True
        log.info("PX4: OFFBOARD mode command sent")

    def px4_engage(self) -> None:
        if not self._px4_offboard:
            log.warning("px4_engage called but px4_offboard is disabled")
            return
        self.px4_set_offboard_mode()
        self.px4_arm()

    def px4_land(self) -> None:
        """Send LAND command to PX4."""
        # VehicleCommand::VEHICLE_CMD_NAV_LAND = 21
        self._publish_px4_vehicle_command(21)
        self._flight_phase = FlightPhase.LAND
        log.info("PX4: LAND command sent")

    @staticmethod
    def _px4_timestamp() -> int:
        return int(time.time() * 1e6)

    # ------------------------------------------------------------------
    # Command publishing
    # ------------------------------------------------------------------

    def _publish_command(self, cmd: ControlCommand) -> None:
        """Translate a ControlCommand into platform-specific messages."""
        # PX4 offboard heartbeat — MUST be published every tick
        if self._px4_offboard:
            self._publish_px4_offboard_heartbeat(cmd)

        # Body-rate command → VehicleRatesSetpoint
        if cmd.body_rates is not None and self._rates_sp_pub is not None:
            self._publish_px4_rates_setpoint(cmd)
        # Position/velocity → TrajectorySetpoint
        elif self._traj_sp_pub is not None:
            self._publish_px4_trajectory_setpoint(cmd)

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

    def _publish_px4_rates_setpoint(self, cmd: ControlCommand) -> None:
        """Publish VehicleRatesSetpoint for body-rate control (MPC, learned)."""
        msg = self._vrs_cls()
        msg.timestamp = self._px4_timestamp()

        # body_rates is [roll_rate, pitch_rate, yaw_rate] in rad/s
        msg.roll = float(cmd.body_rates[0])
        msg.pitch = float(cmd.body_rates[1])
        msg.yaw = float(cmd.body_rates[2])

        if cmd.thrust is not None:
            # PX4 expects negative thrust for upward (NED)
            msg.thrust_body = [0.0, 0.0, float(-cmd.thrust)]
        else:
            msg.thrust_body = [0.0, 0.0, 0.0]

        self._rates_sp_pub.publish(msg)

    def _publish_px4_trajectory_setpoint(self, cmd: ControlCommand) -> None:
        """Publish TrajectorySetpoint in position or velocity mode."""
        msg = self._tsp_cls()
        msg.timestamp = self._px4_timestamp()

        if cmd.position is not None:
            # ENU → NED
            msg.position = [
                float(cmd.position[1]),
                float(cmd.position[0]),
                float(-cmd.position[2]),
            ]
            if cmd.yaw is not None:
                msg.yaw = float(cmd.yaw)
            else:
                msg.yaw = float("nan")
            msg.velocity = [float("nan")] * 3
        else:
            # Velocity mode — ENU → NED
            msg.velocity = [
                float(cmd.linear_velocity[1]),
                float(cmd.linear_velocity[0]),
                float(-cmd.linear_velocity[2]),
            ]
            msg.position = [float("nan")] * 3
            if cmd.angular_velocity is not None and len(cmd.angular_velocity) >= 3:
                msg.yawspeed = float(cmd.angular_velocity[2])
            else:
                msg.yawspeed = float("nan")
            msg.yaw = float("nan")

        self._traj_sp_pub.publish(msg)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_tick(self) -> None:
        """Called at the control rate by the timer."""
        t0 = time.monotonic()

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
                images=self._state.images,
            )

        # Advance flight state machine
        self._advance_flight_phase(state)

        with self._objective_lock:
            objective = self._objective

        if objective.mode == "idle":
            # Record even when idle (for flight recorder completeness)
            self._recorder.record(state, None, self._flight_phase)
            return

        # Build control context
        ctx = ControlContext(
            t_trajectory=time.monotonic() - self._objective_start_time,
            prev_command=self._prev_command,
            image=next(iter(state.images.values()), None) if state.images else None,
        )

        cmd = self._active_controller.compute(state, objective, ctx)
        self._publish_command(cmd)
        self._prev_command = cmd

        # Record
        loop_time = time.monotonic() - t0
        self._recorder.record(state, cmd, self._flight_phase, loop_time)

    # ------------------------------------------------------------------
    # Services (for agent interaction)
    # ------------------------------------------------------------------

    def _setup_services(self) -> None:
        from rosidl_runtime_py.utilities import get_service
        trigger_cls = get_service("std_srvs/srv/Trigger")
        self.create_service(trigger_cls, "/rosie/get_state", self._srv_get_state)

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
        with self._objective_lock:
            old_mode = self._objective.mode
            self._objective = objective
            if objective.mode != old_mode:
                self._active_controller.reset()
                self._objective_start_time = time.monotonic()
                self._prev_command = None
                log.info("Objective changed: %s → %s", old_mode, objective.mode)

    def get_state(self) -> VehicleState:
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
        with self._objective_lock:
            return self._objective

    def get_controller_names(self) -> list[str]:
        return list(self._controllers.keys())

    def get_recorder(self) -> FlightRecorder:
        return self._recorder

    def get_status(self) -> dict:
        state = self.get_state()
        obj = self.get_objective()
        status = {
            "controller": self._active_controller.name,
            "available_controllers": self.get_controller_names(),
            "flight_phase": self._flight_phase.name,
            "objective": obj.as_dict(),
            "state": state.as_dict(),
            "cmd_vel_topic": self._cmd_vel_topic,
            "trajectory_setpoint_topic": self._trajectory_setpoint_topic,
            "vehicle_rates_topic": self._vehicle_rates_topic,
            "recorder": self._recorder.summary(),
        }
        if self._px4_offboard:
            status["px4"] = {
                "offboard_enabled": True,
                "armed": self._px4_armed,
                "offboard_mode": self._px4_offboard_mode,
                "heartbeat_count": self._px4_offboard_heartbeat_count,
            }
        return status


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
        vehicle_rates_topic=os.getenv("HEART_VEHICLE_RATES_TOPIC"),
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
