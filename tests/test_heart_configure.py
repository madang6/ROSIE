"""
Tests for Heart runtime reconfiguration and auto-discovery.

Tests the _parse_topic_list / _identify_robot heuristics, the
heart_configure tool, and the heart_auto_configure tool.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rosie.state import VehicleState, ControlObjective
from rosie.controllers.passthrough import PassthroughController
from rosie.controllers.pid_position import PIDPositionController
import rosie.heart_tools as ht
from rosie.heart_tools import _parse_topic_list, _identify_robot


# ---------------------------------------------------------------------------
# Fake Heart with reconfigure() support
# ---------------------------------------------------------------------------


class _FakeHeart:
    """Minimal Heart stand-in that tracks reconfigure calls."""

    def __init__(self):
        self._state = VehicleState(
            stamp=time.monotonic(),
            position=np.array([1.0, 2.0, 0.0]),
        )
        self._objective = ControlObjective(mode="idle")
        self._controllers = {
            "passthrough": PassthroughController(),
            "pid_position": PIDPositionController(),
        }
        self._active_controller = self._controllers["passthrough"]
        self._cmd_vel_topic = "/cmd_vel"
        self._trajectory_setpoint_topic = None
        self._vehicle_rates_topic = None
        self._px4_offboard = False
        self.reconfigure_calls: list[dict] = []

    def reconfigure(self, **kwargs) -> dict:
        self.reconfigure_calls.append(kwargs)
        # Update stored config
        self._cmd_vel_topic = kwargs.get("cmd_vel_topic")
        self._trajectory_setpoint_topic = kwargs.get("trajectory_setpoint_topic")
        self._vehicle_rates_topic = kwargs.get("vehicle_rates_topic")
        self._px4_offboard = kwargs.get("px4_offboard", False)
        return kwargs

    def get_state(self):
        return self._state

    def get_objective(self):
        return self._objective

    def set_objective(self, obj):
        self._objective = obj

    def switch_controller(self, name):
        if name in self._controllers:
            self._active_controller = self._controllers[name]
            return True
        return False

    def get_controller_names(self):
        return list(self._controllers.keys())

    def get_status(self):
        return {
            "controller": self._active_controller.name,
            "available_controllers": self.get_controller_names(),
            "objective": self._objective.as_dict(),
            "state": self._state.as_dict(),
            "cmd_vel_topic": self._cmd_vel_topic,
        }


@pytest.fixture(autouse=True)
def _fake_heart():
    heart = _FakeHeart()
    ht.set_heart(heart)
    yield heart
    ht.set_heart(None)


# ---------------------------------------------------------------------------
# Tests: _parse_topic_list
# ---------------------------------------------------------------------------


class TestParseTopicList:
    def test_standard_format(self):
        raw = (
            "/turtle1/cmd_vel [geometry_msgs/msg/Twist]\n"
            "/turtle1/pose [turtlesim/msg/Pose]\n"
            "/rosout [rcl_interfaces/msg/Log]\n"
        )
        topics = _parse_topic_list(raw)
        assert len(topics) == 3
        assert topics[0] == ("/turtle1/cmd_vel", "geometry_msgs/msg/Twist")
        assert topics[1] == ("/turtle1/pose", "turtlesim/msg/Pose")

    def test_empty_input(self):
        assert _parse_topic_list("") == []
        assert _parse_topic_list("  \n  \n") == []

    def test_px4_topics(self):
        raw = (
            "/fmu/out/vehicle_odometry [px4_msgs/msg/VehicleOdometry]\n"
            "/fmu/in/trajectory_setpoint [px4_msgs/msg/TrajectorySetpoint]\n"
            "/fmu/in/offboard_control_mode [px4_msgs/msg/OffboardControlMode]\n"
            "/fmu/in/vehicle_command [px4_msgs/msg/VehicleCommand]\n"
            "/fmu/in/vehicle_rates_setpoint [px4_msgs/msg/VehicleRatesSetpoint]\n"
        )
        topics = _parse_topic_list(raw)
        assert len(topics) == 5
        names = [t for t, _ in topics]
        assert "/fmu/out/vehicle_odometry" in names
        assert "/fmu/in/trajectory_setpoint" in names

    def test_multiple_types_takes_first(self):
        raw = "/topic [type1, type2]\n"
        topics = _parse_topic_list(raw)
        assert topics == [("/topic", "type1")]


# ---------------------------------------------------------------------------
# Tests: _identify_robot
# ---------------------------------------------------------------------------


class TestIdentifyRobot:
    def test_ground_robot(self):
        topics = [
            ("/odom", "nav_msgs/msg/Odometry"),
            ("/cmd_vel", "geometry_msgs/msg/Twist"),
            ("/scan", "sensor_msgs/msg/LaserScan"),
            ("/rosout", "rcl_interfaces/msg/Log"),
        ]
        result = _identify_robot(topics)
        assert result["robot_type"] == "ground_robot"
        cfg = result["config"]
        assert cfg["odom_topic"] == "/odom"
        assert cfg["cmd_vel_topic"] == "/cmd_vel"
        assert cfg["px4_offboard"] is False
        assert cfg["vehicle_odom_topic"] is None

    def test_px4_drone(self):
        topics = [
            ("/fmu/out/vehicle_odometry", "px4_msgs/msg/VehicleOdometry"),
            ("/fmu/in/trajectory_setpoint", "px4_msgs/msg/TrajectorySetpoint"),
            ("/fmu/in/offboard_control_mode", "px4_msgs/msg/OffboardControlMode"),
            ("/fmu/in/vehicle_command", "px4_msgs/msg/VehicleCommand"),
            ("/fmu/in/vehicle_rates_setpoint", "px4_msgs/msg/VehicleRatesSetpoint"),
        ]
        result = _identify_robot(topics)
        assert result["robot_type"] == "px4_drone"
        cfg = result["config"]
        assert cfg["vehicle_odom_topic"] == "/fmu/out/vehicle_odometry"
        assert cfg["trajectory_setpoint_topic"] == "/fmu/in/trajectory_setpoint"
        assert cfg["vehicle_rates_topic"] == "/fmu/in/vehicle_rates_setpoint"
        assert cfg["px4_offboard"] is True

    def test_manipulator(self):
        topics = [
            ("/joint_states", "sensor_msgs/msg/JointState"),
            ("/rosout", "rcl_interfaces/msg/Log"),
        ]
        result = _identify_robot(topics)
        assert result["robot_type"] == "manipulator"
        assert result["config"]["joint_state_topic"] == "/joint_states"

    def test_mobile_manipulator(self):
        topics = [
            ("/odom", "nav_msgs/msg/Odometry"),
            ("/cmd_vel", "geometry_msgs/msg/Twist"),
            ("/joint_states", "sensor_msgs/msg/JointState"),
        ]
        result = _identify_robot(topics)
        assert result["robot_type"] == "mobile_manipulator"
        cfg = result["config"]
        assert cfg["odom_topic"] == "/odom"
        assert cfg["joint_state_topic"] == "/joint_states"
        assert cfg["cmd_vel_topic"] == "/cmd_vel"

    def test_camera_detection(self):
        topics = [
            ("/odom", "nav_msgs/msg/Odometry"),
            ("/camera/rgb/image_raw/compressed", "sensor_msgs/msg/CompressedImage"),
            ("/camera/depth/image_raw/compressed", "sensor_msgs/msg/CompressedImage"),
        ]
        result = _identify_robot(topics)
        cfg = result["config"]
        assert "/camera/rgb/image_raw/compressed" in cfg["image_topics"]
        assert "/camera/depth/image_raw/compressed" in cfg["image_topics"]

    def test_unknown_robot(self):
        topics = [
            ("/rosout", "rcl_interfaces/msg/Log"),
            ("/parameter_events", "rcl_interfaces/msg/ParameterEvent"),
        ]
        result = _identify_robot(topics)
        assert result["robot_type"] == "unknown"

    def test_detected_descriptions(self):
        topics = [
            ("/odom", "nav_msgs/msg/Odometry"),
            ("/cmd_vel", "geometry_msgs/msg/Twist"),
        ]
        result = _identify_robot(topics)
        descriptions = result["detected"]
        assert any("Odometry" in d for d in descriptions)
        assert any("Velocity command" in d for d in descriptions)


# ---------------------------------------------------------------------------
# Tests: heart_configure tool
# ---------------------------------------------------------------------------


class TestHeartConfigure:
    def test_basic_configure(self, _fake_heart):
        result = json.loads(ht.heart_configure(
            odom_topic="/odom",
            cmd_vel_topic="/cmd_vel",
        ))
        assert result["status"] == "reconfigured"
        assert len(_fake_heart.reconfigure_calls) == 1

        call = _fake_heart.reconfigure_calls[0]
        assert call["odom_topic"] == "/odom"
        assert call["cmd_vel_topic"] == "/cmd_vel"

    def test_px4_configure(self, _fake_heart):
        result = json.loads(ht.heart_configure(
            vehicle_odom_topic="/fmu/out/vehicle_odometry",
            trajectory_setpoint_topic="/fmu/in/trajectory_setpoint",
            px4_offboard=True,
        ))
        assert result["status"] == "reconfigured"
        call = _fake_heart.reconfigure_calls[0]
        assert call["vehicle_odom_topic"] == "/fmu/out/vehicle_odometry"
        assert call["px4_offboard"] is True

    def test_image_topics(self, _fake_heart):
        result = json.loads(ht.heart_configure(
            image_topics=["/cam0/compressed", "/cam1/compressed"],
        ))
        assert result["status"] == "reconfigured"
        call = _fake_heart.reconfigure_calls[0]
        assert call["image_topics"] == ["/cam0/compressed", "/cam1/compressed"]

    def test_all_none_disables_everything(self, _fake_heart):
        result = json.loads(ht.heart_configure())
        assert result["status"] == "reconfigured"
        call = _fake_heart.reconfigure_calls[0]
        assert call["odom_topic"] is None
        assert call["cmd_vel_topic"] is None

    def test_requires_heart(self):
        ht.set_heart(None)
        with pytest.raises(RuntimeError, match="Heart node is not running"):
            ht.heart_configure(odom_topic="/odom")


# ---------------------------------------------------------------------------
# Tests: heart_auto_configure tool
# ---------------------------------------------------------------------------


class TestHeartAutoConfigure:
    def test_ground_robot_auto(self, _fake_heart):
        """Simulate a turtlebot-like topic graph."""
        fake_topics = (
            "/odom [nav_msgs/msg/Odometry]\n"
            "/cmd_vel [geometry_msgs/msg/Twist]\n"
            "/scan [sensor_msgs/msg/LaserScan]\n"
            "/rosout [rcl_interfaces/msg/Log]\n"
        )
        with patch("rosie.ros2_tools.ros2_list_topics", return_value=fake_topics):
            result = json.loads(ht.heart_auto_configure())

        assert result["status"] == "auto_configured"
        assert result["robot_type"] == "ground_robot"
        assert result["config"]["odom_topic"] == "/odom"
        assert result["config"]["cmd_vel_topic"] == "/cmd_vel"
        assert len(_fake_heart.reconfigure_calls) == 1

    def test_px4_drone_auto(self, _fake_heart):
        """Simulate a PX4 drone topic graph."""
        fake_topics = (
            "/fmu/out/vehicle_odometry [px4_msgs/msg/VehicleOdometry]\n"
            "/fmu/in/trajectory_setpoint [px4_msgs/msg/TrajectorySetpoint]\n"
            "/fmu/in/offboard_control_mode [px4_msgs/msg/OffboardControlMode]\n"
            "/fmu/in/vehicle_command [px4_msgs/msg/VehicleCommand]\n"
            "/fmu/in/vehicle_rates_setpoint [px4_msgs/msg/VehicleRatesSetpoint]\n"
            "/rosout [rcl_interfaces/msg/Log]\n"
        )
        with patch("rosie.ros2_tools.ros2_list_topics", return_value=fake_topics):
            result = json.loads(ht.heart_auto_configure())

        assert result["status"] == "auto_configured"
        assert result["robot_type"] == "px4_drone"
        assert result["config"]["px4_offboard"] is True
        assert result["config"]["vehicle_odom_topic"] == "/fmu/out/vehicle_odometry"

    def test_handles_empty_graph(self, _fake_heart):
        with patch("rosie.ros2_tools.ros2_list_topics", return_value=""):
            result = json.loads(ht.heart_auto_configure())
        assert "error" in result
        assert len(_fake_heart.reconfigure_calls) == 0

    def test_handles_cli_error(self, _fake_heart):
        with patch("rosie.ros2_tools.ros2_list_topics", return_value="[error] daemon not running"):
            result = json.loads(ht.heart_auto_configure())
        assert "error" in result
        assert len(_fake_heart.reconfigure_calls) == 0

    def test_requires_heart(self):
        ht.set_heart(None)
        with pytest.raises(RuntimeError, match="Heart node is not running"):
            ht.heart_auto_configure()

    def test_detected_list_populated(self, _fake_heart):
        fake_topics = (
            "/odom [nav_msgs/msg/Odometry]\n"
            "/joint_states [sensor_msgs/msg/JointState]\n"
            "/cmd_vel [geometry_msgs/msg/Twist]\n"
        )
        with patch("rosie.ros2_tools.ros2_list_topics", return_value=fake_topics):
            result = json.loads(ht.heart_auto_configure())

        assert result["robot_type"] == "mobile_manipulator"
        assert len(result["detected"]) >= 3
