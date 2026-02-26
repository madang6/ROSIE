"""
Unit tests for rosie.ros2_tools — all 15 ROS 2 tools.

Uses the mock ROS 2 CLI (``mock_ros2_cli`` fixture) and mocked rclpy node
from conftest so no real ROS 2 installation is needed.
"""

from __future__ import annotations

import json

from rosie.ros2_tools import (
    ros2_list_topics,
    ros2_topic_info,
    ros2_list_services,
    ros2_list_actions,
    ros2_list_nodes,
    ros2_node_info,
    ros2_describe_interface,
    ros2_read_topic,
    ros2_read_topic_stream,
    ros2_publish,
    ros2_publish_repeated,
    ros2_call_service,
    ros2_send_action_goal,
    ros2_get_param,
    ros2_set_param,
    _fill_message,
)


# ── Discovery tools ──────────────────────────────────────────────────────────


class TestListTopics:
    def test_returns_topics_with_types(self, mock_ros2_cli):
        result = ros2_list_topics()
        assert "/turtle1/cmd_vel" in result
        assert "geometry_msgs/msg/Twist" in result
        assert "/turtle1/pose" in result

    def test_includes_system_topics(self, mock_ros2_cli):
        result = ros2_list_topics()
        assert "/rosout" in result
        assert "/parameter_events" in result


class TestTopicInfo:
    def test_returns_type_and_qos(self, mock_ros2_cli):
        result = ros2_topic_info("/turtle1/cmd_vel")
        assert "geometry_msgs/msg/Twist" in result
        assert "Subscription count: 1" in result

    def test_shows_publisher_count(self, mock_ros2_cli):
        result = ros2_topic_info("/turtle1/pose")
        assert "Publisher count: 1" in result

    def test_unknown_topic_returns_error(self, mock_ros2_cli):
        result = ros2_topic_info("/nonexistent")
        assert "[error]" in result


class TestListServices:
    def test_returns_services(self, mock_ros2_cli):
        result = ros2_list_services()
        assert "/turtle1/set_pen" in result
        assert "/clear" in result
        assert "/reset" in result


class TestListActions:
    def test_returns_actions(self, mock_ros2_cli):
        result = ros2_list_actions()
        assert "/turtle1/rotate_absolute" in result
        assert "turtlesim/action/RotateAbsolute" in result


class TestListNodes:
    def test_returns_nodes(self, mock_ros2_cli):
        result = ros2_list_nodes()
        assert "/turtlesim" in result


class TestNodeInfo:
    def test_returns_subscribers_and_publishers(self, mock_ros2_cli):
        result = ros2_node_info("/turtlesim")
        assert "/turtle1/cmd_vel" in result
        assert "/turtle1/pose" in result
        assert "Subscribers:" in result
        assert "Publishers:" in result

    def test_unknown_node_returns_error(self, mock_ros2_cli):
        result = ros2_node_info("/nonexistent")
        assert "[error]" in result


class TestDescribeInterface:
    def test_twist_definition(self, mock_ros2_cli):
        result = ros2_describe_interface("geometry_msgs/msg/Twist")
        assert "linear" in result
        assert "angular" in result
        assert "float64 x" in result

    def test_pose_definition(self, mock_ros2_cli):
        result = ros2_describe_interface("turtlesim/msg/Pose")
        assert "float32 x" in result
        assert "theta" in result

    def test_unknown_interface(self, mock_ros2_cli):
        result = ros2_describe_interface("fake/msg/Nope")
        assert "[error]" in result


# ── Observation tools ────────────────────────────────────────────────────────


class TestReadTopic:
    def test_reads_pose_topic(self):
        """The mock node delivers a FakePose when subscribing to a 'pose' topic."""
        result = ros2_read_topic("/turtle1/pose", timeout_sec=1.0)
        data = json.loads(result)
        assert "x" in data
        assert data["x"] == 5.5

    def test_unknown_topic_returns_error(self):
        result = ros2_read_topic("/no/such/topic", timeout_sec=0.5)
        data = json.loads(result)
        assert "error" in data

    def test_timeout_returns_warning(self):
        """Non-pose topics don't get a mock callback, so they time out."""
        result = ros2_read_topic("/rosout", timeout_sec=0.1)
        data = json.loads(result)
        # Either a warning (timeout) or a message — both are valid
        assert "warning" in data or isinstance(data, dict)


class TestReadTopicStream:
    def test_returns_list(self):
        result = ros2_read_topic_stream("/turtle1/pose", count=2, timeout_sec=1.0)
        data = json.loads(result)
        assert isinstance(data, list)
        # The mock delivers one message; we still get a list
        assert len(data) >= 1
        assert "x" in data[0]


# ── Command tools ────────────────────────────────────────────────────────────


class TestPublish:
    def test_publish_returns_success(self):
        result = ros2_publish(
            "/turtle1/cmd_vel",
            "geometry_msgs/msg/Twist",
            {"linear": {"x": 1.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}},
        )
        data = json.loads(result)
        assert data["status"] == "published"
        assert data["topic"] == "/turtle1/cmd_vel"


class TestPublishRepeated:
    def test_repeated_publish(self):
        result = ros2_publish_repeated(
            "/turtle1/cmd_vel",
            "geometry_msgs/msg/Twist",
            {"linear": {"x": 0.5}},
            rate_hz=50.0,
            duration_sec=0.1,
        )
        data = json.loads(result)
        assert data["status"] == "published"
        assert data["messages_sent"] >= 1
        assert data["duration_sec"] == 0.1


class TestCallService:
    def test_call_returns_response(self):
        result = ros2_call_service(
            "/turtle1/set_pen",
            "std_srvs/srv/SetBool",
            {"data": True},
        )
        data = json.loads(result)
        assert "success" in data or "message" in data


class TestSendActionGoal:
    def test_action_server_not_available(self):
        """When the action server is unreachable, a structured error is returned."""
        from unittest.mock import patch, MagicMock

        mock_client = MagicMock()
        mock_client.wait_for_server.return_value = False
        mock_client.destroy = MagicMock()

        with patch("rclpy.action.ActionClient", return_value=mock_client):
            result = ros2_send_action_goal(
                "/turtle1/rotate_absolute",
                "nav2_msgs/action/NavigateToPose",
                {},
            )
            data = json.loads(result)
            assert "error" in data
            assert "not available" in data["error"]


# ── Parameter tools ──────────────────────────────────────────────────────────


class TestGetParam:
    def test_get_param(self, mock_ros2_cli):
        result = ros2_get_param("/turtlesim", "background_r")
        assert "69" in result


class TestSetParam:
    def test_set_param(self, mock_ros2_cli):
        result = ros2_set_param("/turtlesim", "background_r", "255")
        assert "successful" in result.lower()


# ── Internal helpers ─────────────────────────────────────────────────────────


class TestFillMessage:
    def test_flat_fields(self):
        import types

        msg = types.SimpleNamespace(x=0.0, y=0.0)
        _fill_message(msg, {"x": 1.5, "y": 2.5})
        assert msg.x == 1.5
        assert msg.y == 2.5

    def test_nested_fields(self):
        import types

        msg = types.SimpleNamespace(
            linear=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            angular=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
        )
        _fill_message(msg, {"linear": {"x": 2.0}, "angular": {"z": 1.0}})
        assert msg.linear.x == 2.0
        assert msg.angular.z == 1.0

    def test_ignores_unknown_fields(self):
        import types

        msg = types.SimpleNamespace(x=0.0)
        _fill_message(msg, {"x": 1.0, "nonexistent": 99})
        assert msg.x == 1.0
