"""
Integration test — simulates a full conversation where the agent discovers
a turtlesim-like robot and commands it, end-to-end.

Mocks only the Ollama API; the ROS 2 tools run against the mock graph from
conftest.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from tests.conftest import make_ollama_text_response, make_ollama_tool_response

from rosie.agent import run_agent


class TestDiscoverAndCommand:
    """Simulate: user says 'move the turtle forward' → agent discovers graph,
    identifies cmd_vel, describes Twist, publishes, reads odom, reports."""

    def test_full_discover_plan_execute_cycle(self, mock_ros2_cli):
        responses = [
            # 1. Agent discovers topics
            make_ollama_tool_response([{"name": "ros2_list_topics"}]),
            # 2. Agent discovers nodes
            make_ollama_tool_response([{"name": "ros2_list_nodes"}]),
            # 3. Agent inspects turtlesim node
            make_ollama_tool_response([
                {"name": "ros2_node_info", "arguments": {"node_name": "/turtlesim"}}
            ]),
            # 4. Agent describes the Twist message
            make_ollama_tool_response([
                {"name": "ros2_describe_interface", "arguments": {"interface_name": "geometry_msgs/msg/Twist"}}
            ]),
            # 5. Agent reads current pose
            make_ollama_tool_response([
                {"name": "ros2_read_topic", "arguments": {"topic_name": "/turtle1/pose"}}
            ]),
            # 6. Agent publishes velocity command
            make_ollama_tool_response([{
                "name": "ros2_publish_repeated",
                "arguments": {
                    "topic_name": "/turtle1/cmd_vel",
                    "msg_type": "geometry_msgs/msg/Twist",
                    "data": {"linear": {"x": 2.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}},
                    "rate_hz": 10.0,
                    "duration_sec": 0.1,
                },
            }]),
            # 7. Agent reads pose again to verify
            make_ollama_tool_response([
                {"name": "ros2_read_topic", "arguments": {"topic_name": "/turtle1/pose"}}
            ]),
            # 8. Agent delivers final report
            make_ollama_text_response(
                "Done! I discovered a turtlesim robot with topics /turtle1/cmd_vel "
                "and /turtle1/pose. I published a forward velocity of 2.0 m/s for "
                "0.1 seconds. The turtle's current position is (5.5, 5.5)."
            ),
        ]
        call_count = {"n": 0}

        with patch("rosie.agent.requests.post") as mock_post:
            def side_effect(*args, **kwargs):
                resp = MagicMock()
                resp.json.return_value = responses[call_count["n"]]
                resp.raise_for_status = MagicMock()
                call_count["n"] += 1
                return resp

            mock_post.side_effect = side_effect

            answer = run_agent("Move the turtle forward")

            # Verify full cycle completed
            assert call_count["n"] == 8
            assert "turtlesim" in answer.lower() or "turtle" in answer.lower()
            assert "5.5" in answer
            assert "cmd_vel" in answer or "velocity" in answer.lower()


class TestServiceDiscoveryAndCall:
    """Simulate: user says 'reset the simulation' → agent finds /reset service."""

    def test_discover_and_call_service(self, mock_ros2_cli):
        responses = [
            # 1. List services
            make_ollama_tool_response([{"name": "ros2_list_services"}]),
            # 2. Call /reset
            make_ollama_tool_response([{
                "name": "ros2_call_service",
                "arguments": {
                    "service_name": "/reset",
                    "service_type": "std_srvs/srv/SetBool",
                    "request": {},
                },
            }]),
            # 3. Report
            make_ollama_text_response("I reset the simulation by calling the /reset service."),
        ]
        call_count = {"n": 0}

        with patch("rosie.agent.requests.post") as mock_post:
            def side_effect(*args, **kwargs):
                resp = MagicMock()
                resp.json.return_value = responses[call_count["n"]]
                resp.raise_for_status = MagicMock()
                call_count["n"] += 1
                return resp

            mock_post.side_effect = side_effect

            answer = run_agent("Reset the simulation")
            assert call_count["n"] == 3
            assert "reset" in answer.lower()


class TestStateEstimation:
    """Simulate: user asks 'where is the turtle?' → agent reads /turtle1/pose."""

    def test_read_state(self, mock_ros2_cli):
        responses = [
            # 1. List topics to find pose
            make_ollama_tool_response([{"name": "ros2_list_topics"}]),
            # 2. Read the pose
            make_ollama_tool_response([
                {"name": "ros2_read_topic", "arguments": {"topic_name": "/turtle1/pose"}}
            ]),
            # 3. Report position
            make_ollama_text_response(
                "The turtle is at position x=5.5, y=5.5 with heading theta=0.0."
            ),
        ]
        call_count = {"n": 0}

        with patch("rosie.agent.requests.post") as mock_post:
            def side_effect(*args, **kwargs):
                resp = MagicMock()
                resp.json.return_value = responses[call_count["n"]]
                resp.raise_for_status = MagicMock()
                call_count["n"] += 1
                return resp

            mock_post.side_effect = side_effect

            answer = run_agent("Where is the turtle?")
            assert "5.5" in answer
            assert call_count["n"] == 3


class TestParameterTuning:
    """Simulate: user says 'change the background to red'."""

    def test_set_parameter(self, mock_ros2_cli):
        responses = [
            # 1. Set parameter
            make_ollama_tool_response([{
                "name": "ros2_set_param",
                "arguments": {
                    "node_name": "/turtlesim",
                    "param_name": "background_r",
                    "value": "255",
                },
            }]),
            # 2. Confirm
            make_ollama_text_response("I set the background red channel to 255."),
        ]
        call_count = {"n": 0}

        with patch("rosie.agent.requests.post") as mock_post:
            def side_effect(*args, **kwargs):
                resp = MagicMock()
                resp.json.return_value = responses[call_count["n"]]
                resp.raise_for_status = MagicMock()
                call_count["n"] += 1
                return resp

            mock_post.side_effect = side_effect

            answer = run_agent("Change the background to red")
            assert "255" in answer or "red" in answer.lower()
