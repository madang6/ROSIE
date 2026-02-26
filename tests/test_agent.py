"""
Tests for rosie.agent — the ReAct agent loop.

Mocks the Ollama /api/chat endpoint to simulate multi-step tool-use
conversations without needing a running LLM.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from tests.conftest import make_ollama_text_response, make_ollama_tool_response

from rosie.agent import run_agent, _execute_tool, _chat


# ── _execute_tool ────────────────────────────────────────────────────────────


class TestExecuteTool:
    def test_known_tool(self, mock_ros2_cli):
        result = _execute_tool("ros2_list_topics", {})
        assert "/turtle1/cmd_vel" in result

    def test_unknown_tool(self):
        result = _execute_tool("nonexistent_tool", {})
        data = json.loads(result)
        assert "error" in data
        assert "Unknown tool" in data["error"]

    def test_tool_exception_caught(self, monkeypatch):
        """If a tool raises, the error is returned as JSON rather than crashing."""
        from rosie import ros2_tools

        monkeypatch.setattr(
            ros2_tools, "ros2_list_topics", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        # Need to re-patch the dispatch map too
        from rosie.tool_registry import TOOL_FUNCTIONS

        TOOL_FUNCTIONS["ros2_list_topics"] = ros2_tools.ros2_list_topics
        result = _execute_tool("ros2_list_topics", {})
        data = json.loads(result)
        assert "error" in data
        assert "boom" in data["error"]


# ── run_agent with mocked Ollama ─────────────────────────────────────────────


class TestRunAgent:
    """Test the agent loop with simulated Ollama responses."""

    def test_simple_text_response(self):
        """Model answers immediately without using tools."""
        with patch("rosie.agent.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = make_ollama_text_response(
                "I see the turtlesim robot."
            )
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            answer = run_agent("What robot is connected?")
            assert "turtlesim" in answer

    def test_single_tool_call(self, mock_ros2_cli):
        """Model calls one tool then gives a text answer."""
        responses = [
            # First call: model wants to list topics
            make_ollama_tool_response([{"name": "ros2_list_topics"}]),
            # Second call: model gives final answer
            make_ollama_text_response(
                "I found topics: /turtle1/cmd_vel and /turtle1/pose."
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

            answer = run_agent("What topics are available?")
            assert "cmd_vel" in answer or "pose" in answer
            assert call_count["n"] == 2

    def test_multi_step_tool_calls(self, mock_ros2_cli):
        """Model does discover → describe → read → answer."""
        responses = [
            # Step 1: list topics
            make_ollama_tool_response([{"name": "ros2_list_topics"}]),
            # Step 2: describe the Twist message
            make_ollama_tool_response([
                {"name": "ros2_describe_interface", "arguments": {"interface_name": "geometry_msgs/msg/Twist"}}
            ]),
            # Step 3: read pose
            make_ollama_tool_response([
                {"name": "ros2_read_topic", "arguments": {"topic_name": "/turtle1/pose"}}
            ]),
            # Step 4: final answer
            make_ollama_text_response(
                "The turtle is at position (5.5, 5.5). "
                "I can send Twist messages to /turtle1/cmd_vel to move it."
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
            assert call_count["n"] == 4

    def test_max_iterations_safety(self, mock_ros2_cli):
        """Agent stops after max iterations even if model keeps calling tools."""
        with patch("rosie.agent.requests.post") as mock_post:
            # Model always wants another tool call, never gives a text answer
            resp = MagicMock()
            resp.json.return_value = make_ollama_tool_response([
                {"name": "ros2_list_topics"}
            ])
            resp.raise_for_status = MagicMock()
            mock_post.return_value = resp

            with patch("rosie.agent.AGENT_MAX_ITERATIONS", 3):
                answer = run_agent("infinite loop test")
                assert "maximum iteration" in answer.lower()

    def test_tool_call_with_publish(self, mock_ros2_cli):
        """Model publishes a velocity command."""
        responses = [
            make_ollama_tool_response([{
                "name": "ros2_publish",
                "arguments": {
                    "topic_name": "/turtle1/cmd_vel",
                    "msg_type": "geometry_msgs/msg/Twist",
                    "data": {"linear": {"x": 1.0}},
                },
            }]),
            make_ollama_text_response("Done! I published a forward velocity command."),
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
            assert "published" in answer.lower() or "forward" in answer.lower()


class TestChatPayload:
    """Verify the payload sent to Ollama is well-formed."""

    def test_payload_includes_tools(self):
        with patch("rosie.agent.requests.post") as mock_post:
            resp = MagicMock()
            resp.json.return_value = make_ollama_text_response("ok")
            resp.raise_for_status = MagicMock()
            mock_post.return_value = resp

            run_agent("test")

            # Inspect the JSON payload
            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "tools" in payload
            assert len(payload["tools"]) == 28
            assert payload["stream"] is False

    def test_payload_includes_system_prompt(self):
        with patch("rosie.agent.requests.post") as mock_post:
            resp = MagicMock()
            resp.json.return_value = make_ollama_text_response("ok")
            resp.raise_for_status = MagicMock()
            mock_post.return_value = resp

            run_agent("hello")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            messages = payload["messages"]
            assert messages[0]["role"] == "system"
            assert "ROSIE" in messages[0]["content"]
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "hello"
