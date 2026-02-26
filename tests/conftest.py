"""
Shared fixtures for ROSIE tests.

Provides a mock ROS 2 environment that simulates a turtlesim-like robot graph
so tests can run without a real ROS 2 installation.
"""

from __future__ import annotations

import json
import sys
import types
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Mock ROS 2 packages *before* any rosie module is imported.
# This prevents ImportError on machines without ROS 2.
# ──────────────────────────────────────────────────────────────────────────────

# --- rclpy mocks ---
_rclpy = types.ModuleType("rclpy")
_rclpy.ok = MagicMock(return_value=True)
_rclpy.init = MagicMock()
_rclpy.spin_once = MagicMock()
_rclpy.spin_until_future_complete = MagicMock()

_rclpy_node = types.ModuleType("rclpy.node")


class _MockNode:
    """Simulates an rclpy Node with a turtlesim-like topic graph."""

    def __init__(self, name: str = "rosie_agent"):
        self.name = name
        self._subscriptions = []
        self._publishers = []

    def get_topic_names_and_types(self):
        return [
            ("/turtle1/cmd_vel", ["geometry_msgs/msg/Twist"]),
            ("/turtle1/pose", ["turtlesim/msg/Pose"]),
            ("/rosout", ["rcl_interfaces/msg/Log"]),
            ("/parameter_events", ["rcl_interfaces/msg/ParameterEvent"]),
        ]

    def create_subscription(self, msg_class, topic, callback, qos):
        sub = MagicMock()
        self._subscriptions.append(sub)
        # Simulate receiving a message for /turtle1/pose
        if "pose" in topic.lower():
            fake = msg_class()
            callback(fake)
        return sub

    def destroy_subscription(self, sub):
        if sub in self._subscriptions:
            self._subscriptions.remove(sub)

    def create_publisher(self, msg_class, topic, qos):
        pub = MagicMock()
        self._publishers.append(pub)
        return pub

    def destroy_publisher(self, pub):
        if pub in self._publishers:
            self._publishers.remove(pub)

    def create_client(self, srv_class, service_name):
        client = MagicMock()
        client.wait_for_service = MagicMock(return_value=True)
        future = MagicMock()
        resp = srv_class.Response()
        future.result = MagicMock(return_value=resp)
        client.call_async = MagicMock(return_value=future)
        return client

    def destroy_client(self, client):
        pass


_rclpy_node.Node = _MockNode
_rclpy.create_node = lambda name: _MockNode(name)

_rclpy_qos = types.ModuleType("rclpy.qos")
_rclpy_qos.QoSProfile = MagicMock
_rclpy_qos.ReliabilityPolicy = MagicMock()
_rclpy_qos.ReliabilityPolicy.BEST_EFFORT = 1
_rclpy_qos.ReliabilityPolicy.RELIABLE = 2
_rclpy_qos.DurabilityPolicy = MagicMock()
_rclpy_qos.DurabilityPolicy.VOLATILE = 1
_rclpy_qos.DurabilityPolicy.TRANSIENT_LOCAL = 2
_rclpy_qos.HistoryPolicy = MagicMock()
_rclpy_qos.HistoryPolicy.KEEP_LAST = 1

_rclpy_action = types.ModuleType("rclpy.action")
_rclpy_action.ActionClient = MagicMock

# --- rosidl_runtime_py mocks ---
_rosidl = types.ModuleType("rosidl_runtime_py")
_rosidl_utilities = types.ModuleType("rosidl_runtime_py.utilities")


class _FakeTwist:
    """Minimal mock of geometry_msgs/msg/Twist."""

    def __init__(self):
        self.linear = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.angular = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)


class _FakePose:
    """Minimal mock of turtlesim/msg/Pose."""

    def __init__(self):
        self.x = 5.5
        self.y = 5.5
        self.theta = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0


class _FakeSetBoolRequest:
    def __init__(self):
        self.data = False


class _FakeSetBoolResponse:
    def __init__(self):
        self.success = True
        self.message = "ok"


class _FakeSetBool:
    Request = _FakeSetBoolRequest
    Response = _FakeSetBoolResponse


class _FakeActionGoal:
    def __init__(self):
        pass


class _FakeActionResult:
    def __init__(self):
        self.status = "SUCCEEDED"


class _FakeAction:
    Goal = _FakeActionGoal
    Result = _FakeActionResult


# Message / interface resolution
_MSG_MAP = {
    "geometry_msgs/msg/Twist": _FakeTwist,
    "turtlesim/msg/Pose": _FakePose,
}

_SRV_MAP = {
    "std_srvs/srv/SetBool": _FakeSetBool,
}

_ACT_MAP = {
    "nav2_msgs/action/NavigateToPose": _FakeAction,
}


def _get_message(type_str):
    return _MSG_MAP.get(type_str, _FakeTwist)


def _get_service(type_str):
    return _SRV_MAP.get(type_str, _FakeSetBool)


def _get_action(type_str):
    return _ACT_MAP.get(type_str, _FakeAction)


_rosidl_utilities.get_message = _get_message
_rosidl_utilities.get_service = _get_service
_rosidl_utilities.get_action = _get_action


def _message_to_ordereddict(msg):
    """Convert a mock message to an OrderedDict, mimicking rosidl behaviour."""
    d = OrderedDict()
    for attr in sorted(vars(msg)):
        if attr.startswith("_"):
            continue
        val = getattr(msg, attr)
        if hasattr(val, "__dict__") and not callable(val):
            d[attr] = OrderedDict(
                (k, v) for k, v in sorted(vars(val).items()) if not k.startswith("_")
            )
        else:
            d[attr] = val
    return d


_rosidl.message_to_ordereddict = _message_to_ordereddict
_rosidl.get_interface_path = MagicMock(return_value="/fake/path")

# Register all mocks in sys.modules so imports succeed
sys.modules["rclpy"] = _rclpy
sys.modules["rclpy.node"] = _rclpy_node
sys.modules["rclpy.qos"] = _rclpy_qos
sys.modules["rclpy.action"] = _rclpy_action
sys.modules["rosidl_runtime_py"] = _rosidl
sys.modules["rosidl_runtime_py.utilities"] = _rosidl_utilities

# ──────────────────────────────────────────────────────────────────────────────
# Now safe to import rosie modules
# ──────────────────────────────────────────────────────────────────────────────

# Reset the cached node so each test gets a fresh one
import rosie.ros2_tools as _ros2_tools  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_ros2_node():
    """Ensure a fresh mock node for every test."""
    _ros2_tools._node = None
    yield
    _ros2_tools._node = None


# ──────────────────────────────────────────────────────────────────────────────
# Simulated ROS CLI output
# ──────────────────────────────────────────────────────────────────────────────

#: Mapping of ``ros2 <subcommand> …`` → simulated stdout.
SIMULATED_CLI: dict[tuple[str, ...], str] = {
    ("topic", "list", "-t"): (
        "/turtle1/cmd_vel [geometry_msgs/msg/Twist]\n"
        "/turtle1/pose [turtlesim/msg/Pose]\n"
        "/rosout [rcl_interfaces/msg/Log]\n"
        "/parameter_events [rcl_interfaces/msg/ParameterEvent]"
    ),
    ("topic", "info", "/turtle1/cmd_vel", "-v"): (
        "Type: geometry_msgs/msg/Twist\n\n"
        "Publisher count: 0\n\n"
        "Subscription count: 1\n"
        "  Node name: turtlesim\n"
        "  QoS: reliability=RELIABLE, durability=VOLATILE"
    ),
    ("topic", "info", "/turtle1/pose", "-v"): (
        "Type: turtlesim/msg/Pose\n\n"
        "Publisher count: 1\n"
        "  Node name: turtlesim\n\n"
        "Subscription count: 0"
    ),
    ("service", "list", "-t"): (
        "/turtle1/set_pen [turtlesim/srv/SetPen]\n"
        "/turtle1/teleport_absolute [turtlesim/srv/TeleportAbsolute]\n"
        "/turtle1/teleport_relative [turtlesim/srv/TeleportRelative]\n"
        "/clear [std_srvs/srv/Empty]\n"
        "/reset [std_srvs/srv/Empty]"
    ),
    ("action", "list", "-t"): (
        "/turtle1/rotate_absolute [turtlesim/action/RotateAbsolute]"
    ),
    ("node", "list"): "/turtlesim\n/rosie_agent",
    ("node", "info", "/turtlesim"): (
        "/turtlesim\n"
        "  Subscribers:\n"
        "    /turtle1/cmd_vel: geometry_msgs/msg/Twist\n"
        "  Publishers:\n"
        "    /turtle1/pose: turtlesim/msg/Pose\n"
        "  Service Servers:\n"
        "    /turtle1/set_pen: turtlesim/srv/SetPen\n"
        "  Action Servers:\n"
        "    /turtle1/rotate_absolute: turtlesim/action/RotateAbsolute"
    ),
    ("interface", "show", "geometry_msgs/msg/Twist"): (
        "# Linear velocity\n"
        "Vector3 linear\n"
        "  float64 x\n"
        "  float64 y\n"
        "  float64 z\n\n"
        "# Angular velocity\n"
        "Vector3 angular\n"
        "  float64 x\n"
        "  float64 y\n"
        "  float64 z"
    ),
    ("interface", "show", "turtlesim/msg/Pose"): (
        "float32 x\n"
        "float32 y\n"
        "float32 theta\n"
        "float32 linear_velocity\n"
        "float32 angular_velocity"
    ),
    ("param", "get", "/turtlesim", "background_r"): "Integer value is: 69",
    ("param", "set", "/turtlesim", "background_r", "255"): "Set parameter successful",
}


@pytest.fixture()
def mock_ros2_cli(monkeypatch):
    """Patch ``subprocess.run`` so ``ros2 …`` commands return simulated output."""
    import subprocess as _sp

    _original_run = _sp.run

    def _fake_run(cmd, **kwargs):
        # cmd = ["ros2", "topic", "list", "-t"] → key = ("topic", "list", "-t")
        if cmd and cmd[0] == "ros2":
            key = tuple(cmd[1:])
            if key in SIMULATED_CLI:
                return _sp.CompletedProcess(cmd, 0, stdout=SIMULATED_CLI[key], stderr="")
            return _sp.CompletedProcess(
                cmd, 1, stdout="", stderr=f"simulated: unknown command {key}"
            )
        return _original_run(cmd, **kwargs)

    monkeypatch.setattr(_sp, "run", _fake_run)


# ──────────────────────────────────────────────────────────────────────────────
# Simulated Ollama API responses
# ──────────────────────────────────────────────────────────────────────────────


def make_ollama_text_response(content: str) -> dict:
    """Build a simulated Ollama /api/chat response with plain text."""
    return {"message": {"role": "assistant", "content": content}}


def make_ollama_tool_response(tool_calls: list[dict]) -> dict:
    """Build a simulated Ollama /api/chat response with tool calls.

    Each element of *tool_calls*: ``{"name": "...", "arguments": {...}}``.
    """
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": tc["name"], "arguments": tc.get("arguments", {})}}
                for tc in tool_calls
            ],
        }
    }
