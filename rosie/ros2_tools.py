"""
ROS 2 introspection and control tools.

Every public function in this module is a *tool* the LLM can invoke.
Functions are intentionally thin wrappers around ``ros2`` CLI / ``rclpy`` so
that the agent discovers robot capabilities at runtime rather than at
compile time.
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from rosidl_runtime_py import get_interface_path
from rosidl_runtime_py.utilities import get_message, get_service, get_action

from rosie.config import TOPIC_READ_TIMEOUT_SEC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_node: Node | None = None
_lock = threading.Lock()


def _get_node() -> Node:
    """Return (and lazily create) a shared rclpy node."""
    global _node
    with _lock:
        if _node is None:
            if not rclpy.ok():
                rclpy.init()
            _node = rclpy.create_node("rosie_agent")
        return _node


def _cli(cmd: list[str], timeout: float = 5.0) -> str:
    """Run a ``ros2 …`` CLI command and return stdout."""
    result = subprocess.run(
        ["ros2"] + cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        return f"[error] {result.stderr.strip()}"
    return result.stdout.strip()


def _msg_to_dict(msg: Any) -> dict:
    """Recursively convert a ROS message to a JSON-serialisable dict."""
    from rosidl_runtime_py import message_to_ordereddict
    return dict(message_to_ordereddict(msg))


# QoS profile that works with most topic configurations
_BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

_RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------


def ros2_list_topics() -> str:
    """List all active ROS 2 topics with their message types."""
    return _cli(["topic", "list", "-t"])


def ros2_topic_info(topic_name: str) -> str:
    """Get detailed info about a topic (type, publisher/subscriber counts, QoS)."""
    return _cli(["topic", "info", topic_name, "-v"])


def ros2_list_services() -> str:
    """List all active ROS 2 services with their types."""
    return _cli(["service", "list", "-t"])


def ros2_list_actions() -> str:
    """List all active ROS 2 actions with their types."""
    return _cli(["action", "list", "-t"])


def ros2_list_nodes() -> str:
    """List all active ROS 2 nodes."""
    return _cli(["node", "list"])


def ros2_node_info(node_name: str) -> str:
    """Get publishers, subscribers, services, and actions of a specific node."""
    return _cli(["node", "info", node_name])


def ros2_describe_interface(interface_name: str) -> str:
    """Show the full definition of a message, service, or action type.

    Examples of *interface_name*:
      - ``geometry_msgs/msg/Twist``
      - ``std_srvs/srv/SetBool``
      - ``nav2_msgs/action/NavigateToPose``
    """
    return _cli(["interface", "show", interface_name])


# ---------------------------------------------------------------------------
# Observation tools
# ---------------------------------------------------------------------------


def ros2_read_topic(topic_name: str, timeout_sec: float | None = None) -> str:
    """Subscribe to *topic_name* and return the next message as JSON.

    Blocks for at most *timeout_sec* seconds (default from config).
    """
    timeout = timeout_sec or TOPIC_READ_TIMEOUT_SEC
    node = _get_node()

    # Discover the message type from the topic list
    topic_names_and_types = node.get_topic_names_and_types()
    msg_type_str: str | None = None
    for name, types in topic_names_and_types:
        if name == topic_name:
            msg_type_str = types[0]
            break
    if msg_type_str is None:
        return json.dumps({"error": f"Topic '{topic_name}' not found."})

    msg_class = get_message(msg_type_str)

    received: list[Any] = []
    event = threading.Event()

    def _cb(msg: Any) -> None:
        received.append(msg)
        event.set()

    # Try best-effort first (works with most sensor topics)
    sub = node.create_subscription(msg_class, topic_name, _cb, _BEST_EFFORT_QOS)
    event.wait(timeout=timeout)

    if not received:
        # Retry with reliable QoS (latched / transient-local topics)
        node.destroy_subscription(sub)
        sub = node.create_subscription(msg_class, topic_name, _cb, _RELIABLE_QOS)
        rclpy.spin_once(node, timeout_sec=timeout)

    node.destroy_subscription(sub)

    if received:
        return json.dumps(_msg_to_dict(received[0]), default=str)
    return json.dumps({"warning": f"No message received on '{topic_name}' within {timeout}s."})


def ros2_read_topic_stream(topic_name: str, count: int = 5, timeout_sec: float = 5.0) -> str:
    """Read *count* consecutive messages from *topic_name* and return as a JSON list.

    Useful for understanding publish rates and value trends.
    """
    node = _get_node()
    topic_names_and_types = node.get_topic_names_and_types()
    msg_type_str: str | None = None
    for name, types in topic_names_and_types:
        if name == topic_name:
            msg_type_str = types[0]
            break
    if msg_type_str is None:
        return json.dumps({"error": f"Topic '{topic_name}' not found."})

    msg_class = get_message(msg_type_str)
    received: list[Any] = []
    event = threading.Event()

    def _cb(msg: Any) -> None:
        received.append(_msg_to_dict(msg))
        if len(received) >= count:
            event.set()

    sub = node.create_subscription(msg_class, topic_name, _cb, _BEST_EFFORT_QOS)
    event.wait(timeout=timeout_sec)
    node.destroy_subscription(sub)

    return json.dumps(received, default=str)


# ---------------------------------------------------------------------------
# Command tools
# ---------------------------------------------------------------------------


def ros2_publish(topic_name: str, msg_type: str, data: dict) -> str:
    """Publish a single message to a topic.

    *msg_type* must be the full interface name, e.g.
    ``geometry_msgs/msg/Twist``.  *data* is a dict matching the message
    fields.
    """
    node = _get_node()
    msg_class = get_message(msg_type)
    pub = node.create_publisher(msg_class, topic_name, 10)

    msg = msg_class()
    _fill_message(msg, data)
    pub.publish(msg)
    # Keep publisher alive briefly so the message is actually sent
    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_publisher(pub)
    return json.dumps({"status": "published", "topic": topic_name})


def ros2_publish_repeated(
    topic_name: str, msg_type: str, data: dict, rate_hz: float = 10.0, duration_sec: float = 1.0
) -> str:
    """Publish a message repeatedly at *rate_hz* for *duration_sec*.

    Needed for velocity commands that require sustained publishing (e.g.
    ``/cmd_vel``).
    """
    import time

    node = _get_node()
    msg_class = get_message(msg_type)
    pub = node.create_publisher(msg_class, topic_name, 10)
    msg = msg_class()
    _fill_message(msg, data)

    period = 1.0 / rate_hz
    end_time = time.monotonic() + duration_sec
    count = 0
    while time.monotonic() < end_time:
        pub.publish(msg)
        count += 1
        rclpy.spin_once(node, timeout_sec=0.01)
        time.sleep(period)

    node.destroy_publisher(pub)
    return json.dumps({
        "status": "published",
        "topic": topic_name,
        "messages_sent": count,
        "duration_sec": duration_sec,
    })


def ros2_call_service(service_name: str, service_type: str, request: dict) -> str:
    """Call a ROS 2 service and return the response.

    *service_type* example: ``std_srvs/srv/SetBool``
    """
    node = _get_node()
    srv_class = get_service(service_type)
    client = node.create_client(srv_class, service_name)

    if not client.wait_for_service(timeout_sec=5.0):
        node.destroy_client(client)
        return json.dumps({"error": f"Service '{service_name}' not available."})

    req = srv_class.Request()
    _fill_message(req, request)
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    node.destroy_client(client)

    if future.result() is not None:
        return json.dumps(_msg_to_dict(future.result()), default=str)
    return json.dumps({"error": "Service call timed out or failed."})


def ros2_send_action_goal(action_name: str, action_type: str, goal: dict) -> str:
    """Send a goal to a ROS 2 action server and return the result.

    *action_type* example: ``nav2_msgs/action/NavigateToPose``
    """
    from rclpy.action import ActionClient

    node = _get_node()
    act_class = get_action(action_type)
    client = ActionClient(node, act_class, action_name)

    if not client.wait_for_server(timeout_sec=5.0):
        client.destroy()
        return json.dumps({"error": f"Action server '{action_name}' not available."})

    goal_msg = act_class.Goal()
    _fill_message(goal_msg, goal)
    send_future = client.send_goal_async(goal_msg)
    rclpy.spin_until_future_complete(node, send_future, timeout_sec=10.0)
    goal_handle = send_future.result()

    if not goal_handle or not goal_handle.accepted:
        client.destroy()
        return json.dumps({"error": "Goal rejected by action server."})

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=60.0)
    client.destroy()

    if result_future.result() is not None:
        return json.dumps(_msg_to_dict(result_future.result().result), default=str)
    return json.dumps({"error": "Action did not return a result in time."})


# ---------------------------------------------------------------------------
# Utility tools
# ---------------------------------------------------------------------------


def ros2_get_param(node_name: str, param_name: str) -> str:
    """Get a parameter value from a running node."""
    return _cli(["param", "get", node_name, param_name])


def ros2_set_param(node_name: str, param_name: str, value: str) -> str:
    """Set a parameter on a running node."""
    return _cli(["param", "set", node_name, param_name, value])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fill_message(msg: Any, data: dict) -> None:
    """Recursively populate a ROS message from a plain dict."""
    for key, value in data.items():
        if not hasattr(msg, key):
            continue
        attr = getattr(msg, key)
        if isinstance(value, dict):
            _fill_message(attr, value)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            # List of sub-messages
            sub_type = type(attr[0]) if len(attr) > 0 else None
            if sub_type is None:
                # Try to infer from the slot type
                setattr(msg, key, value)
            else:
                items = []
                for v in value:
                    sub = sub_type()
                    _fill_message(sub, v)
                    items.append(sub)
                setattr(msg, key, items)
        else:
            setattr(msg, key, value)
