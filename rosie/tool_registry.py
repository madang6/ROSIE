"""
Tool registry — maps Python callables to Ollama function-calling schemas.

Each tool is defined once here.  The registry is consumed by both the agent
loop (to dispatch calls) and the Ollama API request (to advertise available
tools).
"""

from __future__ import annotations

from typing import Any, Callable

from rosie import ros2_tools
from rosie import heart_tools

# ---------------------------------------------------------------------------
# Schema definitions (Ollama / OpenAI-compatible function calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # ---- Discovery ----
    {
        "type": "function",
        "function": {
            "name": "ros2_list_topics",
            "description": (
                "List all active ROS 2 topics with their message types. "
                "Use this first to discover what the robot publishes and subscribes to."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_topic_info",
            "description": (
                "Get detailed info about a specific topic: message type, "
                "publisher/subscriber counts, and QoS profiles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_name": {
                        "type": "string",
                        "description": "Full topic name, e.g. '/cmd_vel'.",
                    }
                },
                "required": ["topic_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_list_services",
            "description": "List all active ROS 2 services with their types.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_list_actions",
            "description": "List all active ROS 2 actions with their types.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_list_nodes",
            "description": "List all active ROS 2 nodes in the graph.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_node_info",
            "description": (
                "Get detailed info about a node: its publishers, subscribers, "
                "service servers/clients, and action servers/clients."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_name": {
                        "type": "string",
                        "description": "Full node name, e.g. '/turtlesim'.",
                    }
                },
                "required": ["node_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_describe_interface",
            "description": (
                "Show the full definition of a message, service, or action type. "
                "Always call this before publishing or calling a service so you "
                "know the exact field names and types."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "interface_name": {
                        "type": "string",
                        "description": (
                            "Full interface name, e.g. 'geometry_msgs/msg/Twist', "
                            "'std_srvs/srv/SetBool', 'nav2_msgs/action/NavigateToPose'."
                        ),
                    }
                },
                "required": ["interface_name"],
            },
        },
    },
    # ---- Observation ----
    {
        "type": "function",
        "function": {
            "name": "ros2_read_topic",
            "description": (
                "Subscribe and read the latest message from a topic. Returns "
                "the message as JSON. Use this to observe robot state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_name": {
                        "type": "string",
                        "description": "Full topic name to read from.",
                    },
                    "timeout_sec": {
                        "type": "number",
                        "description": "Max seconds to wait for a message (default 2.0).",
                    },
                },
                "required": ["topic_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_read_topic_stream",
            "description": (
                "Read multiple consecutive messages from a topic. Returns a "
                "JSON array. Useful for checking publish rates and trends."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_name": {"type": "string"},
                    "count": {
                        "type": "integer",
                        "description": "Number of messages to collect (default 5).",
                    },
                    "timeout_sec": {
                        "type": "number",
                        "description": "Max seconds to wait (default 5.0).",
                    },
                },
                "required": ["topic_name"],
            },
        },
    },
    # ---- Command ----
    {
        "type": "function",
        "function": {
            "name": "ros2_publish",
            "description": (
                "Publish a single message to a topic. Use describe_interface "
                "first to learn the correct field names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_name": {"type": "string"},
                    "msg_type": {
                        "type": "string",
                        "description": "Full message type, e.g. 'geometry_msgs/msg/Twist'.",
                    },
                    "data": {
                        "type": "object",
                        "description": "Message fields as a JSON object.",
                    },
                },
                "required": ["topic_name", "msg_type", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_publish_repeated",
            "description": (
                "Publish a message repeatedly at a given rate for a duration. "
                "Required for velocity commands (e.g. /cmd_vel) that need "
                "sustained publishing to keep the robot moving."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_name": {"type": "string"},
                    "msg_type": {"type": "string"},
                    "data": {"type": "object"},
                    "rate_hz": {
                        "type": "number",
                        "description": "Publish rate in Hz (default 10).",
                    },
                    "duration_sec": {
                        "type": "number",
                        "description": "How long to publish in seconds (default 1.0).",
                    },
                },
                "required": ["topic_name", "msg_type", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_call_service",
            "description": "Call a ROS 2 service and return the response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "service_type": {
                        "type": "string",
                        "description": "Full service type, e.g. 'std_srvs/srv/SetBool'.",
                    },
                    "request": {
                        "type": "object",
                        "description": "Service request fields as a JSON object.",
                    },
                },
                "required": ["service_name", "service_type", "request"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_send_action_goal",
            "description": (
                "Send a goal to a ROS 2 action server and wait for the result. "
                "Preferred over raw publishes for long-running tasks like navigation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_name": {"type": "string"},
                    "action_type": {
                        "type": "string",
                        "description": "Full action type, e.g. 'nav2_msgs/action/NavigateToPose'.",
                    },
                    "goal": {
                        "type": "object",
                        "description": "Goal fields as a JSON object.",
                    },
                },
                "required": ["action_name", "action_type", "goal"],
            },
        },
    },
    # ---- Parameters ----
    {
        "type": "function",
        "function": {
            "name": "ros2_get_param",
            "description": "Get a parameter value from a running ROS 2 node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_name": {"type": "string"},
                    "param_name": {"type": "string"},
                },
                "required": ["node_name", "param_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ros2_set_param",
            "description": "Set a parameter value on a running ROS 2 node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_name": {"type": "string"},
                    "param_name": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["node_name", "param_name", "value"],
            },
        },
    },
    # ---- Heart (state estimation + control loop) ----
    {
        "type": "function",
        "function": {
            "name": "heart_get_state",
            "description": (
                "Get the current fused vehicle state from the Heart: position, "
                "velocity, orientation, joints, and active image topics. "
                "Faster than reading individual topics."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heart_get_status",
            "description": (
                "Get the full Heart status: active controller, current objective, "
                "fused state, and configured topics."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heart_set_velocity",
            "description": (
                "Command the robot to move at a sustained velocity. Unlike "
                "ros2_publish_repeated (which stops after a duration), this "
                "persists until you call heart_stop or set a new objective."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vx": {"type": "number", "description": "Forward velocity m/s (default 0)."},
                    "vy": {"type": "number", "description": "Lateral velocity m/s (default 0)."},
                    "vz": {"type": "number", "description": "Vertical velocity m/s (default 0)."},
                    "yaw_rate": {"type": "number", "description": "Yaw rate rad/s (default 0)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heart_go_to_position",
            "description": (
                "Navigate the robot to a target position using PID feedback "
                "control. The Heart continuously reads state and adjusts "
                "commands until the target is reached."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Target X position (metres)."},
                    "y": {"type": "number", "description": "Target Y position (metres)."},
                    "z": {"type": "number", "description": "Target Z position (metres, default 0)."},
                    "yaw": {"type": "number", "description": "Target yaw (radians, optional)."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heart_stop",
            "description": "Stop the robot by setting the objective to idle.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heart_switch_controller",
            "description": (
                "Switch the active controller (e.g. 'passthrough', 'pid_position', "
                "'learned'). Use heart_get_status to see available controllers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "controller_name": {"type": "string", "description": "Name of the controller to activate."},
                },
                "required": ["controller_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heart_set_trajectory",
            "description": (
                "Set a multi-waypoint trajectory for the robot to follow. "
                "Each waypoint is [x, y, z]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "waypoints": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "description": "List of [x, y, z] waypoints.",
                    },
                },
                "required": ["waypoints"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatch map  (name → callable)
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "ros2_list_topics": ros2_tools.ros2_list_topics,
    "ros2_topic_info": ros2_tools.ros2_topic_info,
    "ros2_list_services": ros2_tools.ros2_list_services,
    "ros2_list_actions": ros2_tools.ros2_list_actions,
    "ros2_list_nodes": ros2_tools.ros2_list_nodes,
    "ros2_node_info": ros2_tools.ros2_node_info,
    "ros2_describe_interface": ros2_tools.ros2_describe_interface,
    "ros2_read_topic": ros2_tools.ros2_read_topic,
    "ros2_read_topic_stream": ros2_tools.ros2_read_topic_stream,
    "ros2_publish": ros2_tools.ros2_publish,
    "ros2_publish_repeated": ros2_tools.ros2_publish_repeated,
    "ros2_call_service": ros2_tools.ros2_call_service,
    "ros2_send_action_goal": ros2_tools.ros2_send_action_goal,
    "ros2_get_param": ros2_tools.ros2_get_param,
    "ros2_set_param": ros2_tools.ros2_set_param,
    # Heart tools
    "heart_get_state": heart_tools.heart_get_state,
    "heart_get_status": heart_tools.heart_get_status,
    "heart_set_velocity": heart_tools.heart_set_velocity,
    "heart_go_to_position": heart_tools.heart_go_to_position,
    "heart_stop": heart_tools.heart_stop,
    "heart_switch_controller": heart_tools.heart_switch_controller,
    "heart_set_trajectory": heart_tools.heart_set_trajectory,
}
