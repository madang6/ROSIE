# ROSIE — ROS 2 Orchestration through Semantic Introspection and Execution

ROSIE is a natural-language agent that discovers and commands **any** ROS 2
robot without being told what the robot is.  It connects a local LLM (via
[Ollama](https://ollama.com)) to the live ROS 2 computation graph through
tool-use, following a discover → identify → plan → execute → report loop.

```
┌─────────────┐      ┌─────────────┐      ┌───────────────────────┐
│  Open WebUI │─────▶│   Ollama    │◀────▶│      ROSIE Agent      │
│  (chat UI)  │      │  (LLM API)  │      │  (FastAPI + rclpy)    │
└─────────────┘      └─────────────┘      └───────────┬───────────┘
                                                      │ tool calls
                                          ┌───────────▼───────────┐
                                          │    ROS 2 Graph        │
                                          │ topics / services /   │
                                          │ actions / parameters  │
                                          └───────────────────────┘
```

## How It Works

The LLM has **no hardcoded knowledge** of the robot.  Instead, it has access
to 15 tools that let it introspect and command the ROS 2 graph at runtime:

| Category      | Tools                                                                 |
|---------------|-----------------------------------------------------------------------|
| **Discovery** | `ros2_list_topics`, `ros2_list_services`, `ros2_list_actions`, `ros2_list_nodes`, `ros2_node_info`, `ros2_topic_info`, `ros2_describe_interface` |
| **Observation** | `ros2_read_topic`, `ros2_read_topic_stream`                        |
| **Command**   | `ros2_publish`, `ros2_publish_repeated`, `ros2_call_service`, `ros2_send_action_goal` |
| **Parameters** | `ros2_get_param`, `ros2_set_param`                                  |

When a user says *"drive forward 1 meter"*, ROSIE:

1. **Discovers** topics → sees `/cmd_vel` (geometry_msgs/msg/Twist), `/odom`
2. **Describes** the Twist message → learns the field layout
3. **Reads** `/odom` → gets the current position
4. **Publishes** a velocity command to `/cmd_vel` (repeated at 10 Hz)
5. **Reads** `/odom` again → confirms it moved ~1 m
6. **Reports** back to the user

The same agent works with a TurtleBot, a drone, a manipulator arm, or any
ROS 2 system — because it introspects rather than assumes.

## Quick Start

### Prerequisites

- Docker + Docker Compose (with NVIDIA container toolkit for GPU inference)
- A running ROS 2 system (robot, simulator, or even just `turtlesim`)

### 1. Start the stack

```bash
# Clone both repos
git clone https://github.com/maximilianadang/localmodels
git clone <this-repo> ROSIE

# Start Ollama + Open WebUI + ROSIE
cd ROSIE
docker compose up -d

# Pull a tool-use capable model
docker exec ollama ollama pull qwen3:14b
```

### 2. Launch a test robot (optional)

```bash
# In a separate terminal with ROS 2 sourced:
ros2 run turtlesim turtlesim_node
```

### 3. Chat with ROSIE

Open http://localhost:8080 and select the **ROSIE** model in the dropdown.
Then try:

- *"What robot is connected?"*
- *"Move the turtle forward"*
- *"Draw a square"*

Or use the CLI:

```bash
docker exec -it rosie python3 -m rosie.agent
```

## Architecture

```
ROSIE/
├── rosie/
│   ├── __init__.py
│   ├── config.py           # Environment-driven configuration
│   ├── ros2_tools.py       # ROS 2 introspection & control (rclpy + CLI)
│   ├── tool_registry.py    # Tool schemas (Ollama function-calling format)
│   ├── agent.py            # ReAct agent loop
│   └── server.py           # FastAPI sidecar (HTTP API for the pipe)
├── openwebui/
│   └── rosie_pipe.py       # OpenWebUI pipe — routes chat to the agent
├── docker-compose.yml      # Full stack: Ollama + Open WebUI + ROSIE
├── Dockerfile              # ROS 2 Humble + Python agent
└── requirements.txt
```

### Key Design Decisions

**Tool-use over prompt engineering.**  Rather than describing the robot in the
system prompt, we give the model tools to discover it.  This means the same
agent works with any robot without reconfiguration.

**ROS 2 as the universal API.**  Topics, services, and actions are already a
well-structured API for robot state and control.  The agent treats them as
such — no custom middleware needed.

**Sidecar architecture.**  ROSIE runs in its own container with ROS 2
installed.  It joins the DDS network (via host networking) to talk to the
robot, and exposes an HTTP API that OpenWebUI's pipe calls.  This keeps the
LLM stack and the ROS stack cleanly separated.

**Ollama for local inference.**  No cloud API keys, no data leaving the
network.  Models like Qwen 3 14B support function calling well enough for
this use case.

## Configuration

All settings are controlled via environment variables:

| Variable               | Default                     | Description                          |
|------------------------|-----------------------------|--------------------------------------|
| `OLLAMA_HOST`          | `http://ollama:11434`       | Ollama API endpoint                  |
| `OLLAMA_MODEL`         | `qwen3:14b`                 | Model to use for reasoning           |
| `AGENT_MAX_ITERATIONS` | `25`                        | Max tool-use loops per request       |
| `TOPIC_READ_TIMEOUT_SEC` | `2.0`                     | Timeout when reading a topic         |
| `ROS_DOMAIN_ID`        | `0`                         | ROS 2 domain ID                      |

## Adding Custom Tools

Add a function to `rosie/ros2_tools.py`, then register it in
`rosie/tool_registry.py` with a schema and dispatch entry.  The agent will
automatically discover and use it.

## License

MIT
