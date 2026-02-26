"""
OpenWebUI *Pipe* that exposes ROSIE as a selectable model inside the web UI.

Install by copying this file into OpenWebUI's ``pipelines/`` directory (or
mounting it via Docker volume).  Once loaded, a model called "ROSIE" will
appear in the model dropdown.  Every chat message is routed through the
ROSIE agent loop, which introspects and commands the live ROS 2 graph.

Reference: https://docs.openwebui.com/pipelines/
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Generator

import requests

log = logging.getLogger("rosie.pipe")

# ---------------------------------------------------------------------------
# Configuration — matches rosie/config.py defaults
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "25"))

# URL of the ROSIE FastAPI sidecar that executes ROS 2 tools on the host.
# When running in the same docker-compose, this is the service name.
ROSIE_API_URL = os.getenv("ROSIE_API_URL", "http://rosie:8000")

# ---------------------------------------------------------------------------
# Pipe metadata (used by OpenWebUI)
# ---------------------------------------------------------------------------


class Pipe:
    """OpenWebUI pipeline that routes messages through the ROSIE agent."""

    class Valves:  # noqa: D106  (OpenWebUI naming convention)
        OLLAMA_HOST: str = OLLAMA_HOST
        OLLAMA_MODEL: str = OLLAMA_MODEL
        ROSIE_API_URL: str = ROSIE_API_URL
        AGENT_MAX_ITERATIONS: int = AGENT_MAX_ITERATIONS

    def __init__(self) -> None:
        self.name = "ROSIE"
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # The ``pipe`` method is the entry-point called by OpenWebUI.
    # ------------------------------------------------------------------

    def pipe(self, body: dict[str, Any]) -> str | Generator[str, None, None]:
        """Handle a chat completion request from OpenWebUI.

        We delegate to the ROSIE sidecar, which runs in a ROS 2 environment
        and has direct access to the ROS graph.
        """
        messages = body.get("messages", [])
        if not messages:
            return "No messages received."

        user_msg = messages[-1].get("content", "")
        log.info("Pipe received: %s", user_msg[:120])

        try:
            resp = requests.post(
                f"{self.valves.ROSIE_API_URL}/agent/run",
                json={
                    "message": user_msg,
                    "model": self.valves.OLLAMA_MODEL,
                },
                timeout=180,
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("response", "[ROSIE] Empty response from agent.")
        except requests.RequestException as exc:
            log.exception("ROSIE API call failed")
            return f"[ROSIE] Failed to reach agent backend: {exc}"
