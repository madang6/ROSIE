"""
ROSIE agent — agentic loop that connects Ollama tool-use to ROS 2.

The loop follows the standard ReAct pattern:
  1. Send conversation + tool definitions to Ollama.
  2. If the model returns tool calls, execute them and append results.
  3. Repeat until the model returns a plain text response or we hit the
     iteration limit.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import requests

from rosie.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    AGENT_MAX_ITERATIONS,
    SYSTEM_PROMPT,
)
from rosie.tool_registry import TOOL_DEFINITIONS, TOOL_FUNCTIONS

log = logging.getLogger("rosie.agent")

# ---------------------------------------------------------------------------
# Ollama chat helpers
# ---------------------------------------------------------------------------


def _chat(messages: list[dict], model: str = OLLAMA_MODEL) -> dict:
    """Call the Ollama ``/api/chat`` endpoint with tool definitions."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call to the matching Python function."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    try:
        return fn(**arguments)
    except Exception as exc:
        log.exception("Tool %s failed", name)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_agent(user_message: str, *, model: str | None = None) -> str:
    """Run the full agent loop for a single user request.

    Returns the final assistant text response.
    """
    model = model or OLLAMA_MODEL
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for iteration in range(1, AGENT_MAX_ITERATIONS + 1):
        log.info("--- iteration %d ---", iteration)
        response = _chat(messages, model=model)
        assistant_msg = response.get("message", {})
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            # Model produced a final text answer
            return assistant_msg.get("content", "")

        # Execute each tool call and feed results back
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"].get("arguments", {})
            log.info("tool call: %s(%s)", fn_name, json.dumps(fn_args, default=str))

            result = _execute_tool(fn_name, fn_args)
            log.info("tool result (truncated): %.500s", result)

            messages.append({
                "role": "tool",
                "content": result,
            })

    return "[ROSIE] Reached maximum iteration limit.  Please refine your request."


def run_interactive() -> None:
    """Run an interactive REPL in the terminal."""
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("ROSIE — ROS 2 Natural-Language Agent")
    print("Type your command (Ctrl-C to quit).\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user_input:
            continue
        answer = run_agent(user_input)
        print(f"\nrosie> {answer}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(run_agent(" ".join(sys.argv[1:])))
    else:
        run_interactive()
