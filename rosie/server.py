"""
FastAPI sidecar that exposes the ROSIE agent over HTTP.

This runs inside the ROS 2 container so it has access to the full ROS graph.
OpenWebUI's pipe (or any HTTP client) can POST natural-language commands here.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from rosie.agent import run_agent
from rosie.config import OLLAMA_MODEL

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

app = FastAPI(title="ROSIE", version="0.1.0")


class AgentRequest(BaseModel):
    message: str
    model: str = OLLAMA_MODEL


class AgentResponse(BaseModel):
    response: str


@app.post("/agent/run", response_model=AgentResponse)
def agent_run(req: AgentRequest) -> AgentResponse:
    """Execute the ROSIE agent loop for a single user message."""
    result = run_agent(req.message, model=req.model)
    return AgentResponse(response=result)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
