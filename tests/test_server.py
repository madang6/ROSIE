"""
Tests for rosie.server — FastAPI endpoints.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from rosie.server import app


@pytest.fixture()
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAgentRunEndpoint:
    def test_valid_request(self, client):
        with patch("rosie.server.run_agent", return_value="The turtle is at (5.5, 5.5)."):
            resp = client.post(
                "/agent/run",
                json={"message": "Where is the turtle?"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "response" in data
            assert "5.5" in data["response"]

    def test_custom_model(self, client):
        with patch("rosie.server.run_agent", return_value="ok") as mock_agent:
            resp = client.post(
                "/agent/run",
                json={"message": "test", "model": "gemma3:12b"},
            )
            assert resp.status_code == 200
            mock_agent.assert_called_once_with("test", model="gemma3:12b")

    def test_missing_message_returns_422(self, client):
        resp = client.post("/agent/run", json={})
        assert resp.status_code == 422

    def test_empty_message_accepted(self, client):
        with patch("rosie.server.run_agent", return_value=""):
            resp = client.post("/agent/run", json={"message": ""})
            assert resp.status_code == 200

    def test_agent_error_propagates(self, client):
        with patch("rosie.server.run_agent", side_effect=RuntimeError("llm down")):
            with pytest.raises(RuntimeError, match="llm down"):
                client.post("/agent/run", json={"message": "test"})
