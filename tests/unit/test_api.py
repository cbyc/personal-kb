"""Tests for the REST API."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.memory import ConversationMemory
from src.models import QueryResult

import api as api_module


@pytest.fixture
def mock_agent():
    agent = AsyncMock()
    agent.ask_async.return_value = QueryResult(
        answer="Project Alpha uses microservices.",
        sources=["data/notes/project_alpha.txt"],
    )
    return agent


@pytest.fixture
def client(mock_agent):
    # Set module-level globals so endpoints use the mock
    api_module.agent = mock_agent
    api_module.memory = ConversationMemory(max_turns=10)

    # Patch build_pipeline so lifespan doesn't run real indexing
    with patch("api.build_pipeline", return_value=mock_agent):
        with TestClient(api_module.app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestQueryEndpoint:
    def test_successful_query(self, client, mock_agent):
        response = client.post("/api/v1/query", json={"question": "What is project Alpha?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Project Alpha uses microservices."
        assert data["sources"] == ["data/notes/project_alpha.txt"]
        mock_agent.ask_async.assert_called_once()

    def test_empty_question_rejected(self, client):
        response = client.post("/api/v1/query", json={"question": ""})
        assert response.status_code == 422

    def test_missing_question_field(self, client):
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_question_too_long(self, client):
        long_question = "x" * 1001
        response = client.post("/api/v1/query", json={"question": long_question})
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "bad_request"

    def test_conversation_memory_updated(self, client, mock_agent):
        client.post("/api/v1/query", json={"question": "First question"})
        client.post("/api/v1/query", json={"question": "Follow-up question"})

        assert api_module.memory.turn_count == 2
        # Second call should include history from first turn
        second_call = mock_agent.ask_async.call_args_list[1]
        history = second_call.kwargs.get("message_history") or second_call[1].get("message_history")
        assert len(history) == 2  # 1 turn = request + response

    def test_query_with_no_sources(self, client, mock_agent):
        mock_agent.ask_async.return_value = QueryResult(
            answer="I don't have information about that.",
            sources=[],
        )
        response = client.post("/api/v1/query", json={"question": "Unknown topic?"})
        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []
