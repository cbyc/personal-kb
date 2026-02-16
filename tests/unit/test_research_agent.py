"""Tests for the ResearchAgent."""

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from src.agents.research import ResearchAgent
from src.models import KBResponse


@pytest.fixture
def research_agent() -> ResearchAgent:
    """A ResearchAgent instance for testing."""
    return ResearchAgent()


class TestResearchAgentSynthesize:
    """Tests for the ResearchAgent.synthesize() method."""

    def test_returns_kb_response(self, research_agent: ResearchAgent):
        """synthesize() should return a KBResponse object."""
        context = (
            "[Source: project_alpha.txt | Type: note]\nProject Alpha deadline is March 30, 2024."
        )
        result = research_agent.synthesize("What is Project Alpha's deadline?", context)
        assert isinstance(result, KBResponse)

    def test_answer_contains_relevant_info(self, research_agent: ResearchAgent):
        """synthesize() should include relevant information from context."""
        context = (
            "[Source: project_alpha.txt | Type: note]\nProject Alpha deadline is March 30, 2024."
        )
        result = research_agent.synthesize("What is Project Alpha's deadline?", context)
        assert "March 30" in result.answer or "2024" in result.answer

    def test_sources_populated(self, research_agent: ResearchAgent):
        """synthesize() should populate the sources list."""
        context = (
            "[Source: project_alpha.txt | Type: note]\nProject Alpha deadline is March 30, 2024."
        )
        result = research_agent.synthesize("What is Project Alpha's deadline?", context)
        assert len(result.sources) > 0

    def test_no_info_response(self, research_agent: ResearchAgent):
        """synthesize() should indicate no info when context is empty."""
        context = "No relevant information found in the knowledge base."
        result = research_agent.synthesize("What is quantum computing?", context)
        assert "don't have" in result.answer.lower() or "no information" in result.answer.lower()


class TestResearchAgentMultiSource:
    """Tests for multi-source synthesis."""

    def test_multi_source_synthesis(self, research_agent: ResearchAgent):
        """synthesize() should combine information from multiple sources."""
        context = (
            "[Source: project_alpha.txt | Type: note]\n"
            "Project Alpha uses Python and FastAPI.\n\n---\n\n"
            "[Source: meeting_2024_01.txt | Type: note]\n"
            "The team decided to use Alembic for database migrations."
        )
        result = research_agent.synthesize("What tech stack does the project use?", context)
        assert isinstance(result, KBResponse)
        assert len(result.sources) > 0


class TestResearchAgentFollowUp:
    """Tests for follow-up question handling with conversation history."""

    def test_follow_up_with_no_chunks_accepts_empty_sources(self, research_agent: ResearchAgent):
        """Follow-up answered from history alone should not crash on empty sources."""
        context = "No relevant information found in the knowledge base."
        history = [
            ModelRequest(parts=[UserPromptPart(content="What is Project Alpha's deadline?")]),
            ModelResponse(parts=[TextPart(content="Project Alpha's deadline is March 30, 2024.")]),
        ]
        result = research_agent.synthesize(
            "Tell me more about that deadline", context, message_history=history
        )
        assert isinstance(result, KBResponse)

    def test_follow_up_with_chunks_returns_sources(self, research_agent: ResearchAgent):
        """Follow-up with fresh chunks should still populate sources."""
        context = (
            "[Source: project_alpha.txt | Type: note]\n"
            "Project Alpha uses Python and FastAPI for the backend."
        )
        history = [
            ModelRequest(parts=[UserPromptPart(content="What is Project Alpha?")]),
            ModelResponse(parts=[TextPart(content="Project Alpha is a customer-facing web app.")]),
        ]
        result = research_agent.synthesize(
            "What tech stack does it use?", context, message_history=history
        )
        assert isinstance(result, KBResponse)
        assert len(result.sources) > 0
