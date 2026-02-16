"""Evaluation: Conversation Memory

Tests multi-turn conversation capabilities:
- Follow-up questions resolve correctly using prior context
- Context carries across turns
- Memory doesn't bleed incorrect context into unrelated questions
"""

import pytest

from src.config import get_settings
from src.memory import ConversationMemory
from src.pipeline import build_pipeline


@pytest.fixture(scope="module")
def agent():
    """Build the pipeline once for all memory eval tests."""
    settings = get_settings()
    settings.bookmark_sync_enabled = False
    return build_pipeline(settings)


class TestMemoryFollowUp:
    """Tests for follow-up question resolution with conversation memory."""

    def test_follow_up_resolves_correctly(self, agent):
        """A follow-up question should use prior context to resolve references."""
        memory = ConversationMemory(max_turns=10)

        # Turn 1: Ask about the project deadline
        result1 = agent.ask(
            "What is Project Alpha's deadline?",
            message_history=memory.get_history(),
        )
        memory.add_turn("What is Project Alpha's deadline?", result1.answer)
        assert "March 30" in result1.answer or "2024" in result1.answer

        # Turn 2: Follow-up referencing "it"
        result2 = agent.ask(
            "What tech stack does it use?",
            message_history=memory.get_history(),
        )
        # Should resolve "it" to Project Alpha and mention its tech stack
        answer_lower = result2.answer.lower()
        assert any(
            term in answer_lower
            for term in ["python", "fastapi", "react", "typescript", "postgresql"]
        ), f"Follow-up didn't resolve 'it' to Project Alpha. Got: {result2.answer}"

    def test_context_carries_across_turns(self, agent):
        """Information from earlier turns should influence later responses."""
        memory = ConversationMemory(max_turns=10)

        # Turn 1: Ask about meeting attendees
        result1 = agent.ask(
            "Who attended the January meeting?",
            message_history=memory.get_history(),
        )
        memory.add_turn("Who attended the January meeting?", result1.answer)
        assert "Sarah" in result1.answer or "Marcus" in result1.answer

        # Turn 2: Ask about what was decided
        result2 = agent.ask(
            "What did they decide about database migration?",
            message_history=memory.get_history(),
        )
        # Should resolve "they" to the meeting attendees and mention Alembic
        assert "Alembic" in result2.answer or "alembic" in result2.answer.lower()

    def test_no_context_bleed(self, agent):
        """Memory from one topic should not bleed into an unrelated question."""
        memory = ConversationMemory(max_turns=10)

        # Turn 1: Ask about sourdough
        result1 = agent.ask(
            "How long should I proof sourdough?",
            message_history=memory.get_history(),
        )
        memory.add_turn("How long should I proof sourdough?", result1.answer)

        # Turn 2: Ask about something completely different
        result2 = agent.ask(
            "What is gradient descent?",
            message_history=memory.get_history(),
        )
        # Answer about gradient descent should NOT mention sourdough
        assert "sourdough" not in result2.answer.lower()
        assert "bread" not in result2.answer.lower()
        # Should mention optimization or learning
        answer_lower = result2.answer.lower()
        assert any(
            term in answer_lower
            for term in ["optimization", "loss", "parameter", "learning", "gradient"]
        )


def test_memory_follow_up(agent):
    """Run memory follow-up test as a standalone test function."""
    test = TestMemoryFollowUp()
    test.test_follow_up_resolves_correctly(agent)


def test_memory_context_carries(agent):
    """Run memory context test as a standalone test function."""
    test = TestMemoryFollowUp()
    test.test_context_carries_across_turns(agent)


def test_memory_no_bleed(agent):
    """Run memory no-bleed test as a standalone test function."""
    test = TestMemoryFollowUp()
    test.test_no_context_bleed(agent)
