"""Tests for ConversationMemory."""

from pydantic_ai.messages import ModelRequest, ModelResponse

from src.memory import ConversationMemory


class TestConversationMemory:
    """Tests for the ConversationMemory class."""

    def test_empty_history(self):
        """New memory should have empty history."""
        memory = ConversationMemory()
        assert memory.get_history() == []
        assert memory.turn_count == 0

    def test_add_turn(self):
        """Adding a turn should store it in history."""
        memory = ConversationMemory()
        memory.add_turn("What is X?", "X is Y.")
        assert memory.turn_count == 1
        history = memory.get_history()
        assert len(history) == 2  # request + response

    def test_history_format(self):
        """History should contain alternating ModelRequest and ModelResponse."""
        memory = ConversationMemory()
        memory.add_turn("Question 1", "Answer 1")
        history = memory.get_history()
        assert isinstance(history[0], ModelRequest)
        assert isinstance(history[1], ModelResponse)

    def test_multiple_turns(self):
        """Multiple turns should be stored in order."""
        memory = ConversationMemory()
        memory.add_turn("Q1", "A1")
        memory.add_turn("Q2", "A2")
        assert memory.turn_count == 2
        history = memory.get_history()
        assert len(history) == 4  # 2 turns * 2 messages each

    def test_max_turns_limit(self):
        """History should not exceed max_turns."""
        memory = ConversationMemory(max_turns=2)
        memory.add_turn("Q1", "A1")
        memory.add_turn("Q2", "A2")
        memory.add_turn("Q3", "A3")
        assert memory.turn_count == 2
        # Oldest turn (Q1/A1) should be dropped
        history = memory.get_history()
        assert len(history) == 4  # 2 turns * 2 messages
        # First message should be Q2, not Q1
        request = history[0]
        assert isinstance(request, ModelRequest)
        assert request.parts[0].content == "Q2"

    def test_clear(self):
        """clear() should remove all history."""
        memory = ConversationMemory()
        memory.add_turn("Q1", "A1")
        memory.add_turn("Q2", "A2")
        memory.clear()
        assert memory.turn_count == 0
        assert memory.get_history() == []

    def test_max_turns_property(self):
        """max_turns property should return the configured limit."""
        memory = ConversationMemory(max_turns=5)
        assert memory.max_turns == 5

    def test_default_max_turns(self):
        """Default max_turns should be 10."""
        memory = ConversationMemory()
        assert memory.max_turns == 10

    def test_user_message_preserved(self):
        """User message text should be preserved in history."""
        memory = ConversationMemory()
        memory.add_turn("What is the deadline?", "March 30, 2024")
        history = memory.get_history()
        request = history[0]
        assert request.parts[0].content == "What is the deadline?"

    def test_assistant_message_preserved(self):
        """Assistant message text should be preserved in history."""
        memory = ConversationMemory()
        memory.add_turn("What is the deadline?", "March 30, 2024")
        history = memory.get_history()
        response = history[1]
        assert response.parts[0].content == "March 30, 2024"
