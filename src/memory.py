"""Conversation memory for session-scoped multi-turn conversations."""

from collections import deque

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


class ConversationMemory:
    """Manages conversation history within a session.

    Stores the last N turns (user + assistant pairs) in memory.
    History is compatible with pydantic-ai's message_history parameter.
    No persistence across sessions.
    """

    def __init__(self, max_turns: int = 10):
        """Initialize conversation memory.

        Args:
            max_turns: Maximum number of turns (Q&A pairs) to keep.
                Oldest turns are dropped when the limit is exceeded.
        """
        self._max_turns = max_turns
        self._turns: deque[tuple[ModelRequest, ModelResponse]] = deque(maxlen=max_turns)

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Record a conversation turn.

        Args:
            user_message: The user's question.
            assistant_message: The assistant's response.
        """
        request = ModelRequest(parts=[UserPromptPart(content=user_message)])
        response = ModelResponse(parts=[TextPart(content=assistant_message)])
        self._turns.append((request, response))

    def get_history(self) -> list[ModelMessage]:
        """Get conversation history in pydantic-ai message format.

        Returns:
            List of ModelMessage objects (alternating request/response)
            suitable for passing as message_history to Agent.run().
        """
        messages: list[ModelMessage] = []
        for request, response in self._turns:
            messages.append(request)
            messages.append(response)
        return messages

    def clear(self) -> None:
        """Clear all conversation history."""
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        """Number of turns currently stored."""
        return len(self._turns)

    @property
    def max_turns(self) -> int:
        """Maximum number of turns that can be stored."""
        return self._max_turns
