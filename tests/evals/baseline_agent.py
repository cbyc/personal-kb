"""Naive baseline agent — answers questions using only the LLM, no retrieval.

Used as a comparison point to measure the improvement of the RAG system.
"""

from src.config import get_settings


class BaselineAgent:
    """A naive agent that answers questions using only the LLM's general knowledge.

    No retrieval, no knowledge base, no tools — just the raw LLM.
    This serves as the baseline for measuring RAG improvement.
    """

    def __init__(self):
        self._agent = self._create_agent()

    @staticmethod
    def _create_agent():
        """Create a minimal pydantic-ai Agent with no tools."""
        from pydantic_ai import Agent

        settings = get_settings()
        return Agent(
            settings.llm_model,
            system_prompt=(
                "You are a helpful assistant. Answer questions to the best of your ability. "
                "If you don't know the answer, say so."
            ),
        )

    def ask(self, question: str) -> str:
        """Ask a question using only the LLM's general knowledge.

        Args:
            question: The user's question.

        Returns:
            The LLM's answer as a string.
        """
        result = self._agent.run_sync(question)
        return result.output

    async def ask_async(self, question: str) -> str:
        """Ask a question using only the LLM's general knowledge (async).

        Args:
            question: The user's question.

        Returns:
            The LLM's answer as a string.
        """
        result = await self._agent.run(question)
        return result.output
