"""Pydantic AI RAG agent for the personal knowledge base."""

from dataclasses import dataclass

from src.embeddings import EmbeddingModel
from src.vectorstore import VectorStore

SYSTEM_PROMPT = (
    "You are a helpful personal knowledge base assistant. "
    "Use the retrieve tool to search for relevant information before answering. "
    "If the retrieved information does not contain the answer, "
    "respond with: I don't have information about that in my knowledge base. "
    "Always cite which source document the information came from. "
    "Do NOT use your general knowledge to answer questions."
)


@dataclass
class KBDeps:
    """Dependencies for the knowledge base agent."""

    vectorstore: VectorStore
    embedding_model: EmbeddingModel


def create_agent():
    """Create the RAG agent. Separated to avoid import-time model validation.

    Returns:
        A configured pydantic-ai Agent with the retrieve tool.
    """
    from pydantic_ai import Agent, RunContext

    agent = Agent(
        "google-gla:gemini-2.0-flash",
        deps_type=KBDeps,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.tool
    def retrieve(ctx: RunContext[KBDeps], query: str) -> str:
        """Search the knowledge base for information relevant to the query.

        Args:
            ctx: The run context with dependencies.
            query: The search query.

        Returns:
            Formatted string of relevant chunks with their sources.
        """
        raise NotImplementedError("retrieve tool not yet implemented")

    return agent


def ask(question: str, deps: KBDeps) -> str:
    """Ask a question to the knowledge base agent.

    Args:
        question: The user's question.
        deps: The agent dependencies (vectorstore + embedding model).

    Returns:
        The agent's answer as a string.
    """
    raise NotImplementedError("ask not yet implemented")
