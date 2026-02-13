"""Pydantic AI RAG agent for the personal knowledge base."""

from dataclasses import dataclass

from src.config import get_settings
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


class KBAgent:
    """Personal knowledge base agent.

    Creates the underlying pydantic-ai agent once at construction time
    and reuses it across all queries.
    """

    def __init__(self, deps: KBDeps):
        self._agent = self._create_agent()
        self._deps = deps

    @staticmethod
    def _create_agent():
        """Create the pydantic-ai Agent.

        Uses a deferred import to avoid import-time model validation.

        Returns:
            A configured pydantic-ai Agent with the retrieve tool.
        """
        from pydantic_ai import Agent, RunContext

        settings = get_settings()
        agent = Agent(
            settings.llm_model,
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
            deps = ctx.deps
            query_embedding = deps.embedding_model.embed_text(query)
            results = deps.vectorstore.search(query_embedding, top_k=5)

            if not results:
                return "No relevant information found in the knowledge base."

            formatted = []
            for r in results:
                formatted.append(f"[Source: {r.chunk.source}]\n{r.chunk.text}")
            return "\n\n---\n\n".join(formatted)

        return agent

    async def ask_async(self, question: str) -> str:
        """Ask a question (async version).

        Args:
            question: The user's question.

        Returns:
            The agent's answer as a string.
        """
        result = await self._agent.run(question, deps=self._deps)
        return result.output

    def ask(self, question: str) -> str:
        """Ask a question (sync version).

        Args:
            question: The user's question.

        Returns:
            The agent's answer as a string.
        """
        result = self._agent.run_sync(question, deps=self._deps)
        return result.output
