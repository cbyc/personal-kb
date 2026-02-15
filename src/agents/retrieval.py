"""Retrieval Agent — searches the vector store for relevant chunks."""

from dataclasses import dataclass

from src.config import get_settings
from src.embeddings import EmbeddingModel
from src.models import SearchResult
from src.vectorstore import VectorStore


@dataclass
class RetrievalDeps:
    """Dependencies for the retrieval agent."""

    vectorstore: VectorStore
    embedding_model: EmbeddingModel


class RetrievalAgent:
    """Agent that searches the knowledge base and returns relevant chunks.

    Wraps the retrieval tool logic in a dedicated agent with its own
    pydantic-ai Agent instance.
    """

    def __init__(self, deps: RetrievalDeps):
        self._agent = self._create_agent()
        self._deps = deps

    @property
    def deps(self) -> RetrievalDeps:
        """Access the agent's dependencies."""
        return self._deps

    @staticmethod
    def _create_agent():
        """Create the pydantic-ai Agent for retrieval."""
        from pydantic_ai import Agent, RunContext

        settings = get_settings()
        agent = Agent(
            settings.llm_model,
            deps_type=RetrievalDeps,
            system_prompt=(
                "You are a retrieval agent. Your job is to search the knowledge base "
                "using the retrieve tool and return the results. Always call the retrieve "
                "tool with the user's query. Return the formatted results exactly as received."
            ),
        )

        @agent.tool
        def retrieve(ctx: RunContext[RetrievalDeps], query: str) -> str:
            """Search the knowledge base for information relevant to the query.

            Args:
                ctx: The run context with dependencies.
                query: The search query.

            Returns:
                Formatted string of relevant chunks with their sources.
            """
            deps = ctx.deps
            query_embedding = deps.embedding_model.embed_text(query)
            results = deps.vectorstore.search(
                query_embedding,
                top_k=5,
                score_threshold=settings.search_score_threshold,
            )

            if not results:
                return "No relevant information found in the knowledge base."

            formatted = []
            for r in results:
                header = f"[Source: {r.chunk.source} | Type: {r.chunk.source_type}]"
                if r.chunk.url:
                    header += f" [URL: {r.chunk.url}]"
                formatted.append(f"{header}\n{r.chunk.text}")
            return "\n\n---\n\n".join(formatted)

        return agent

    def search(self, query: str) -> list[SearchResult]:
        """Search the knowledge base directly (without LLM).

        This bypasses the LLM and performs a direct vector search,
        used by the orchestrator to get raw results.

        Args:
            query: The search query.

        Returns:
            List of SearchResult objects with relevance scores.
        """
        settings = get_settings()
        query_embedding = self._deps.embedding_model.embed_text(query)
        return self._deps.vectorstore.search(
            query_embedding,
            top_k=5,
            score_threshold=settings.search_score_threshold,
        )

    def format_results(self, results: list[SearchResult]) -> str:
        """Format search results into a string for the research agent.

        Args:
            results: List of SearchResult objects.

        Returns:
            Formatted string of chunks with source metadata.
        """
        if not results:
            return "No relevant information found in the knowledge base."

        formatted = []
        for r in results:
            header = f"[Source: {r.chunk.source} | Type: {r.chunk.source_type}]"
            if r.chunk.url:
                header += f" [URL: {r.chunk.url}]"
            formatted.append(f"{header}\n{r.chunk.text}")
        return "\n\n---\n\n".join(formatted)
