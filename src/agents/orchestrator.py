"""Orchestrator Agent — coordinates retrieval, research, and guard agents."""

import logging
from collections.abc import Sequence

from pydantic_ai.messages import ModelMessage

from src.agents.guard import GuardAgent
from src.agents.research import ResearchAgent
from src.agents.retrieval import RetrievalAgent, RetrievalDeps
from src.config import get_settings
from src.embeddings import EmbeddingModel
from src.models import KBResponse, QueryResult, SearchResult
from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """Orchestrates the multi-agent pipeline.

    Routes user queries through guard, retrieval, and research agents:
    1. Guard Agent validates input (if guardrails enabled)
    2. RetrievalAgent searches the knowledge base
    3. ResearchAgent synthesizes an answer from retrieved chunks
    4. Guard Agent validates output (if guardrails enabled)
    5. Returns the final QueryResult
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        embedding_model: EmbeddingModel,
    ):
        self._settings = get_settings()
        self._vectorstore = vectorstore
        self._embedding_model = embedding_model

        retrieval_deps = RetrievalDeps(
            vectorstore=vectorstore,
            embedding_model=embedding_model,
        )
        self._retrieval_agent = RetrievalAgent(retrieval_deps)
        self._research_agent = ResearchAgent()
        self._guard_agent = GuardAgent() if self._settings.guardrails_enabled else None

    @property
    def vectorstore(self) -> VectorStore:
        """Access the vector store."""
        return self._vectorstore

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Access the embedding model."""
        return self._embedding_model

    @staticmethod
    def _filter_cited_sources(
        kb_response: KBResponse,
        search_results: list[SearchResult],
    ) -> list[SearchResult]:
        """Keep only the search results that the research agent actually cited.

        Uses substring matching on titles because the LLM may return a partial
        or slightly different form of the source name (e.g. 'project_alpha.txt'
        vs 'data/notes/project_alpha.txt').
        """
        if not kb_response.sources:
            return []
        cited_titles = [ref.title for ref in kb_response.sources]
        cited_urls = {ref.url for ref in kb_response.sources if ref.url}

        def _is_cited(r: SearchResult) -> bool:
            if r.chunk.url and r.chunk.url in cited_urls:
                return True
            return any(t in r.chunk.source or r.chunk.source in t for t in cited_titles)

        return [r for r in search_results if _is_cited(r)]

    def _validate_query(self, question: str) -> None:
        """Validate the user query before processing.

        Args:
            question: The user's question.

        Raises:
            ValueError: If the query exceeds the maximum allowed length.
        """
        if len(question) > self._settings.max_query_length:
            raise ValueError(
                f"Query too long ({len(question)} chars). "
                f"Maximum allowed: {self._settings.max_query_length} chars."
            )

    def ask(
        self,
        question: str,
        message_history: Sequence[ModelMessage] | None = None,
    ) -> QueryResult:
        """Process a question through the multi-agent pipeline (sync).

        Args:
            question: The user's question.
            message_history: Optional conversation history for follow-up context.

        Returns:
            A QueryResult with the answer and source documents.

        Raises:
            ValueError: If the query exceeds the maximum allowed length.
        """
        self._validate_query(question)

        # Step 1: Guard input validation
        if self._guard_agent:
            verdict = self._guard_agent.validate_input(question)
            if not verdict.allowed:
                return QueryResult(answer=verdict.reason, sources=[])

        # Step 2: Retrieve relevant chunks
        search_results = self._retrieval_agent.search(question)
        context = self._retrieval_agent.format_results(search_results)

        # Step 3: Synthesize answer with conversation history
        try:
            kb_response = self._research_agent.synthesize(
                question, context, message_history=message_history
            )
        except Exception:
            logger.warning("Research agent failed, returning fallback response", exc_info=True)
            return QueryResult(
                answer="I found relevant information but could not synthesize a response. "
                "Please try rephrasing your question.",
                sources=search_results,
            )

        # Step 4: Guard output validation
        if self._guard_agent:
            output_verdict = self._guard_agent.validate_output(
                question, kb_response.answer, context
            )
            if not output_verdict.allowed:
                return QueryResult(
                    answer="I could not verify my response. Please try rephrasing your question.",
                    sources=[],
                )

        cited_sources = self._filter_cited_sources(kb_response, search_results)
        return QueryResult(answer=kb_response.answer, sources=cited_sources)

    async def ask_async(
        self,
        question: str,
        message_history: Sequence[ModelMessage] | None = None,
    ) -> QueryResult:
        """Process a question through the multi-agent pipeline (async).

        Args:
            question: The user's question.
            message_history: Optional conversation history for follow-up context.

        Returns:
            A QueryResult with the answer and source documents.

        Raises:
            ValueError: If the query exceeds the maximum allowed length.
        """
        self._validate_query(question)

        # Step 1: Guard input validation
        if self._guard_agent:
            verdict = await self._guard_agent.validate_input_async(question)
            if not verdict.allowed:
                return QueryResult(answer=verdict.reason, sources=[])

        # Step 2: Retrieve relevant chunks
        search_results = self._retrieval_agent.search(question)
        context = self._retrieval_agent.format_results(search_results)

        # Step 3: Synthesize answer with conversation history
        try:
            kb_response = await self._research_agent.synthesize_async(
                question, context, message_history=message_history
            )
        except Exception:
            logger.warning("Research agent failed, returning fallback response", exc_info=True)
            return QueryResult(
                answer="I found relevant information but could not synthesize a response. "
                "Please try rephrasing your question.",
                sources=search_results,
            )

        # Step 4: Guard output validation
        if self._guard_agent:
            output_verdict = await self._guard_agent.validate_output_async(
                question, kb_response.answer, context
            )
            if not output_verdict.allowed:
                return QueryResult(
                    answer="I could not verify my response. Please try rephrasing your question.",
                    sources=[],
                )

        cited_sources = self._filter_cited_sources(kb_response, search_results)
        return QueryResult(answer=kb_response.answer, sources=cited_sources)
