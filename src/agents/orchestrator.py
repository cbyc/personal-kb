"""Orchestrator Agent — coordinates retrieval and research agents."""

from src.agents.research import ResearchAgent
from src.agents.retrieval import RetrievalAgent, RetrievalDeps
from src.config import get_settings
from src.embeddings import EmbeddingModel
from src.models import QueryResult
from src.vectorstore import VectorStore


class OrchestratorAgent:
    """Orchestrates the multi-agent pipeline.

    Routes user queries through the retrieval and research agents:
    1. Validates the query
    2. Calls RetrievalAgent to search the knowledge base
    3. Calls ResearchAgent to synthesize an answer from retrieved chunks
    4. Returns the final QueryResult
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

    @property
    def vectorstore(self) -> VectorStore:
        """Access the vector store."""
        return self._vectorstore

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Access the embedding model."""
        return self._embedding_model

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

    def ask(self, question: str) -> QueryResult:
        """Process a question through the multi-agent pipeline (sync).

        Args:
            question: The user's question.

        Returns:
            A QueryResult with the answer and source documents.

        Raises:
            ValueError: If the query exceeds the maximum allowed length.
        """
        self._validate_query(question)

        # Step 1: Retrieve relevant chunks
        search_results = self._retrieval_agent.search(question)
        context = self._retrieval_agent.format_results(search_results)

        # Step 2: Synthesize answer
        kb_response = self._research_agent.synthesize(question, context)

        return QueryResult(answer=kb_response.answer, sources=search_results)

    async def ask_async(self, question: str) -> QueryResult:
        """Process a question through the multi-agent pipeline (async).

        Args:
            question: The user's question.

        Returns:
            A QueryResult with the answer and source documents.

        Raises:
            ValueError: If the query exceeds the maximum allowed length.
        """
        self._validate_query(question)

        # Step 1: Retrieve relevant chunks
        search_results = self._retrieval_agent.search(question)
        context = self._retrieval_agent.format_results(search_results)

        # Step 2: Synthesize answer
        kb_response = await self._research_agent.synthesize_async(question, context)

        return QueryResult(answer=kb_response.answer, sources=search_results)
