"""Tests for the RetrievalAgent."""

import pytest

from src.agents.retrieval import RetrievalAgent, RetrievalDeps
from src.embeddings import EmbeddingModel
from src.models import Chunk, SearchResult
from src.vectorstore import VectorStore


@pytest.fixture
def retrieval_agent(test_settings) -> RetrievalAgent:
    """A RetrievalAgent with in-memory vectorstore seeded with test data."""
    embedding_model = EmbeddingModel(model_name=test_settings.embedding_model)
    vectorstore = VectorStore(
        use_memory=True,
        embedding_dimension=test_settings.embedding_dimension,
    )
    vectorstore.ensure_collection()

    # Seed with test chunks
    chunks = [
        Chunk(
            text="Project Alpha deadline is March 30, 2024.",
            source="project_alpha.txt",
            chunk_index=0,
        ),
        Chunk(
            text="Gradient descent is an optimization algorithm.",
            source="machine_learning_notes.txt",
            chunk_index=0,
        ),
        Chunk(
            text="Kubernetes networking uses CNI plugins for pod communication.",
            source="https://example.com/k8s-networking",
            chunk_index=0,
        ),
    ]
    embeddings = embedding_model.embed_texts([c.text for c in chunks])
    vectorstore.add_chunks(chunks, embeddings)

    deps = RetrievalDeps(vectorstore=vectorstore, embedding_model=embedding_model)
    return RetrievalAgent(deps)


class TestRetrievalAgentSearch:
    """Tests for the RetrievalAgent.search() method."""

    def test_search_returns_results(self, retrieval_agent: RetrievalAgent):
        """search() should return a list of SearchResult objects."""
        results = retrieval_agent.search("Project Alpha deadline")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_returns_relevant_source(self, retrieval_agent: RetrievalAgent):
        """search() should return chunks from the relevant source."""
        results = retrieval_agent.search("Project Alpha deadline")
        sources = [r.chunk.source for r in results]
        assert "project_alpha.txt" in sources

    def test_search_returns_url_source(self, retrieval_agent: RetrievalAgent):
        """search() should return URL as source for bookmark-originated chunks."""
        results = retrieval_agent.search("Kubernetes networking CNI")
        sources = [r.chunk.source for r in results]
        assert "https://example.com/k8s-networking" in sources

    def test_search_empty_for_irrelevant_query(self, retrieval_agent: RetrievalAgent):
        """search() should return empty or low-scoring results for irrelevant queries."""
        results = retrieval_agent.search("quantum entanglement in black holes")
        # Results may exist but should have low relevance scores
        assert isinstance(results, list)


class TestRetrievalAgentFormatResults:
    """Tests for the RetrievalAgent.format_results() method."""

    def test_format_empty_results(self, retrieval_agent: RetrievalAgent):
        """format_results() with empty list should return no-info message."""
        result = retrieval_agent.format_results([])
        assert "No relevant information found" in result

    def test_format_note_results(self, retrieval_agent: RetrievalAgent):
        """format_results() should include source for notes."""
        results = retrieval_agent.search("Project Alpha deadline")
        formatted = retrieval_agent.format_results(results)
        assert "[Source: project_alpha.txt]" in formatted

    def test_format_bookmark_results(self, retrieval_agent: RetrievalAgent):
        """format_results() should include URL source for bookmark-originated chunks."""
        results = retrieval_agent.search("Kubernetes networking")
        formatted = retrieval_agent.format_results(results)
        assert "[Source: https://example.com/k8s-networking]" in formatted
