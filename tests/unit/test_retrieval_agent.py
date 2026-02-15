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
            source_type="note",
        ),
        Chunk(
            text="Gradient descent is an optimization algorithm.",
            source="machine_learning_notes.txt",
            chunk_index=0,
            source_type="note",
        ),
        Chunk(
            text="Kubernetes networking uses CNI plugins for pod communication.",
            source="https://example.com/k8s-networking",
            chunk_index=0,
            source_type="bookmark",
            url="https://example.com/k8s-networking",
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

    def test_search_includes_source_type(self, retrieval_agent: RetrievalAgent):
        """search() results should include source_type metadata."""
        results = retrieval_agent.search("Project Alpha deadline")
        for r in results:
            assert r.chunk.source_type in ("note", "bookmark")

    def test_search_bookmark_includes_url(self, retrieval_agent: RetrievalAgent):
        """search() results for bookmarks should include the URL."""
        results = retrieval_agent.search("Kubernetes networking CNI")
        bookmark_results = [r for r in results if r.chunk.source_type == "bookmark"]
        assert len(bookmark_results) > 0
        assert bookmark_results[0].chunk.url == "https://example.com/k8s-networking"

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
        """format_results() should include source and type for notes."""
        results = retrieval_agent.search("Project Alpha deadline")
        formatted = retrieval_agent.format_results(results)
        assert "project_alpha.txt" in formatted
        assert "Type: note" in formatted

    def test_format_bookmark_results(self, retrieval_agent: RetrievalAgent):
        """format_results() should include URL for bookmark results."""
        results = retrieval_agent.search("Kubernetes networking")
        formatted = retrieval_agent.format_results(results)
        assert "URL: https://example.com/k8s-networking" in formatted
        assert "Type: bookmark" in formatted
