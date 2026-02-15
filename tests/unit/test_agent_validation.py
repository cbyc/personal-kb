"""Tests for agent input validation."""

import pytest

from src.agents.orchestrator import OrchestratorAgent
from src.embeddings import EmbeddingModel
from src.vectorstore import VectorStore


@pytest.fixture
def orchestrator(test_settings):
    """An OrchestratorAgent with in-memory vectorstore for validation tests."""
    embedding_model = EmbeddingModel(model_name=test_settings.embedding_model)
    vectorstore = VectorStore(
        use_memory=True,
        embedding_dimension=test_settings.embedding_dimension,
    )
    vectorstore.ensure_collection()
    return OrchestratorAgent(
        vectorstore=vectorstore,
        embedding_model=embedding_model,
    )


class TestQueryValidation:
    """Tests for query length validation."""

    def test_short_query_accepted(self, orchestrator: OrchestratorAgent):
        """A query within the length limit should not raise."""
        # Just validate — don't actually call the LLM.
        orchestrator._validate_query("What is Project Alpha?")

    def test_query_at_limit_accepted(self, orchestrator: OrchestratorAgent):
        """A query exactly at the max length should not raise."""
        query = "a" * orchestrator._settings.max_query_length
        orchestrator._validate_query(query)

    def test_query_over_limit_raises(self, orchestrator: OrchestratorAgent):
        """A query exceeding the max length should raise ValueError."""
        query = "a" * (orchestrator._settings.max_query_length + 1)
        with pytest.raises(ValueError, match="Query too long"):
            orchestrator._validate_query(query)

    def test_ask_rejects_long_query(self, orchestrator: OrchestratorAgent):
        """ask() should raise ValueError before calling the LLM."""
        query = "a" * (orchestrator._settings.max_query_length + 1)
        with pytest.raises(ValueError, match="Query too long"):
            orchestrator.ask(query)

    @pytest.mark.asyncio
    async def test_ask_async_rejects_long_query(self, orchestrator: OrchestratorAgent):
        """ask_async() should raise ValueError before calling the LLM."""
        query = "a" * (orchestrator._settings.max_query_length + 1)
        with pytest.raises(ValueError, match="Query too long"):
            await orchestrator.ask_async(query)
