"""Tests for agent input validation."""

import pytest

from src.agent import KBAgent, KBDeps
from src.embeddings import EmbeddingModel
from src.vectorstore import VectorStore


@pytest.fixture
def kb_agent(test_settings):
    """A KBAgent with in-memory vectorstore for validation tests."""
    embedding_model = EmbeddingModel(model_name=test_settings.embedding_model)
    vectorstore = VectorStore(
        use_memory=True,
        embedding_dimension=test_settings.embedding_dimension,
    )
    vectorstore.ensure_collection()
    deps = KBDeps(vectorstore=vectorstore, embedding_model=embedding_model)
    return KBAgent(deps)


class TestQueryValidation:
    """Tests for query length validation."""

    def test_short_query_accepted(self, kb_agent: KBAgent):
        """A query within the length limit should not raise."""
        # Just validate — don't actually call the LLM.
        kb_agent._validate_query("What is Project Alpha?")

    def test_query_at_limit_accepted(self, kb_agent: KBAgent):
        """A query exactly at the max length should not raise."""
        query = "a" * kb_agent._max_query_length
        kb_agent._validate_query(query)

    def test_query_over_limit_raises(self, kb_agent: KBAgent):
        """A query exceeding the max length should raise ValueError."""
        query = "a" * (kb_agent._max_query_length + 1)
        with pytest.raises(ValueError, match="Query too long"):
            kb_agent._validate_query(query)

    def test_ask_rejects_long_query(self, kb_agent: KBAgent):
        """ask() should raise ValueError before calling the LLM."""
        query = "a" * (kb_agent._max_query_length + 1)
        with pytest.raises(ValueError, match="Query too long"):
            kb_agent.ask(query)

    @pytest.mark.asyncio
    async def test_ask_async_rejects_long_query(self, kb_agent: KBAgent):
        """ask_async() should raise ValueError before calling the LLM."""
        query = "a" * (kb_agent._max_query_length + 1)
        with pytest.raises(ValueError, match="Query too long"):
            await kb_agent.ask_async(query)
