"""Shared pytest fixtures for unit tests and evals."""

from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.config import Settings
from src.models import Chunk, Document

# Load .env into OS environment so pydantic-ai providers (Google, OpenAI) can find API keys.
load_dotenv()


@pytest.fixture
def test_data_dir() -> Path:
    """Path to the synthetic test data directory."""
    return Path(__file__).parent.parent / "data" / "notes"


@pytest.fixture
def sample_document() -> Document:
    """A sample Document for testing."""
    return Document(
        content="This is a test document with some content about testing.",
        source="test_file.txt",
        title="Test Document",
    )


@pytest.fixture
def sample_long_document() -> Document:
    """A document long enough to produce multiple chunks."""
    content = "Word " * 200  # ~1000 characters
    return Document(
        content=content.strip(),
        source="long_file.txt",
        title="Long Document",
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """A list of sample chunks for testing."""
    return [
        Chunk(text="First chunk of text about Python.", source="doc1.txt", chunk_index=0),
        Chunk(text="Second chunk about machine learning.", source="doc1.txt", chunk_index=1),
        Chunk(text="Third chunk about cooking recipes.", source="doc2.txt", chunk_index=0),
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Fake embeddings for testing (384-dimensional vectors with distinct elements)."""
    dim = 384
    return [
        [1.0] + [0.0] * (dim - 1),
        [0.0, 1.0] + [0.0] * (dim - 2),
        [0.0, 0.0, 1.0] + [0.0] * (dim - 3),
    ]


@pytest.fixture
def test_settings() -> Settings:
    """Settings configured for testing."""
    return Settings(
        qdrant_use_memory=True,
        notes_dir="data/notes",
        chunk_size=500,
        chunk_overlap=50,
    )
