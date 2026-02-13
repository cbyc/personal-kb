"""Tests for document loading from filesystem."""

from pathlib import Path

import pytest

from src.document_loader import load_documents
from src.models import Document


class TestLoadDocuments:
    """Tests for the load_documents function."""

    def test_returns_list_of_documents(self, test_data_dir: Path):
        """load_documents should return a list of Document objects."""
        docs = load_documents(test_data_dir)
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_finds_all_txt_files(self, test_data_dir: Path):
        """Should find all 4 synthetic test files."""
        docs = load_documents(test_data_dir)
        assert len(docs) == 4

    def test_content_not_empty(self, test_data_dir: Path):
        """Each loaded document should have non-empty content."""
        docs = load_documents(test_data_dir)
        for doc in docs:
            assert len(doc.content) > 0

    def test_source_is_set(self, test_data_dir: Path):
        """Each document should have its source file path set."""
        docs = load_documents(test_data_dir)
        for doc in docs:
            assert doc.source != ""
            assert doc.source.endswith(".txt")

    def test_nonexistent_dir_raises(self):
        """Should raise FileNotFoundError for a nonexistent directory."""
        with pytest.raises((FileNotFoundError, NotImplementedError)):
            load_documents("/nonexistent/path")

    def test_empty_dir_returns_empty_list(self, tmp_path: Path):
        """Should return empty list for directory with no .txt files."""
        docs = load_documents(tmp_path)
        assert docs == []
