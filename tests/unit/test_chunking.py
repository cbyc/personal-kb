"""Tests for text chunking logic."""

from src.document_loader import chunk_document
from src.models import Chunk, Document


class TestChunkDocument:
    """Tests for the chunk_document function."""

    def test_returns_list_of_chunks(self, sample_document: Document):
        """chunk_document should return a list of Chunk objects."""
        chunks = chunk_document(sample_document)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_short_document_single_chunk(self, sample_document: Document):
        """A document shorter than chunk_size should produce exactly one chunk."""
        chunks = chunk_document(sample_document, chunk_size=5000)
        assert len(chunks) == 1

    def test_long_document_multiple_chunks(self, sample_long_document: Document):
        """A document longer than chunk_size should produce multiple chunks."""
        chunks = chunk_document(sample_long_document, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunk_text_not_empty(self, sample_long_document: Document):
        """No chunk should have empty text."""
        chunks = chunk_document(sample_long_document, chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_chunk_size_respected(self, sample_long_document: Document):
        """Each chunk's text should not exceed chunk_size characters."""
        chunk_size = 500
        chunks = chunk_document(sample_long_document, chunk_size=chunk_size)
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size

    def test_chunk_overlap_present(self, sample_long_document: Document):
        """Consecutive chunks should have overlapping text."""
        chunks = chunk_document(sample_long_document, chunk_size=500, chunk_overlap=50)
        if len(chunks) >= 2:
            overlap_text = chunks[0].text[-50:]
            assert overlap_text in chunks[1].text

    def test_source_preserved(self, sample_document: Document):
        """Chunks should inherit the source from their document."""
        chunks = chunk_document(sample_document)
        for chunk in chunks:
            assert chunk.source == sample_document.source

    def test_indices_sequential(self, sample_long_document: Document):
        """Chunk indices should be sequential starting from 0."""
        chunks = chunk_document(sample_long_document, chunk_size=200)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_all_content_covered(self, sample_long_document: Document):
        """The union of all chunk texts should cover all original content (no overlap)."""
        chunks = chunk_document(sample_long_document, chunk_size=500, chunk_overlap=0)
        reconstructed = "".join(c.text for c in chunks)
        assert reconstructed == sample_long_document.content
