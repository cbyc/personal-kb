"""File loading and text chunking."""

from pathlib import Path

from src.models import Chunk, Document


def load_documents(directory: str | Path) -> list[Document]:
    """Load all .txt files from a directory.

    Args:
        directory: Path to the directory containing .txt files.

    Returns:
        List of Document objects with content and source metadata.
    """
    raise NotImplementedError("load_documents not yet implemented")


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split a document into overlapping text chunks.

    Args:
        document: The document to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of Chunk objects with text, source, and index.
    """
    raise NotImplementedError("chunk_document not yet implemented")


def load_and_chunk(
    directory: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Load documents from a directory and chunk them in one step.

    Args:
        directory: Path to the directory containing .txt files.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of all Chunk objects from all documents.
    """
    raise NotImplementedError("load_and_chunk not yet implemented")
