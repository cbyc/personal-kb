"""File loading and text chunking."""

from pathlib import Path

from src.models import Chunk, Document


def load_documents(directory: str | Path) -> list[Document]:
    """Load all .txt files from a directory.

    Args:
        directory: Path to the directory containing .txt files.

    Returns:
        List of Document objects with content and source metadata.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    documents = []
    for file_path in sorted(dir_path.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                content=content,
                source=file_path.name,
                title=file_path.stem,
            )
        )
    return documents


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
    text = document.content
    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append(
            Chunk(
                text=chunk_text,
                source=document.source,
                chunk_index=chunk_index,
            )
        )

        chunk_index += 1
        start += chunk_size - chunk_overlap

    return chunks


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
    documents = load_documents(directory)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return all_chunks
