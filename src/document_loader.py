"""Text chunking and convenience loading functions.

The actual document loading logic lives in src/loaders/ (notes_loader,
bookmark_loader).  This module provides chunking and the load_and_chunk
convenience wrapper.
"""

import re
from pathlib import Path

from src.loaders.notes_loader import load_documents  # re-export for back-compat
from src.models import Chunk, Document

# Sentence-ending patterns: period/question mark/exclamation followed by whitespace, or paragraph break.
_SENTENCE_BOUNDARY = re.compile(r"[.!?]\s|\n\n")

__all__ = ["load_documents", "chunk_document", "load_and_chunk"]


def _find_split_point(text: str, start: int, end: int, chunk_size: int) -> int:
    """Find the best split point, preferring sentence then word boundaries.

    Searches backward from end within the last 20% of the chunk for a sentence
    boundary. If none is found, falls back to the nearest word boundary (space).
    If neither exists, returns the raw character offset.

    Args:
        text: The full document text.
        start: Start index of the current chunk.
        end: Raw end index (start + chunk_size).
        chunk_size: Maximum characters per chunk.

    Returns:
        The adjusted end index for the chunk.
    """
    # If end is past the document, no need to search for a boundary.
    if end >= len(text):
        return len(text)

    # Search window: last 20% of the chunk.
    search_start = end - int(chunk_size * 0.2)
    if search_start < start:
        search_start = start

    window = text[search_start:end]

    # Find the last sentence boundary in the window (closest to end).
    sentence_matches = list(_SENTENCE_BOUNDARY.finditer(window))
    if sentence_matches:
        # Use the last match — split right after the sentence-ending punctuation + space.
        return search_start + sentence_matches[-1].end()

    # Fall back to word boundary: find last space before end.
    last_space = text.rfind(" ", search_start, end)
    if last_space > start:
        return last_space + 1  # Split after the space.

    # No boundary found — split at raw character offset.
    return end


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split a document into overlapping text chunks.

    Prefers splitting at sentence boundaries, falling back to word boundaries,
    then raw character offsets. Each chunk is at most chunk_size characters.

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
        split_at = _find_split_point(text, start, end, chunk_size)
        chunk_text = text[start:split_at]

        chunks.append(
            Chunk(
                text=chunk_text,
                source=document.source,
                chunk_index=chunk_index,
                source_type=document.source_type,
                url=document.url,
            )
        )

        chunk_index += 1
        # If we've reached the end of the document, stop.
        if split_at >= len(text):
            break
        next_start = max(split_at - chunk_overlap, 0)
        # Ensure forward progress to avoid infinite loops.
        if next_start <= start:
            next_start = split_at
        start = next_start

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
