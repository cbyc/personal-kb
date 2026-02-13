"""Qdrant vector store operations."""

from src.models import Chunk, SearchResult


class VectorStore:
    """Manages Qdrant vector store operations."""

    def __init__(
        self,
        collection_name: str = "personal_kb",
        url: str | None = None,
        use_memory: bool = True,
        embedding_dimension: int = 384,
    ):
        """Initialize the vector store client.

        Args:
            collection_name: Name of the Qdrant collection.
            url: Qdrant server URL (ignored if use_memory is True).
            use_memory: Use in-memory storage for development/testing.
            embedding_dimension: Dimension of the embedding vectors.
        """
        raise NotImplementedError("VectorStore.__init__ not yet implemented")

    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        raise NotImplementedError("ensure_collection not yet implemented")

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks with their embeddings to the vector store.

        Args:
            chunks: List of text chunks to store.
            embeddings: Corresponding embedding vectors.
        """
        raise NotImplementedError("add_chunks not yet implemented")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search for similar chunks by embedding.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        raise NotImplementedError("search not yet implemented")

    def delete_collection(self) -> None:
        """Delete the collection (for cleanup in tests)."""
        raise NotImplementedError("delete_collection not yet implemented")
