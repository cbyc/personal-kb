"""Embedding generation using sentence-transformers."""


class EmbeddingModel:
    """Wrapper around sentence-transformers for generating embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier.
        """
        raise NotImplementedError("EmbeddingModel.__init__ not yet implemented")

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        raise NotImplementedError("embed_text not yet implemented")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple text strings.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        raise NotImplementedError("embed_texts not yet implemented")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        raise NotImplementedError("dimension not yet implemented")
