"""Embedding generation using sentence-transformers."""

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around sentence-transformers for generating embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier.
        """
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        embedding = self._model.encode(text)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple text strings.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._model.get_sentence_embedding_dimension()
