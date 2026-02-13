"""Tests for embedding generation."""

from src.embeddings import EmbeddingModel


class TestEmbeddingModel:
    """Tests for the EmbeddingModel class."""

    def test_init_creates_model(self):
        """EmbeddingModel should initialize without error."""
        model = EmbeddingModel()
        assert model is not None

    def test_embed_text_returns_list_of_floats(self):
        """embed_text should return a list of floats."""
        model = EmbeddingModel()
        embedding = model.embed_text("Hello world")
        assert isinstance(embedding, list)
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_text_correct_dimension(self):
        """embed_text should return a vector of 384 dimensions."""
        model = EmbeddingModel()
        embedding = model.embed_text("Hello world")
        assert len(embedding) == 384

    def test_embed_texts_batch(self):
        """embed_texts should handle multiple texts."""
        model = EmbeddingModel()
        texts = ["Hello", "World", "Test"]
        embeddings = model.embed_texts(texts)
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_embed_texts_empty_list(self):
        """embed_texts with empty list should return empty list."""
        model = EmbeddingModel()
        embeddings = model.embed_texts([])
        assert embeddings == []

    def test_similar_texts_higher_similarity(self):
        """Semantically similar texts should have higher cosine similarity."""
        model = EmbeddingModel()
        e1 = model.embed_text("machine learning algorithms")
        e2 = model.embed_text("deep learning neural networks")
        e3 = model.embed_text("sourdough bread recipe")

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x**2 for x in a) ** 0.5
            norm_b = sum(x**2 for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        sim_related = cosine_sim(e1, e2)
        sim_unrelated = cosine_sim(e1, e3)
        assert sim_related > sim_unrelated

    def test_dimension_property(self):
        """The dimension property should return 384."""
        model = EmbeddingModel()
        assert model.dimension == 384
