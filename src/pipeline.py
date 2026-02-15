"""Pipeline initialization for the personal knowledge base.

Provides a single function to build the full RAG pipeline, avoiding
duplicated setup code across main.py, tests, and evals.
"""

from src.agent import KBAgent, KBDeps
from src.config import Settings, get_settings
from src.document_loader import load_and_chunk
from src.embeddings import EmbeddingModel
from src.vectorstore import VectorStore


def build_pipeline(settings: Settings) -> KBAgent:
    """Build the full RAG pipeline: load, chunk, embed, index, and create agent.

    Args:
        settings: Application settings. Uses default settings if not provided.

    Returns:
        A fully initialized KBAgent ready to answer questions.
        The underlying deps (vectorstore, embedding_model) are accessible
        via agent.deps.
    """

    embedding_model = EmbeddingModel(model_name=settings.embedding_model)
    vectorstore = VectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        use_memory=settings.qdrant_use_memory,
        embedding_dimension=settings.embedding_dimension,
    )

    chunks = load_and_chunk(settings.notes_dir, settings.chunk_size, settings.chunk_overlap)
    embeddings = embedding_model.embed_texts([c.text for c in chunks])
    vectorstore.ensure_collection()
    vectorstore.add_chunks(chunks, embeddings)

    deps = KBDeps(vectorstore=vectorstore, embedding_model=embedding_model)
    return KBAgent(deps)
