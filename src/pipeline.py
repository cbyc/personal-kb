"""Pipeline initialization for the personal knowledge base.

Provides a single function to build the full RAG pipeline, avoiding
duplicated setup code across main.py, tests, and evals.
"""

import logging
from pathlib import Path

from src.agents.orchestrator import OrchestratorAgent
from src.config import Settings
from src.document_loader import chunk_document, load_and_chunk
from src.embeddings import EmbeddingModel
from src.loaders.bookmark_loader import load_bookmarks
from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)


def build_pipeline(settings: Settings, *, reindex: bool = False) -> OrchestratorAgent:
    """Build the full RAG pipeline: load, chunk, embed, index, and create orchestrator.

    Loads notes from the notes directory, and optionally syncs Firefox bookmarks
    if bookmark_sync_enabled is True.

    Args:
        settings: Application settings.
        reindex: If True, clear existing data and reindex everything from scratch.

    Returns:
        A fully initialized OrchestratorAgent ready to answer questions.
        The underlying vectorstore and embedding_model are accessible
        via orchestrator.vectorstore and orchestrator.embedding_model.
    """

    embedding_model = EmbeddingModel(model_name=settings.embedding_model)
    vectorstore = VectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        use_memory=settings.qdrant_use_memory,
        embedding_dimension=settings.embedding_dimension,
    )

    if reindex:
        logger.info("Reindex requested — clearing existing data...")
        vectorstore.delete_collection()
        sync_state = Path(settings.bookmark_sync_state_path)
        if sync_state.exists():
            sync_state.unlink()
            logger.info("Removed bookmark sync state: %s", sync_state)

    # Load and chunk notes
    chunks = load_and_chunk(settings.notes_dir, settings.chunk_size, settings.chunk_overlap)

    # Load and chunk bookmarks (if enabled)
    if settings.bookmark_sync_enabled:
        logger.info("Bookmark sync enabled, loading bookmarks...")
        bookmark_docs = load_bookmarks(
            profile_path=settings.firefox_profile_path,
            sync_state_path=settings.bookmark_sync_state_path,
            fetch_timeout=settings.bookmark_fetch_timeout,
            max_content_length=settings.bookmark_max_content_length,
        )
        for doc in bookmark_docs:
            chunks.extend(chunk_document(doc, settings.chunk_size, settings.chunk_overlap))
        logger.info("Total chunks after bookmark sync: %d", len(chunks))

    embeddings = embedding_model.embed_texts([c.text for c in chunks])
    vectorstore.ensure_collection()
    vectorstore.add_chunks(chunks, embeddings)

    return OrchestratorAgent(
        vectorstore=vectorstore,
        embedding_model=embedding_model,
    )
