"""CLI entry point for the Personal Knowledge Base."""

import sys

from src.agent import KBAgent, KBDeps
from src.config import get_settings
from src.document_loader import load_and_chunk
from src.embeddings import EmbeddingModel
from src.vectorstore import VectorStore


def main():
    """Run the personal KB CLI."""
    settings = get_settings()

    print("Personal KB - Second Brain")
    print("Loading knowledge base...")

    # Initialize components
    embedding_model = EmbeddingModel(model_name=settings.embedding_model)
    vectorstore = VectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        use_memory=settings.qdrant_use_memory,
        embedding_dimension=settings.embedding_dimension,
    )

    # Load and index documents
    chunks = load_and_chunk(settings.notes_dir, settings.chunk_size, settings.chunk_overlap)
    embeddings = embedding_model.embed_texts([c.text for c in chunks])
    vectorstore.ensure_collection()
    vectorstore.add_chunks(chunks, embeddings)

    deps = KBDeps(vectorstore=vectorstore, embedding_model=embedding_model)
    agent = KBAgent(deps)

    print(f"Loaded {len(chunks)} chunks. Ready for questions!")

    # Single query mode via CLI argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = agent.ask(question)
        print(f"\n{answer}")
        return

    # Interactive mode
    print("Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer = agent.ask(question)
        print(f"\nKB: {answer}\n")


if __name__ == "__main__":
    main()
