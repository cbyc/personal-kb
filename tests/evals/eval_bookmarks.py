"""Evaluation: Bookmark Integration

Tests that the RAG pipeline correctly retrieves and cites bookmark content.
Uses synthetic bookmark documents (pre-fetched content) to avoid live HTTP.
"""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from src.agents.orchestrator import OrchestratorAgent
from src.config import get_settings
from src.document_loader import chunk_document, load_and_chunk
from src.embeddings import EmbeddingModel
from src.models import Document
from src.vectorstore import VectorStore


# Synthetic bookmark documents simulating pre-fetched web content.
_BOOKMARK_DOCUMENTS = [
    Document(
        content=(
            "Rust Programming Language Guide. "
            "Rust is a systems programming language focused on safety, speed, and concurrency. "
            "Key features include ownership and borrowing, zero-cost abstractions, and pattern matching. "
            "The borrow checker enforces memory safety at compile time without garbage collection. "
            "Cargo is the official build tool and package manager for Rust. "
            "Rust was first released in 2015 and has been voted the most loved programming language "
            "in Stack Overflow surveys multiple years in a row."
        ),
        source="https://doc.rust-lang.org/book/",
    ),
    Document(
        content=(
            "Effective Remote Team Management. "
            "Managing remote teams requires intentional communication practices. "
            "Daily standups should be kept to 15 minutes and focus on blockers. "
            "Use asynchronous communication for non-urgent matters to respect time zones. "
            "Tools like Slack, Notion, and Linear help maintain transparency. "
            "Schedule regular one-on-ones (weekly or biweekly) to build trust. "
            "Document decisions in shared wikis to reduce meeting fatigue. "
            "Remote retrospectives should use anonymous feedback tools."
        ),
        source="https://example.com/remote-teams-guide",
    ),
    Document(
        content=(
            "Understanding WebAssembly (Wasm). "
            "WebAssembly is a binary instruction format that runs in modern browsers. "
            "It enables near-native performance for web applications. "
            "Languages like C, C++, Rust, and Go can compile to WebAssembly. "
            "Wasm modules run in a sandboxed environment for security. "
            "Use cases include gaming, image processing, and scientific computing on the web. "
            "WASI (WebAssembly System Interface) extends Wasm beyond the browser."
        ),
        source="https://webassembly.org/docs/",
    ),
]


def _build_pipeline_with_bookmarks():
    """Build a pipeline that includes both notes and bookmark documents."""
    settings = get_settings()
    embedding_model = EmbeddingModel(model_name=settings.embedding_model)
    vectorstore = VectorStore(
        collection_name="eval_bookmarks",
        url=settings.qdrant_url,
        use_memory=True,
        embedding_dimension=settings.embedding_dimension,
    )

    # Load notes
    chunks = load_and_chunk(settings.notes_dir, settings.chunk_size, settings.chunk_overlap)

    # Add bookmark documents
    for doc in _BOOKMARK_DOCUMENTS:
        chunks.extend(chunk_document(doc, settings.chunk_size, settings.chunk_overlap))

    embeddings = embedding_model.embed_texts([c.text for c in chunks])
    vectorstore.ensure_collection()
    vectorstore.add_chunks(chunks, embeddings)

    return OrchestratorAgent(
        vectorstore=vectorstore,
        embedding_model=embedding_model,
    )


def _build_ask_task():
    """Build pipeline once, return an async task function."""
    agent = _build_pipeline_with_bookmarks()

    async def ask_question(question: str) -> str:
        result = await agent.ask_async(question)
        # Include source info in the answer so judges can verify citations
        source_info = ""
        if result.sources:
            source_strs = list(dict.fromkeys(result.sources))
            source_info = "\n\nRetrieved sources: " + "; ".join(source_strs)
        return result.answer + source_info

    return ask_question


def _build_dataset() -> Dataset:
    """Build the bookmark evaluation dataset."""
    judge_model = get_settings().judge_model
    return Dataset(
        name="bookmark_integration",
        cases=[
            Case(
                name="bookmark_only_retrieval",
                inputs="What is the borrow checker in Rust?",
                evaluators=[
                    LLMJudge(
                        rubric=(
                            "The answer explains that Rust's borrow checker enforces memory safety "
                            "at compile time. The answer should be based on retrieved content "
                            "and cite the source URL (https://doc.rust-lang.org/book/)."
                        ),
                        model=judge_model,
                        include_input=True,
                    ),
                ],
            ),
            Case(
                name="bookmark_url_in_citation",
                inputs="How should remote team standups be structured?",
                evaluators=[
                    LLMJudge(
                        rubric=(
                            "The answer mentions that daily standups should be 15 minutes "
                            "and focus on blockers. The sources should cite the URL "
                            "https://example.com/remote-teams-guide."
                        ),
                        model=judge_model,
                        include_input=True,
                    ),
                ],
            ),
            Case(
                name="mixed_source_answer",
                inputs="What programming concepts are covered in my knowledge base?",
                evaluators=[
                    LLMJudge(
                        rubric=(
                            "The answer should reference information from multiple sources, "
                            "potentially including both notes (e.g., machine learning notes) "
                            "and web sources (e.g., Rust, WebAssembly). "
                            "It should cite sources from both file paths and URLs."
                        ),
                        model=judge_model,
                        include_input=True,
                    ),
                ],
            ),
            Case(
                name="bookmark_webassembly_retrieval",
                inputs="What is WASI and how does it extend WebAssembly?",
                evaluators=[
                    LLMJudge(
                        rubric=(
                            "The answer explains that WASI (WebAssembly System Interface) "
                            "extends WebAssembly beyond the browser. The answer should "
                            "cite the source URL (https://webassembly.org/docs/)."
                        ),
                        model=judge_model,
                        include_input=True,
                    ),
                ],
            ),
        ],
    )


def test_bookmark_integration():
    """Run bookmark integration evaluation."""
    ask_question = _build_ask_task()
    dataset = _build_dataset()
    report = dataset.evaluate_sync(ask_question)
    report.print()
    assert not report.failures, (
        f"Eval had task errors: {[f.error_message for f in report.failures]}"
    )
    assert report.cases, "No eval cases completed"
    for case in report.cases:
        assert not case.evaluator_failures, (
            f"Case '{case.name}' had evaluator errors: {case.evaluator_failures}"
        )
        assert case.assertions and all(r.value for r in case.assertions.values()), (
            f"Case '{case.name}' failed assertions: {case.assertions}"
        )
