"""Evaluation: Semantic Relevance

Tests whether retrieved chunks are semantically relevant to the query.
Uses LLMJudge to assess whether the retrieved content actually addresses the question.
"""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from src.config import get_settings
from src.document_loader import load_and_chunk
from src.embeddings import EmbeddingModel
from src.vectorstore import VectorStore

JUDGE_MODEL = "google-gla:gemini-2.0-flash"


def _build_retrieval_task():
    """Build retrieval pipeline once, return a task function that queries it."""
    settings = get_settings()
    chunks = load_and_chunk(settings.notes_dir, settings.chunk_size, settings.chunk_overlap)
    model = EmbeddingModel(model_name=settings.embedding_model)
    store = VectorStore(use_memory=True, embedding_dimension=settings.embedding_dimension)
    store.ensure_collection()
    embeddings = model.embed_texts([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)

    def retrieve_for_query(query: str) -> str:
        query_embedding = model.embed_text(query)
        results = store.search(query_embedding, top_k=3)
        return "\n".join(f"[Source: {r.chunk.source}] {r.chunk.text}" for r in results)

    return retrieve_for_query


semantic_relevance_dataset = Dataset(
    name="semantic_relevance",
    cases=[
        Case(
            name="project_deadline_relevance",
            inputs="What is Project Alpha's deadline?",
            evaluators=[
                LLMJudge(
                    rubric="The retrieved text contains information about Project Alpha deadlines or timeline.",
                    model=JUDGE_MODEL,
                    include_input=True,
                ),
            ],
        ),
        Case(
            name="gradient_descent_relevance",
            inputs="What is gradient descent?",
            evaluators=[
                LLMJudge(
                    rubric="The retrieved text contains an explanation of gradient descent.",
                    model=JUDGE_MODEL,
                    include_input=True,
                ),
            ],
        ),
        Case(
            name="sourdough_relevance",
            inputs="How long should I proof sourdough?",
            evaluators=[
                LLMJudge(
                    rubric="The retrieved text contains information about sourdough proofing times.",
                    model=JUDGE_MODEL,
                    include_input=True,
                ),
            ],
        ),
        Case(
            name="cross_topic_no_bleed",
            inputs="What programming language does Project Alpha use?",
            evaluators=[
                LLMJudge(
                    rubric=(
                        "The retrieved text contains information about Project Alpha's "
                        "technology stack or programming languages, and does NOT contain "
                        "unrelated content like recipes or ML theory."
                    ),
                    model=JUDGE_MODEL,
                    include_input=True,
                ),
            ],
        ),
    ],
)


def test_semantic_relevance():
    """Run semantic relevance evaluation."""
    retrieve_for_query = _build_retrieval_task()
    report = semantic_relevance_dataset.evaluate_sync(retrieve_for_query)
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
