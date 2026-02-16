"""Evaluation: Retrieval Accuracy

Tests whether the retrieval step returns chunks from the correct source documents.
Uses Contains evaluator to check source file names appear in results.
"""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains

from src.config import get_settings
from src.pipeline import build_pipeline


def _build_retrieval_task():
    """Build retrieval pipeline once, return a task function that queries it."""
    settings = get_settings()
    settings.bookmark_sync_enabled = False
    agent = build_pipeline(settings)

    def retrieve_for_query(query: str) -> str:
        query_embedding = agent.embedding_model.embed_text(query)
        results = agent.vectorstore.search(query_embedding, top_k=3)
        return "\n".join(f"[Source: {r.chunk.source}] {r.chunk.text}" for r in results)

    return retrieve_for_query


retrieval_accuracy_dataset = Dataset(
    name="retrieval_accuracy",
    cases=[
        Case(
            name="project_deadline_query",
            inputs="What is Project Alpha's deadline?",
            expected_output="project_alpha.txt",
            evaluators=[Contains(value="project_alpha.txt")],
        ),
        Case(
            name="database_migration_query",
            inputs="What was decided about the database migration?",
            expected_output="meeting_2024_01.txt",
            evaluators=[Contains(value="meeting_2024_01.txt")],
        ),
        Case(
            name="sourdough_proofing_query",
            inputs="How long should I proof sourdough?",
            expected_output="recipe_sourdough.txt",
            evaluators=[Contains(value="recipe_sourdough.txt")],
        ),
        Case(
            name="gradient_descent_query",
            inputs="What is gradient descent?",
            expected_output="machine_learning_notes.txt",
            evaluators=[Contains(value="machine_learning_notes.txt")],
        ),
        Case(
            name="meeting_attendees_query",
            inputs="Who attended the January meeting?",
            expected_output="meeting_2024_01.txt",
            evaluators=[Contains(value="meeting_2024_01.txt")],
        ),
    ],
)


def test_retrieval_accuracy():
    """Run retrieval accuracy evaluation."""
    retrieve_for_query = _build_retrieval_task()
    report = retrieval_accuracy_dataset.evaluate_sync(retrieve_for_query)
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
