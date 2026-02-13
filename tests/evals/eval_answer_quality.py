"""Evaluation: Answer Quality

Tests the full RAG pipeline end-to-end: question -> retrieval -> LLM answer.
Uses LLMJudge to assess answer quality, accuracy, and source citation.
Tests that the agent correctly declines to answer out-of-scope questions.
"""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from src.agent import KBAgent, KBDeps
from src.config import get_settings
from src.document_loader import load_and_chunk
from src.embeddings import EmbeddingModel
from src.vectorstore import VectorStore

JUDGE_MODEL = "google-gla:gemini-2.0-flash"


def _build_ask_task():
    """Build full RAG pipeline once, return an async task function that answers questions."""
    settings = get_settings()
    chunks = load_and_chunk(settings.notes_dir, settings.chunk_size, settings.chunk_overlap)
    model = EmbeddingModel(model_name=settings.embedding_model)
    store = VectorStore(use_memory=True, embedding_dimension=settings.embedding_dimension)
    store.ensure_collection()
    embeddings = model.embed_texts([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)
    deps = KBDeps(vectorstore=store, embedding_model=model)
    agent = KBAgent(deps)

    async def ask_question(question: str) -> str:
        return await agent.ask_async(question)

    return ask_question


answer_quality_dataset = Dataset(
    name="answer_quality",
    cases=[
        Case(
            name="project_deadline_answer",
            inputs="What is Project Alpha's deadline?",
            expected_output="March 30, 2024",
            evaluators=[
                LLMJudge(
                    rubric=(
                        "The answer correctly states Project Alpha's MVP deadline "
                        "as March 30, 2024 and is based on the retrieved context."
                    ),
                    model=JUDGE_MODEL,
                    include_input=True,
                    include_expected_output=True,
                ),
            ],
        ),
        Case(
            name="database_migration_answer",
            inputs="What was decided about the database migration?",
            evaluators=[
                LLMJudge(
                    rubric=(
                        "The answer mentions that the team decided to use Alembic "
                        "for database migrations."
                    ),
                    model=JUDGE_MODEL,
                    include_input=True,
                ),
            ],
        ),
        Case(
            name="sourdough_proofing_answer",
            inputs="How long should I proof sourdough?",
            evaluators=[
                LLMJudge(
                    rubric=(
                        "The answer mentions cold proofing overnight for 12-16 hours "
                        "and/or bulk fermentation for 4-6 hours."
                    ),
                    model=JUDGE_MODEL,
                    include_input=True,
                ),
            ],
        ),
        Case(
            name="gradient_descent_answer",
            inputs="What is gradient descent?",
            evaluators=[
                LLMJudge(
                    rubric=(
                        "The answer explains gradient descent as an optimization algorithm "
                        "that minimizes a loss function by iteratively adjusting parameters."
                    ),
                    model=JUDGE_MODEL,
                    include_input=True,
                ),
            ],
        ),
        Case(
            name="out_of_scope_question",
            inputs="What is the capital of France?",
            evaluators=[
                LLMJudge(
                    rubric=(
                        "The answer correctly indicates that the information is not "
                        "available in the knowledge base. It should NOT answer 'Paris' "
                        "from general knowledge."
                    ),
                    model=JUDGE_MODEL,
                    include_input=True,
                ),
            ],
        ),
        Case(
            name="meeting_attendees_answer",
            inputs="Who attended the January meeting?",
            evaluators=[
                LLMJudge(
                    rubric=(
                        "The answer lists the meeting attendees, including at least "
                        "Sarah Chen, Marcus Johnson, Priya Patel, Alex Rivera, "
                        "and David Kim."
                    ),
                    model=JUDGE_MODEL,
                    include_input=True,
                ),
            ],
        ),
    ],
)


def test_answer_quality():
    """Run answer quality evaluation."""
    ask_question = _build_ask_task()
    report = answer_quality_dataset.evaluate_sync(ask_question)
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
