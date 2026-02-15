"""Evaluation: Baseline Comparison

Compares the RAG system against a naive baseline (LLM-only, no retrieval).
Demonstrates measurable improvement from the RAG architecture.
Uses LLMJudge to score both systems on the same questions.
"""

import json
from pathlib import Path

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from src.config import get_settings
from src.pipeline import build_pipeline

from tests.evals.baseline_agent import BaselineAgent


# Questions with expected answers that require knowledge base data
COMPARISON_QUESTIONS = [
    {
        "name": "project_deadline",
        "question": "What is Project Alpha's deadline?",
        "rubric": (
            "The answer correctly states Project Alpha's MVP deadline "
            "as March 30, 2024, based on specific project documentation."
        ),
    },
    {
        "name": "database_migration",
        "question": "What was decided about the database migration?",
        "rubric": (
            "The answer mentions that the team decided to use Alembic "
            "for database migrations, citing meeting notes."
        ),
    },
    {
        "name": "sourdough_proofing",
        "question": "How long should I proof sourdough?",
        "rubric": (
            "The answer mentions cold proofing for 12-16 hours "
            "and/or bulk fermentation for 4-6 hours."
        ),
    },
    {
        "name": "meeting_attendees",
        "question": "Who attended the January meeting?",
        "rubric": (
            "The answer lists specific attendees including "
            "Sarah Chen, Marcus Johnson, Priya Patel."
        ),
    },
]


def _build_rag_dataset() -> tuple[Dataset, callable]:
    """Build dataset and task for the RAG system."""
    settings = get_settings()
    agent = build_pipeline(settings)
    judge_model = settings.judge_model

    async def rag_task(question: str) -> str:
        result = await agent.ask_async(question)
        return result.answer

    cases = [
        Case(
            name=q["name"],
            inputs=q["question"],
            evaluators=[
                LLMJudge(
                    rubric=q["rubric"],
                    model=judge_model,
                    include_input=True,
                ),
            ],
        )
        for q in COMPARISON_QUESTIONS
    ]

    return Dataset(name="rag_system", cases=cases), rag_task


def _build_baseline_dataset() -> tuple[Dataset, callable]:
    """Build dataset and task for the naive baseline."""
    baseline = BaselineAgent()
    judge_model = get_settings().judge_model

    async def baseline_task(question: str) -> str:
        return baseline.ask(question)

    cases = [
        Case(
            name=q["name"],
            inputs=q["question"],
            evaluators=[
                LLMJudge(
                    rubric=q["rubric"],
                    model=judge_model,
                    include_input=True,
                ),
            ],
        )
        for q in COMPARISON_QUESTIONS
    ]

    return Dataset(name="baseline", cases=cases), baseline_task


def test_baseline_comparison():
    """Run baseline comparison evaluation and save results."""
    # Run RAG evaluation
    rag_dataset, rag_task = _build_rag_dataset()
    rag_report = rag_dataset.evaluate_sync(rag_task)
    rag_report.print()

    # Run baseline evaluation
    baseline_dataset, baseline_task = _build_baseline_dataset()
    baseline_report = baseline_dataset.evaluate_sync(baseline_task)
    baseline_report.print()

    # Count passes for each
    rag_passes = sum(
        1
        for case in rag_report.cases
        if case.assertions and all(r.value for r in case.assertions.values())
    )
    baseline_passes = sum(
        1
        for case in baseline_report.cases
        if case.assertions and all(r.value for r in case.assertions.values())
    )

    total = len(COMPARISON_QUESTIONS)
    rag_rate = rag_passes / total if total > 0 else 0
    baseline_rate = baseline_passes / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"RAG System:  {rag_passes}/{total} ({rag_rate:.0%}) passed")
    print(f"Baseline:    {baseline_passes}/{total} ({baseline_rate:.0%}) passed")
    print(f"Improvement: {rag_rate - baseline_rate:+.0%}")
    print(f"{'='*60}\n")

    # Save results
    results_dir = Path("eval_results")
    results_dir.mkdir(exist_ok=True)
    results = {
        "rag_pass_rate": rag_rate,
        "baseline_pass_rate": baseline_rate,
        "improvement": rag_rate - baseline_rate,
        "rag_passes": rag_passes,
        "baseline_passes": baseline_passes,
        "total_questions": total,
    }
    results_path = results_dir / "baseline_comparison.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {results_path}")

    # RAG should outperform baseline
    assert rag_passes >= baseline_passes, (
        f"RAG system ({rag_passes}/{total}) should outperform "
        f"baseline ({baseline_passes}/{total})"
    )
