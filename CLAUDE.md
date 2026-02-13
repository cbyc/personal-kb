# Personal KB

Agentic AI app built with pydantic-ai.

## Development

- Package manager: `uv`
- Python: 3.13+
- Run: `uv run main.py`
- Test: `uv run pytest`
- Unit tests only: `uv run pytest tests/unit/`
- Evals only: `uv run pytest tests/evals/`
- Lint: `uv run ruff check`
- Format: `uv run ruff format`

## Approach

- **Evaluation-driven development**: write evals before implementing agentic features. Evals define expected behavior and drive iteration.
- **Test-driven development**: use TDD approach for deterministic parts of the code.
- Keep agents minimal. Start with the simplest agent that passes evals, then iterate.
- Keep each iteration as minimal as possible and commit after it's done.
- An iteration is done when all the tests and evals pass.

## Structure

- `main.py` — CLI entrypoint
- `src/` — source modules (agent, embeddings, vectorstore, document_loader, config, models)
- `tests/unit/` — unit tests
- `tests/evals/` — pydantic-evals evaluation suites
- `data/notes/` — knowledge base documents (.txt)
