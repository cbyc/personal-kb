# Personal KB — AI-Powered Knowledge Base

A multi-agent RAG (Retrieval-Augmented Generation) system that turns your personal notes and Firefox bookmarks into a searchable, conversational knowledge base. Ask questions in natural language and get accurate, cited answers from your own documents.

## Features

- **Multi-agent architecture** — Orchestrator, Retrieval, Research, and Guard agents collaborate to answer queries
- **Firefox bookmark sync** — Automatically imports and indexes bookmarked web pages with incremental sync
- **Conversation memory** — Session-scoped context allows multi-turn follow-up questions
- **Input & output guardrails** — Detects prompt injection, off-topic queries, and validates response grounding
- **Structured citations** — Every answer includes source references with document titles and URLs
- **Evaluation-driven development** — 6 eval suites with 30+ cases covering answer quality, retrieval, guardrails, memory, bookmarks, and baseline comparison

## Architecture

```
User Query
    |
    v
+---------------------+
|  Orchestrator Agent  |
|  (coordination)      |
+-----+-------+-------+
      |       |
      v       |
+-----------+ |
|   Guard   | |  <-- Input validation (prompt injection, off-topic)
|   Agent   | |
+-----+-----+ |
      |       |
      v       |
+-----------+ |
| Retrieval | |  <-- Vector search over Qdrant
|   Agent   | |
+-----+-----+ |
      |       |
      v       |
+-----------+ |
| Research  | |  <-- Answer synthesis with citations
|   Agent   | |
+-----+-----+ |
      |       |
      v       |
+-----------+ |
|   Guard   | |  <-- Output validation (hallucination check)
|   Agent   | |
+-----+-----+ |
      |       |
      v       |
  Response
  with
  citations
```

**Data flow:**
1. **Guard Agent** checks the query for prompt injection and off-topic content
2. **Retrieval Agent** embeds the query and searches the Qdrant vector store
3. **Research Agent** synthesizes an answer from retrieved chunks with citations
4. **Guard Agent** validates the response is grounded in the retrieved context

**Data sources:**
- **Notes** — `.txt` files loaded from `data/notes/`
- **Bookmarks** — Firefox bookmarks read from `places.sqlite`, web content extracted via `trafilatura`

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- A Google API key (for Gemini LLM)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd personal-kb

# Install dependencies
uv sync --extra dev

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Configuration

Create a `.env` file with the following settings:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Google API key for Gemini LLM |
| `LLM_MODEL` | `google-gla:gemini-2.0-flash` | LLM model identifier |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence transformer model |
| `QDRANT_USE_MEMORY` | `true` | Use in-memory Qdrant (no server needed) |
| `NOTES_DIR` | `data/notes` | Directory containing `.txt` note files |
| `BOOKMARK_SYNC_ENABLED` | `true` | Enable Firefox bookmark sync |
| `FIREFOX_PROFILE_PATH` | `auto` | Firefox profile path (`auto` to detect) |
| `GUARDRAILS_ENABLED` | `true` | Enable input/output guardrails |
| `CONVERSATION_HISTORY_LENGTH` | `10` | Max conversation turns to remember |
| `CHUNK_SIZE` | `500` | Max characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Character overlap between chunks |

## Usage

### Single Query

```bash
uv run main.py "What is Project Alpha's deadline?"
```

### Interactive Mode

```bash
uv run main.py
```

Example session:

```
Personal KB - Second Brain
Loading knowledge base...
Ready for questions!
Type 'quit' to exit.

You: What is Project Alpha's deadline?

KB: The MVP deadline for Project Alpha is March 30, 2024.

You: Who is working on it?

KB: Based on the meeting notes, the team includes Sarah Chen, Marcus Johnson,
Priya Patel, Alex Rivera, and David Kim.

You: What did they decide about database migrations?

KB: The team decided to use Alembic for database migrations, as discussed in
the January meeting.

You: quit
```

### Adding Notes

Place `.txt` files in `data/notes/`. They are automatically loaded and indexed on startup.

### Bookmark Sync

When `BOOKMARK_SYNC_ENABLED=true`, the system:
1. Reads your Firefox bookmarks from `places.sqlite`
2. Fetches and extracts text content from bookmarked pages
3. Indexes the content alongside your notes
4. Tracks sync state for incremental updates (only new bookmarks are processed on subsequent runs)

Set `FIREFOX_PROFILE_PATH` to your profile path, or leave as `auto` for automatic detection.

## Evaluation

The project uses evaluation-driven development with `pydantic-evals`. Six eval suites cover different aspects:

### Running Evals

```bash
# All evals
uv run pytest tests/evals/

# Individual suites
uv run pytest tests/evals/eval_answer_quality.py      # End-to-end answer quality (7 cases)
uv run pytest tests/evals/eval_retrieval_accuracy.py   # Retrieval precision (3 cases)
uv run pytest tests/evals/eval_semantic_relevance.py   # Semantic matching (6 cases)
uv run pytest tests/evals/eval_guardrails.py           # Input/output guardrails (10 cases)
uv run pytest tests/evals/eval_memory.py               # Conversation memory (6 cases)
uv run pytest tests/evals/eval_bookmarks.py            # Bookmark integration (4 cases)
uv run pytest tests/evals/eval_baseline_comparison.py  # RAG vs naive baseline
```

### Eval Suites

| Suite | Cases | What It Tests |
|---|---|---|
| Answer Quality | 7 | End-to-end RAG answers, out-of-scope rejection, prompt injection |
| Retrieval Accuracy | 3 | Correct documents retrieved for queries |
| Semantic Relevance | 6 | Embedding similarity and ranking |
| Guardrails | 10 | Prompt injection variants, off-topic, output quality |
| Memory | 6 | Follow-up resolution, context carry-over, no cross-session bleed |
| Bookmarks | 4 | Bookmark retrieval, mixed-source, URL citations |
| Baseline Comparison | 2 | RAG system vs naive LLM (no retrieval) |

### Baseline Comparison

The baseline comparison eval demonstrates the value of RAG by comparing:
- **RAG system** (retrieval + synthesis): Uses vector search to find relevant chunks before answering
- **Naive baseline** (LLM only): Answers directly without any knowledge base access

Results are saved to `eval_results/baseline_comparison.json`.

## Unit Tests

```bash
# All unit tests
uv run pytest tests/unit/

# Individual test files
uv run pytest tests/unit/test_retrieval_agent.py    # Retrieval agent (8 tests)
uv run pytest tests/unit/test_research_agent.py     # Research agent (5 tests)
uv run pytest tests/unit/test_guard_agent.py        # Guard agent (5 tests)
uv run pytest tests/unit/test_memory.py             # Conversation memory (10 tests)
uv run pytest tests/unit/test_bookmark_loader.py    # Bookmark loader (14 tests)
uv run pytest tests/unit/test_chunking.py           # Text chunking
uv run pytest tests/unit/test_file_loading.py       # File loading
uv run pytest tests/unit/test_embeddings.py         # Embeddings
uv run pytest tests/unit/test_qdrant_ops.py         # Vector store
uv run pytest tests/unit/test_agent_validation.py   # Input validation
```

## Project Structure

```
personal-kb/
├── main.py                          # CLI entrypoint
├── src/
│   ├── agents/
│   │   ├── orchestrator.py          # Multi-agent coordinator
│   │   ├── retrieval.py             # Vector search agent
│   │   ├── research.py              # Answer synthesis agent
│   │   └── guard.py                 # Input/output guardrail agent
│   ├── loaders/
│   │   ├── notes_loader.py          # .txt file loader
│   │   └── bookmark_loader.py       # Firefox bookmark loader
│   ├── config.py                    # Settings (pydantic-settings)
│   ├── document_loader.py           # Text chunking
│   ├── embeddings.py                # Sentence transformer embeddings
│   ├── memory.py                    # Conversation memory
│   ├── models.py                    # Pydantic data models
│   ├── pipeline.py                  # Pipeline builder
│   ├── tracing.py                   # OpenTelemetry tracing
│   └── vectorstore.py              # Qdrant vector store
├── tests/
│   ├── unit/                        # Unit tests (81 tests)
│   └── evals/                       # Evaluation suites (38+ cases)
├── data/
│   └── notes/                       # Knowledge base documents
└── eval_results/                    # Saved evaluation results
```

## Development

```bash
# Run all tests and evals
uv run pytest

# Lint
uv run ruff check

# Format
uv run ruff format
```

## Tech Stack

- **LLM**: Google Gemini 2.0 Flash (via `pydantic-ai`)
- **Agent framework**: `pydantic-ai`
- **Evaluation**: `pydantic-evals` with `LLMJudge`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Vector store**: Qdrant (in-memory mode)
- **Content extraction**: `trafilatura` (for bookmark web pages)
- **Tracing**: OpenTelemetry + Arize Phoenix
