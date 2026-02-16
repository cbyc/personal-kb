"""CLI entry point for the Personal Knowledge Base."""

import argparse

from src.config import get_settings
from src.memory import ConversationMemory
from src.models import QueryResult
from src.pipeline import build_pipeline
from src.tracing import setup_tracing


def format_sources(result: QueryResult) -> str:
    """Format deduplicated source citations from a QueryResult.

    Returns a string like:
      Sources:
        - data/notes/project_alpha.txt
        - https://example.com/page
    or empty string if no sources.
    """
    if not result.sources:
        return ""
    seen = list(dict.fromkeys(result.sources))
    lines = [f"  - {ref}" for ref in seen]
    return "Sources:\n" + "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Personal KB - Second Brain")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Clear all indexed data and reindex notes and bookmarks from scratch.",
    )
    parser.add_argument("question", nargs="*", help="Question to ask (interactive mode if omitted)")
    return parser.parse_args()


def main():
    """Run the personal KB CLI."""
    args = parse_args()
    settings = get_settings()
    setup_tracing()

    print("Personal KB - Second Brain")
    if args.reindex:
        print("Reindexing all data from scratch...")
    print("Loading knowledge base...")

    agent = build_pipeline(settings, reindex=args.reindex)

    print("Ready for questions!")

    # Single query mode via CLI argument
    if args.question:
        question = " ".join(args.question)
        result = agent.ask(question)
        print(f"\n{result.answer}")
        sources = format_sources(result)
        if sources:
            print(f"\n{sources}")
        return

    # Interactive mode with conversation memory
    memory = ConversationMemory(max_turns=settings.conversation_history_length)
    print("Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        result = agent.ask(question, message_history=memory.get_history())
        print(f"\nKB: {result.answer}")
        sources = format_sources(result)
        if sources:
            print(f"\n{sources}")
        print()
        memory.add_turn(question, result.answer)


if __name__ == "__main__":
    main()
