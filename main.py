"""CLI entry point for the Personal Knowledge Base."""

import sys

from src.config import get_settings
from src.memory import ConversationMemory
from src.pipeline import build_pipeline
from src.tracing import setup_tracing


def main():
    """Run the personal KB CLI."""
    settings = get_settings()
    setup_tracing()

    print("Personal KB - Second Brain")
    print("Loading knowledge base...")

    agent = build_pipeline(settings)

    print("Ready for questions!")

    # Single query mode via CLI argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = agent.ask(question)
        print(f"\n{result.answer}")
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
        print(f"\nKB: {result.answer}\n")
        memory.add_turn(question, result.answer)


if __name__ == "__main__":
    main()
