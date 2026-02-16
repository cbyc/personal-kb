"""Notes loader — loads .txt files from a directory as Documents."""

from pathlib import Path

from src.models import Document


def load_documents(directory: str | Path) -> list[Document]:
    """Load all .txt files from a directory.

    Args:
        directory: Path to the directory containing .txt files.

    Returns:
        List of Document objects with content and source metadata.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    documents = []
    for file_path in sorted(dir_path.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                content=content,
                source=str(file_path),
            )
        )
    return documents
