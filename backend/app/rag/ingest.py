from pathlib import Path

from pypdf import PdfReader


def read_document(path: Path) -> str:
    """Extract text from a TXT, Markdown, or PDF document."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    raise ValueError(f"Unsupported document format: {suffix}")


def chunk_document(text: str, source: str) -> list[dict]:
    """Split a document into section-based chunks."""
    chunks = []
    heading = "General Information"
    lines = []

    def save_chunk():
        content = "\n".join(lines).strip()

        if content:
            chunks.append({
                "source": source,
                "section": heading,
                "text": content,
            })

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            save_chunk()
            heading = stripped.lstrip("#").strip()
            lines = []
        elif stripped:
            lines.append(stripped)

    save_chunk()
    return chunks


def load_knowledge_base(directory: str = "knowledge_base") -> list[dict]:
    """Load and chunk all supported knowledge-base documents."""
    base = Path(directory)
    chunks = []

    for path in sorted(base.iterdir()):
        if path.is_file() and path.suffix.lower() in {
            ".txt", ".md", ".pdf"
        }:
            text = read_document(path)
            chunks.extend(chunk_document(text, path.name))

    return chunks


if __name__ == "__main__":
    documents = load_knowledge_base()

    print(f"Total chunks created: {len(documents)}")

    for index, chunk in enumerate(documents, start=1):
        print(f"\nChunk {index}")
        print(f"Section: {chunk['section']}")
        print(f"Source: {chunk['source']}")
        print(f"Preview: {chunk['text'][:100]}")
