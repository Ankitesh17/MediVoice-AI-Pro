import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .embeddings import embed_query, embed_texts
from .ingest import load_knowledge_base

INDEX_DIR = Path("faiss_index")
INDEX_FILE = INDEX_DIR / "clinic.index"
METADATA_FILE = INDEX_DIR / "metadata.json"


def build_index() -> None:
    """Generate and save the FAISS index."""
    chunks = load_knowledge_base()

    if not chunks:
        raise ValueError("Knowledge base is empty.")

    texts = [
        f"{chunk['section']}\n{chunk['text']}"
        for chunk in chunks
    ]

    vectors = embed_texts(texts)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    METADATA_FILE.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _load_index.cache_clear()

    print("SUCCESS: FAISS index created!")
    print("Total vectors:", index.ntotal)
    print("Vector dimensions:", dimension)
    print("Index saved to:", INDEX_FILE)


@lru_cache(maxsize=1)
def _load_index(
    index_file: Path, metadata_file: Path, index_mtime: int, metadata_mtime: int,
) -> tuple[Any, list[dict[str, Any]]]:
    index = faiss.read_index(str(index_file))
    chunks = json.loads(metadata_file.read_text(encoding="utf-8"))
    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and metadata are inconsistent.")
    return index, chunks


def search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Retrieve the most semantically similar chunks."""
    try:
        index, chunks = _load_index(
            INDEX_FILE.resolve(), METADATA_FILE.resolve(),
            INDEX_FILE.stat().st_mtime_ns, METADATA_FILE.stat().st_mtime_ns,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError("FAISS index not found. Build the index first.") from exc

    query_vector = np.asarray(
        embed_query(query), dtype=np.float32
    )

    scores, indices = index.search(
        query_vector,
        min(top_k, index.ntotal),
    )

    results = []

    for score, idx in zip(scores[0], indices[0], strict=True):
        if idx < 0:
            continue

        result = dict(chunks[int(idx)])
        result["score"] = float(score)
        results.append(result)

    return results


if __name__ == "__main__":
    build_index()

    question = "What time does the GP surgery open?"

    print("\nTEST QUESTION:", question)

    for result in search(question):
        print("\nSection:", result["section"])
        print("Similarity:", round(result["score"], 4))
        print("Source:", result["source"])
        print("Text:", result["text"][:250])
