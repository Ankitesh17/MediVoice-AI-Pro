import json
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

    print("SUCCESS: FAISS index created!")
    print("Total vectors:", index.ntotal)
    print("Vector dimensions:", dimension)
    print("Index saved to:", INDEX_FILE)


def search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Retrieve the most semantically similar chunks."""
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            "FAISS index not found. Build the index first."
        )

    index = faiss.read_index(str(INDEX_FILE))

    chunks = json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and metadata are inconsistent.")

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
