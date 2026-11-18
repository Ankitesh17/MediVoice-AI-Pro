from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Create normalized embeddings for document chunks."""
    model = get_model()

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Create a normalized embedding for a search query."""
    return embed_texts([query])


if __name__ == "__main__":
    from .ingest import load_knowledge_base

    chunks = load_knowledge_base()

    texts = [
        f"{chunk['section']}\n{chunk['text']}"
        for chunk in chunks
    ]

    vectors = embed_texts(texts)

    print("SUCCESS: Embeddings generated!")
    print("Total chunks:", len(chunks))
    print("Embedding shape:", vectors.shape)
    print("Data type:", vectors.dtype)
