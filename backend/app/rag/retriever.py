
from .vector_store import search

MIN_SIMILARITY = 0.45


def retrieve_context(question: str, top_k: int = 3) -> dict:
    """Retrieve relevant clinic knowledge for a question."""

    question = question.strip()

    if not question:
        return {
            "found": False,
            "context": "",
            "sources": [],
        }

    results = search(question, top_k=top_k)

    relevant = [
        result
        for result in results
        if result["score"] >= MIN_SIMILARITY
    ]

    if not relevant:
        return {
            "found": False,
            "context": "",
            "sources": [],
        }

    context_parts = []

    for result in relevant:
        context_parts.append(
            f"Section: {result['section']}\n"
            f"Source: {result['source']}\n"
            f"Information: {result['text']}"
        )

    return {
        "found": True,
        "context": "\n\n".join(context_parts),
        "sources": [
            {
                "section": result["section"],
                "source": result["source"],
                "score": result["score"],
            }
            for result in relevant
        ],
    }


if __name__ == "__main__":
    questions = [
        "What are the clinic opening hours?",
        "How can I cancel my appointment?",
        "How early should I request a repeat prescription?",
        "What should I do in a medical emergency?",
        "Does the clinic have free parking?",
    ]

    for question in questions:
        print("\n" + "=" * 55)
        print("QUESTION:", question)

        result = retrieve_context(question)

        print("CONTEXT FOUND:", result["found"])

        if result["found"]:
            for source in result["sources"]:
                print(
                    f"Section: {source['section']} "
                    f"| Score: {source['score']:.4f}"
                )
        else:
            print(
                "FALLBACK: I don't have verified "
                "information about that in the "
                "clinic knowledge base."
            )
