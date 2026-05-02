"""
query_db.py - Inspect vector DB results for a given text query.

Usage:
    uv run python query_db.py "MaryAnn"
    uv run python query_db.py "MaryAnn" --k 15
"""

import argparse
import sys

from _proj_embedding import embeddings_model
from _proj_vector_db import get_vectorstore

sys.stdout.reconfigure(encoding="utf-8")


def query_db(text: str, k: int = 10) -> None:
    """
    Convert text to embedding, fetch top-k matching chunks from the vector DB,
    and print them with similarity scores.
    """
    # 1. Embed the query text
    print(f"\n=== Query: {text!r} | Top-{k} results ===\n")
    embedding_vector = embeddings_model.embed_query(text)
    print(f"Embedding dim: {len(embedding_vector)}")

    # 2. Connect to Chroma and run similarity_search_by_vector
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(text, k=k)

    # 3. Print results
    print(f"\nFetched {len(results)} chunks:\n")
    print("-" * 60)
    for i, (doc, score) in enumerate(results):
        print(f"[{i}] Score: {score:.4f}")
        print(doc.page_content)
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the vector DB by text.")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="The text to search for (optional, will prompt if not provided).",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of chunks to return (default: 10).")
    args = parser.parse_args()

    query_text = args.query
    if not query_text:
        query_text = input("Enter search query: ").strip()

    query_db(query_text, k=args.k)
