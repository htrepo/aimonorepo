import math

from pydantic import BaseModel, Field

from src._proj_rag import RETRIEVAL_K
from src._proj_vector_db import get_retriever
from eval.test import TestQuestion

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank — average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in the top-k retrieved chunks")
    total_keywords: int = Field(description="Total number of keywords being evaluated")
    keyword_coverage: float = Field(description="Percentage of keywords found in retrieved chunks (0–100)")


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def fetch_context(question: str) -> list:
    """Retrieve the top-k document chunks for a given question.

    Args:
        question: The natural-language question to retrieve context for.

    Returns:
        List of LangChain Document objects ordered by relevance score.
    """
    return get_retriever(k=RETRIEVAL_K).invoke(question)


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate the Reciprocal Rank for a single keyword (case-insensitive).
    mrr means -> mean reciprocal rank - it is a measure of ranking quality.
    The Reciprocal Rank is 1/rank where rank is the position (1-indexed) of the
    first retrieved document containing the keyword. Returns 0.0 if the keyword
    is not found in any document.

    Args:
        keyword: The keyword to search for.
        retrieved_docs: Ordered list of retrieved LangChain Document objects.

    Returns:
        Reciprocal rank score in [0, 1].
    """
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain for a ranked list of binary relevances.

    DCG = sum_{i=1}^{k} rel_i / log2(i + 1)

    Args:
        relevances: List of binary relevance scores (1 = relevant, 0 = not relevant).
        k: Cut-off rank.

    Returns:
        DCG value.
    """
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because ranks start at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = RETRIEVAL_K) -> float:
    """Calculate Normalized DCG for a single keyword using binary relevance.

    nDCG = DCG / IDCG, where IDCG is the DCG of the ideal (best-case) ranking.
    Binary relevance: a document is relevant (1) if it contains the keyword.

    Args:
        keyword: The keyword to use as the relevance signal.
        retrieved_docs: Ordered list of retrieved LangChain Document objects.
        k: Cut-off rank (default RETRIEVAL_K).

    Returns:
        nDCG score in [0, 1].
    """
    keyword_lower = keyword.lower()

    # Binary relevance for each retrieved doc
    relevances: list[int] = [1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]]

    dcg: float = calculate_dcg(relevances, k)

    # Ideal DCG — best case is all relevant docs at the top
    ideal_relevances: list[int] = sorted(relevances, reverse=True)
    idcg: float = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = RETRIEVAL_K) -> RetrievalEval:
    """Evaluate retrieval performance for a single test question.

    Retrieves the top-k chunks from ChromaDB and computes:
      - Mean Reciprocal Rank (MRR) averaged across all keywords
      - Normalized DCG (nDCG) averaged across all keywords
      - Keyword coverage (how many keywords appear in at least one retrieved chunk)

    Args:
        test: TestQuestion object containing the question and expected keywords.
        k: Number of top documents to retrieve (default RETRIEVAL_K = 10).

    Returns:
        RetrievalEval object with all metrics populated.
    """
    retrieved_docs = fetch_context(test.question)

    # MRR: average reciprocal rank across all keywords
    mrr_scores: [float] = [calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    avg_mrr: float = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # nDCG: average normalized DCG across all keywords
    ndcg_scores: [float] = [calculate_ndcg(kw, retrieved_docs, k) for kw in test.keywords]
    avg_ndcg: float = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Keyword coverage: fraction of keywords found in any retrieved chunk
    keywords_found: int = sum(1 for score in mrr_scores if score > 0)
    total_keywords: int = len(test.keywords)
    keyword_coverage: float = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )
