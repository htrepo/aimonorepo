"""
eval.py — RAG Evaluation Framework for the MLOps & Trustworthy AI Assistant.

Evaluates two orthogonal dimensions of RAG performance:
  1. Retrieval Quality  — Did the vector store surface the right chunks?
  2. Answer Quality     — Did the LLM produce an accurate, complete, relevant answer?

Usage (CLI):
    python eval/main_eval.py <test_row_number>

Usage (programmatic):
    from eval.eval import evaluate_all_retrieval, evaluate_all_answers
"""

import math
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Make the eval package importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.test import TestQuestion, load_tests  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"          # LLM-as-a-judge model
DB_NAME = "vectors_db"                    # ChromaDB persist directory (relative to project root)
EMBEDDING_MODEL = "all-mpnet-base-v2"    # Must match the model used in main_vectordb.py
TOP_K = 10                               # Number of chunks to retrieve per query


# ---------------------------------------------------------------------------
# Lazy-initialized singletons (avoid loading GPU models at import time)
# ---------------------------------------------------------------------------
_retriever = None
_llm = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        project_root = Path(__file__).resolve().parent.parent
        db_path = str(project_root / DB_NAME)
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings_model)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return _retriever


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    return _llm


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


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback comparing the generated answer to the reference answer"
    )
    accuracy: float = Field(
        description=(
            "How factually correct is the answer compared to the reference? "
            "1 (wrong – any wrong answer must score 1) to 5 (perfect)."
        )
    )
    completeness: float = Field(
        description=(
            "How complete is the answer in addressing all aspects of the question? "
            "1 (very poor – missing key info) to 5 (all reference info is present). "
            "Only score 5 if ALL reference information is included."
        )
    )
    relevance: float = Field(
        description=(
            "How relevant is the answer to the specific question? "
            "1 (off-topic) to 5 (directly answers the question with no extraneous info). "
            "Only score 5 if the answer is completely on-topic."
        )
    )


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
    return _get_retriever().invoke(question)


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate the Reciprocal Rank for a single keyword (case-insensitive).

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


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = TOP_K) -> float:
    """Calculate Normalized DCG for a single keyword using binary relevance.

    nDCG = DCG / IDCG, where IDCG is the DCG of the ideal (best-case) ranking.
    Binary relevance: a document is relevant (1) if it contains the keyword.

    Args:
        keyword: The keyword to use as the relevance signal.
        retrieved_docs: Ordered list of retrieved LangChain Document objects.
        k: Cut-off rank (default TOP_K).

    Returns:
        nDCG score in [0, 1].
    """
    keyword_lower = keyword.lower()

    # Binary relevance for each retrieved doc
    relevances = [1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]]

    dcg = calculate_dcg(relevances, k)

    # Ideal DCG — best case is all relevant docs at the top
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = TOP_K) -> RetrievalEval:
    """Evaluate retrieval performance for a single test question.

    Retrieves the top-k chunks from ChromaDB and computes:
      - Mean Reciprocal Rank (MRR) averaged across all keywords
      - Normalized DCG (nDCG) averaged across all keywords
      - Keyword coverage (how many keywords appear in at least one retrieved chunk)

    Args:
        test: TestQuestion object containing the question and expected keywords.
        k: Number of top documents to retrieve (default TOP_K = 10).

    Returns:
        RetrievalEval object with all metrics populated.
    """
    retrieved_docs = fetch_context(test.question)

    # MRR: average reciprocal rank across all keywords
    mrr_scores = [calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # nDCG: average normalized DCG across all keywords
    ndcg_scores = [calculate_ndcg(kw, retrieved_docs, k) for kw in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Keyword coverage: fraction of keywords found in any retrieved chunk
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


# ---------------------------------------------------------------------------
# Answer quality evaluation (LLM-as-a-judge)
# ---------------------------------------------------------------------------


def answer_question(question: str) -> tuple[str, list]:
    """Run the RAG pipeline to generate an answer for a question.

    Retrieves context from ChromaDB, then calls the LLM with a grounded prompt.

    Args:
        question: Natural-language question.

    Returns:
        Tuple of (generated_answer string, list of retrieved Document objects).
    """
    retrieved_docs = fetch_context(question)

    context_parts = [f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)]
    context = "\n\n".join(context_parts)

    prompt = (
        "Be concise. Answer in 1-2 sentences maximum.\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n"
        f"Using only the context above, answer the question: {question}\n"
        "If the answer is not in the context, say 'I don't know.'"
    )

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip(), retrieved_docs


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """Evaluate answer quality using LLM-as-a-judge.

    Generates an answer via the RAG pipeline, then asks a second LLM call to
    judge the answer on three dimensions: accuracy, completeness, and relevance.

    Args:
        test: TestQuestion object containing the question and reference answer.

    Returns:
        Tuple of:
          - AnswerEval object with feedback and scores
          - The generated answer string
          - The list of retrieved Document objects
    """
    generated_answer, retrieved_docs = answer_question(test.question)

    # LLM-as-a-judge prompt
    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator assessing the quality of answers from an AI assistant. "
                "Evaluate the generated answer by comparing it to the reference answer. "
                "Be strict: only give 5/5 scores for truly perfect answers."
            ),
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? If the answer is wrong, score must be 1.
2. Completeness: How thoroughly does it address all aspects, covering all information from the reference answer?
3. Relevance: How well does it directly answer the specific question, giving no unnecessary additional information?

Respond with:
- feedback: A concise 2-3 sentence evaluation.
- accuracy: Score from 1 (wrong) to 5 (perfect).
- completeness: Score from 1 (missing key info) to 5 (all reference info present).
- relevance: Score from 1 (off-topic) to 5 (directly on-point).""",
        },
    ]

    llm = _get_llm()

    # Use structured JSON output via prompt instructions
    judge_prompt = (
        judge_messages[0]["content"] + "\n\n" + judge_messages[1]["content"] +
        "\n\nRespond ONLY with valid JSON matching this schema: "
        '{"feedback": "...", "accuracy": <float>, "completeness": <float>, "relevance": <float>}'
    )
    raw = llm.invoke([HumanMessage(content=judge_prompt)]).content.strip()

    # Parse JSON — strip markdown fences if present
    import json, re  # noqa: E401
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    answer_eval = AnswerEval(**data)

    return answer_eval, generated_answer, retrieved_docs


# ---------------------------------------------------------------------------
# Batch evaluation generators
# ---------------------------------------------------------------------------


def evaluate_all_retrieval(test_file: str | None = None):
    """Generator that evaluates retrieval for every test in the test file.

    Yields:
        Tuple of (TestQuestion, RetrievalEval, progress_fraction) for each test.
    """
    tests = load_tests(test_file) if test_file else load_tests()
    total = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total
        yield test, result, progress


def evaluate_all_answers(test_file: str | None = None):
    """Generator that evaluates answer quality for every test in the test file.

    Yields:
        Tuple of (TestQuestion, AnswerEval, progress_fraction) for each test.
    """
    tests = load_tests(test_file) if test_file else load_tests()
    total = len(tests)
    for index, test in enumerate(tests):
        result, _, _ = evaluate_answer(test)
        progress = (index + 1) / total
        yield test, result, progress


def aggregate_retrieval_results(results: list[tuple[TestQuestion, RetrievalEval]]) -> dict:
    """Aggregate retrieval metrics across all tests.

    Args:
        results: List of (TestQuestion, RetrievalEval) tuples.

    Returns:
        Dictionary with avg_mrr, avg_ndcg, avg_keyword_coverage, and per-category breakdown.
    """
    if not results:
        return {}

    total = len(results)
    avg_mrr = sum(r.mrr for _, r in results) / total
    avg_ndcg = sum(r.ndcg for _, r in results) / total
    avg_coverage = sum(r.keyword_coverage for _, r in results) / total

    # Per-category breakdown
    by_category: dict[str, list[RetrievalEval]] = {}
    for test, result in results:
        by_category.setdefault(test.category, []).append(result)

    category_stats = {}
    for cat, cat_results in by_category.items():
        n = len(cat_results)
        category_stats[cat] = {
            "count": n,
            "avg_mrr": sum(r.mrr for r in cat_results) / n,
            "avg_ndcg": sum(r.ndcg for r in cat_results) / n,
            "avg_keyword_coverage": sum(r.keyword_coverage for r in cat_results) / n,
        }

    return {
        "total_tests": total,
        "avg_mrr": round(avg_mrr, 4),
        "avg_ndcg": round(avg_ndcg, 4),
        "avg_keyword_coverage": round(avg_coverage, 2),
        "by_category": category_stats,
    }


def aggregate_answer_results(results: list[tuple[TestQuestion, AnswerEval]]) -> dict:
    """Aggregate answer quality metrics across all tests.

    Args:
        results: List of (TestQuestion, AnswerEval) tuples.

    Returns:
        Dictionary with avg_accuracy, avg_completeness, avg_relevance, and per-category breakdown.
    """
    if not results:
        return {}

    total = len(results)
    avg_accuracy = sum(r.accuracy for _, r in results) / total
    avg_completeness = sum(r.completeness for _, r in results) / total
    avg_relevance = sum(r.relevance for _, r in results) / total

    by_category: dict[str, list[AnswerEval]] = {}
    for test, result in results:
        by_category.setdefault(test.category, []).append(result)

    category_stats = {}
    for cat, cat_results in by_category.items():
        n = len(cat_results)
        category_stats[cat] = {
            "count": n,
            "avg_accuracy": round(sum(r.accuracy for r in cat_results) / n, 2),
            "avg_completeness": round(sum(r.completeness for r in cat_results) / n, 2),
            "avg_relevance": round(sum(r.relevance for r in cat_results) / n, 2),
        }

    return {
        "total_tests": total,
        "avg_accuracy": round(avg_accuracy, 2),
        "avg_completeness": round(avg_completeness, 2),
        "avg_relevance": round(avg_relevance, 2),
        "by_category": category_stats,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_cli_evaluation(test_number: int) -> None:
    """Run full retrieval + answer evaluation for a single test by index.

    Args:
        test_number: Zero-based index into the test file.
    """
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question  : {test.question}")
    print(f"Keywords  : {test.keywords}")
    print(f"Category  : {test.category}")
    print(f"Reference : {test.reference_answer}")

    # ── Retrieval Evaluation ──────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR              : {retrieval_result.mrr:.4f}")
    print(f"nDCG             : {retrieval_result.ndcg:.4f}")
    print(f"Keywords Found   : {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage : {retrieval_result.keyword_coverage:.1f}%")

    # ── Answer Evaluation ────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("Answer Evaluation (LLM-as-a-Judge)")
    print(f"{'=' * 80}")

    answer_result, generated_answer, _ = evaluate_answer(test)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores (1 = very poor, 5 = ideal):")
    print(f"  Accuracy     : {answer_result.accuracy:.2f}/5")
    print(f"  Completeness : {answer_result.completeness:.2f}/5")
    print(f"  Relevance    : {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main() -> None:
    """CLI entry point. Prompts for test number."""
    tests = load_tests()
    print(f"\n[INFO] Loaded {len(tests)} test cases.")

    try:
        val = input(f"Enter test row number to evaluate (0-{len(tests)-1}): ").strip()
        if not val:
            print("Exiting.")
            return
        test_number = int(val)
    except ValueError:
        print("Error: Please enter a valid integer.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()
