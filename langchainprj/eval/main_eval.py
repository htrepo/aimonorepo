"""
eval.py — RAG Evaluation Framework for the MLOps & Trustworthy AI Assistant.

Evaluates two orthogonal dimensions of RAG performance:
  1. Retrieval Quality  — Did the vector store surface the right chunks?
  2. Answer Quality     — Did the LLM produce an accurate, complete, relevant answer?

Usage (CLI):
    python eval/main_eval.py <test_row_number>
"""

import sys
from pathlib import Path

# Make the project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from eval.llm_eval import evaluate_answer
from eval.retriever_eval import evaluate_retrieval
from eval.test import load_tests  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(override=True)


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
        val = input(f"Enter test row number to evaluate (0-{len(tests) - 1}): ").strip()
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
