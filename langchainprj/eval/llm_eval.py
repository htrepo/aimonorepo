import json
import re

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from _proj_rag import get_llm, run_rag_pipeline
from eval.test import TestQuestion

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(description="Concise feedback comparing the generated answer to the reference answer")
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
# Answer quality evaluation (LLM-as-a-judge)
# ---------------------------------------------------------------------------


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
    generated_answer, retrieved_docs = run_rag_pipeline(test.question)

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

    llm = get_llm()

    # Use structured JSON output via prompt instructions
    judge_prompt = (
        judge_messages[0]["content"]
        + "\n\n"
        + judge_messages[1]["content"]
        + "\n\nRespond ONLY with valid JSON matching this schema: "
        '{"feedback": "...", "accuracy": <float>, "completeness": <float>, "relevance": <float>}'
    )
    raw = llm.invoke([HumanMessage(content=judge_prompt)]).content.strip()

    # Parse JSON — strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    answer_eval = AnswerEval(**data)

    return answer_eval, generated_answer, retrieved_docs
