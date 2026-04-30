import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer for the MLOps RAG assistant."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(
        description=(
            "Question category: direct_fact | temporal | comparative | numerical | "
            "relationship | spanning | holistic"
        )
    )


def load_tests(test_file: str = TEST_FILE) -> list[TestQuestion]:
    """Load test questions from a JSONL file.

    Args:
        test_file: Path to the JSONL file. Defaults to the bundled tests.jsonl.

    Returns:
        List of TestQuestion objects.
    """
    tests = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                tests.append(TestQuestion(**data))
    return tests
