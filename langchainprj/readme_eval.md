# 📊 RAG Evaluation Framework — `eval/`

This document describes the evaluation suite for the **MLOps & Trustworthy AI Assistant** RAG pipeline.
It mirrors the pattern from `llm_engineering/week5/evaluation/` and adapts it to this project's
ChromaDB + Gemini stack.

---

# Table of Contents

1. [Overview](#overview)
2. [Retrieval Failure](#retrieval-failure)
3. [Generation Failure](#generation-failure)
4. [File Structure](#file-structure)
5. [TESTING](#testing)
6. [Extending the Test Suite](#extending-the-test-suite)

---

# Overview

A RAG system has **two distinct failure modes**:

| Failure Mode | What Goes Wrong | How We Detect It |
|---|---|---|
| **Retrieval failure** | The vector store doesn't surface the right chunks | Retrieval metrics (MRR, nDCG, keyword coverage) |
| **Generation failure** | The LLM has the right context but produces a bad answer | LLM-as-a-judge (accuracy, completeness, relevance) |

This framework evaluates both independently, giving a clear signal about *where* to focus improvements.

---

# Retrieval Failure

This section details how we measure and diagnose failures in the retrieval step — when the vector
store fails to surface the relevant context for a query.

## Retrieval Metrics

Retrieval evaluation checks whether the vector store returns chunks that *contain* the expected keywords.
This is a proxy for semantic relevance — if the right chunk is retrieved, the LLM has a chance of
answering correctly.

### Mean Reciprocal Rank (MRR)

- **Component**: Formula, Retriever

> **"How early does the first relevant chunk appear?"**

$$\text{MRR} = \frac{1}{|Q|} \sum_{q} \frac{1}{\text{rank}_q}$$

- For each keyword, find the rank (1-indexed position) of the **first** retrieved chunk that contains it.
- If the keyword is in chunk #1 → RR = 1.0 (best possible)
- If the keyword is in chunk #3 → RR = 0.33
- If the keyword is not found → RR = 0.0
- Average RR across all keywords for the question.

**What it tells you:** MRR is sensitive to whether the most relevant chunk appears at the very top.
A low MRR means the retriever is burying relevant chunks.

### Normalized Discounted Cumulative Gain (nDCG)

- **Component**: Formula, Retriever

> **"Are all relevant chunks ranked highly, with discounting for lower positions?"**

$$\text{nDCG} = \frac{\text{DCG}}{\text{IDCG}}$$

Where:

$$\text{DCG} = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)}$$

- **Binary relevance**: 1 if the chunk contains the keyword, 0 otherwise.
- **DCG** sums relevance scores, discounted by log position.
- **IDCG** is the DCG of the ideal ranking (all 1s at the top).
- **nDCG** = DCG / IDCG, normalized to [0, 1].

**What it tells you:** nDCG rewards retrievers that rank *all* relevant chunks highly, not just the first one.
It is especially useful when a question has multiple keywords spread across different chunks.

### Keyword Coverage

- **Component**: Formula, Retriever

> **"What fraction of expected keywords were found anywhere in the top-k results?"**

$$\text{Coverage} = \frac{\text{keywords found}}{|\text{total keywords}|} \times 100\%$$

- A keyword is "found" if its MRR > 0 (i.e., it appears in any retrieved chunk).
- Coverage of 100% means every keyword was surfaced; 0% means the retriever missed everything.

**What it tells you:** A quick health-check. If coverage is low, the embedding model or chunking
strategy is failing to represent the relevant content.

## Concepts & Methods

- **`fetch_context()`**: Retrieves the top-`k` document chunks from ChromaDB for a given question.
- **`calculate_mrr()`**: Computes the reciprocal rank for a single keyword.
- **`calculate_ndcg()`**: Computes nDCG for a single keyword.
- **`evaluate_retrieval()`**: Fetches context and computes MRR, nDCG, and keyword coverage.

## Benchmarking Retrieval Health

| MRR | nDCG | Interpretation |
|---|---|---|
| > 0.8 | > 0.8 | Excellent — relevant chunks appear at the top |
| 0.5–0.8 | 0.5–0.8 | Good — most keywords found but not always at rank 1 |
| 0.3–0.5 | 0.3–0.5 | Moderate — consider larger `k`, better chunking, or reranking |
| < 0.3 | < 0.3 | Poor — fundamental embedding or chunking issue |

---

# Generation Failure

This section details how we measure and diagnose failures in the generation step — when the LLM
has the correct context but fails to produce a high-quality answer.

## Answer Quality Metrics

Answer evaluation uses a **second LLM call** (LLM-as-a-judge) to score the generated answer against
the reference answer on three dimensions (1–5).

### Accuracy (1–5)

- **Component**: LLM

> **"Is the answer factually correct compared to the reference?"**

- Score 5: All facts are correct and match the reference answer.
- **Score 1 is mandatory if any fact is wrong.**

### Completeness (1–5)

- **Component**: LLM

> **"Does the answer cover all aspects of the reference answer?"**

- Score 5: ALL information from the reference answer is present.

### Relevance (1–5)

- **Component**: LLM

> **"Does the answer directly address the question without extraneous information?"**

- Score 5: Perfectly on-topic, no unnecessary fluff.

## Concepts & Methods

- **`answer_question()`**: Runs the full RAG pipeline (Retrieval + Generation).
- **`evaluate_answer()`**: Calls `answer_question()` then asks an LLM judge to score the result.

## Benchmarking Answer Quality

| Avg Score | Interpretation |
|---|---|
| 4.5–5.0 | Production-ready |
| 3.5–4.5 | Good — minor issues |
| 2.5–3.5 | Acceptable — noticeable gaps |
| < 2.5 | Needs work — check retrieval first |

---

# File Structure

```
langchainprj/
├── eval/
│   ├── __init__.py
│   ├── test.py
│   ├── tests.jsonl
│   └── main_eval.py
├── main_vectordb.py
├── main_gradio.py
└── readme_eval.md
```

---

# TESTING

## Test Data — `tests.jsonl`

The test file is a [JSONL](https://jsonlines.org/) file — one JSON object per line.
Each object represents a single test case with four fields:

```json
{
  "question": "What is MaryAnn's salary?",
  "keywords": ["MaryAnn", "1M", "dollars"],
  "reference_answer": "MaryAnn is being paid 1M dollars for her job.",
  "category": "direct_fact"
}
```

### Fields

| Field | Type | Purpose |
|---|---|---|
| `question` | `str` | Natural-language question |
| `keywords` | `list[str]` | Words/phrases that **must appear** in retrieved chunks |
| `reference_answer` | `str` | Gold-standard answer for comparison |
| `category` | `str` | Reasoning category being tested |

### Question Categories

| Category | What It Tests | Example |
|---|---|---|
| `direct_fact` | Single-hop lookup of an explicit fact | "What is MaryAnn's salary?" |
| `temporal` | Facts anchored to a time or sequence | "When should data governance be introduced?" |
| `comparative` | Comparing two things or quantifying a difference | "How does MLOps differ from DevOps?" |
| `numerical` | Exact numbers, counts, or statistics | "How many documents are indexed?" |
| `relationship` | How two entities are connected | "What are the components of an MLOps pipeline?" |
| `spanning` | Requires synthesising info across multiple chunks | "Combining both documents, what are the critical success factors?" |
| `holistic` | Requires reasoning over the entire corpus | "What are all the risks of deploying AI without monitoring?" |

## Data Model — `test.py`

```python
class TestQuestion(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str
```

**`load_tests()`**: Reads the JSONL file and returns a list of `TestQuestion` instances.

## CLI Execution

To run a single test and see full feedback:

```powershell
# Run Test #0
& .venv/Scripts/python.exe eval/main_eval.py 0
```

## Diagnosing Problems

When the system underperforms, compare retrieval and answer scores to locate the bottleneck:

| Scenario | Diagnosis |
|---|---|
| High retrieval + low answer scores | LLM prompt engineering issue |
| Low retrieval + low answer scores | Fix the vector store/embedding first |
| High retrieval + high answer scores | System is working well |
| Low retrieval + high answer scores | LLM is hallucinating (memorized answers) |

---

# Extending the Test Suite

### Adding new test cases

Append a new JSON line to `eval/tests.jsonl`:

```json
{"question": "Your new question?", "keywords": ["key1", "key2"], "reference_answer": "The ideal answer.", "category": "direct_fact"}
```

### Configuration

Edit `eval/main_eval.py`:
- `MODEL`: Switch between `gemini-1.5-flash` or `gemini-1.5-pro` for judging.
- `TOP_K`: Number of chunks to retrieve (default: 10).
