# 🛡️ MLOps & Trustworthy AI Assistant

Expert guidance for data leaders on operationalizing AI with trust. This project implements a robust RAG (Retrieval-Augmented Generation) pipeline with advanced features like Multi-Hop Retrieval, Query Rewriting, and a comprehensive Evaluation Framework.

---

## 🚀 Quick Start

### 1. Installation
Ensure you have `uv` installed. Then run:
```powershell
uv sync
```

### 2. Setup Environment
Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
HF_TOKEN=your_token
```

### 3. Build Vector Database
```powershell
uv run python src/main_vectordb.py
```

### 4. Launch Assistant
```powershell
uv run python src/main_gradio.py
```

---

## 🏗️ RAG Architecture: Multi-Hop & Query Rewriting

The pipeline is designed to overcome a common failure case in standard single-pass RAG: **Missing Context across Disparate Documents (Entity Disconnects)**.

### The Problem
In standard RAG, when a user asks *"who is paid 1M dollars"*, the search typically only finds chunks matching "1M dollars". If the document containing the salary only says `"MaryAnn is paid 1M dollars"`, but a separate document contains her full title (`"MaryAnn Fleming, Head of Homebuying Services"`), the generation model literally does not know her full name on the first shot.

### The Solution: Two-Pass (Multi-Hop) RAG + Query Rewriting

To solve this, the pipeline is divided into the following sequential steps:

#### 1. Unified Query Rewriting ("The Front Door")
Instead of generating multiple expansion queries or running a separate conversational contextualization step, we pass both the **user's query** and the **conversation history** to an initial LLM prompt. 
- The LLM rewrites the query into a single, highly specific search string optimized for vector retrieval.
- This effectively handles conversational references (like "who is she?") and extracts maximum possible intent from the user's input.

#### 2. Pass 1 Retrieval
We use the rewritten query to retrieve the top `K` most relevant chunks. In the "MaryAnn" example, this pass retrieves the chunk containing `"MaryAnn is being paid 1M dollars"`.

#### 3. Multi-Hop Entity Extraction (The "Bridge")
We take the context retrieved in Pass 1 and ask the LLM to extract the names of any people mentioned.
- If it finds `"MaryAnn"`, we generate a new, specific query: `"MaryAnn full name job title role"`.

#### 4. Pass 2 Retrieval (Entity Context)
We query the vector database again using the specific entity queries to fetch secondary context. This successfully retrieves the chunk containing `"MaryAnn Fleming, Head of Homebuying Services"`.

#### 5. Deduplication and Context Merging
The chunks from Pass 1 and Pass 2 are combined, deduplicated, and passed into the final Answer Generation prompt.

#### 6. Answer Generation
The final LLM prompt is fed the enriched context and is explicitly instructed:
*"Always use the full names and titles of any persons mentioned, if available in the context."*

### Benefits
- **Zero-Shot Entity Resolution**: The LLM successfully outputs full entity names in its first response, without requiring the user to ask a follow-up question.
- **Efficient Contextualization**: By shifting the history contextualization from the Gradio frontend (`main_gradio.py`) directly into the RAG pipeline (`_proj_rag.py`), the codebase is cleaner and the rewrite step serves a dual purpose.
- **Robust against Hallucination**: The model is no longer forced to guess or hallucinate last names, as the required context is reliably bridged into the prompt.

---

## 📊 Evaluation Framework — `eval/`

Evaluates two orthogonal dimensions of RAG performance:
1.  **Retrieval Quality**: Did the vector store surface the right chunks? (MRR, nDCG, Keyword Coverage)
2.  **Answer Quality**: Did the LLM produce an accurate, complete, relevant answer? (LLM-as-a-judge)

### Failure Modes

| Failure Mode | What Goes Wrong | How We Detect It |
|---|---|---|
| **Retrieval failure** | The vector store doesn't surface the right chunks | Retrieval metrics (MRR, nDCG, keyword coverage) |
| **Generation failure** | The LLM has the right context but produces a bad answer | LLM-as-a-judge (accuracy, completeness, relevance) |

### Retrieval Metrics

- **Mean Reciprocal Rank (MRR)**: Measures how early the first relevant chunk appears.
- **Normalized Discounted Cumulative Gain (nDCG)**: Rewards retrievers that rank all relevant chunks highly.
- **Keyword Coverage**: Fraction of expected keywords found in the top-k results.

### Answer Quality Metrics (LLM-as-a-judge)

- **Accuracy (1–5)**: Is the answer factually correct?
- **Completeness (1–5)**: Does it cover all aspects of the reference?
- **Relevance (1–5)**: Does it directly address the question?

### CLI Execution
To run an evaluation for a specific test case:
```powershell
uv run python eval/main_eval.py <test_row_number>
```

---

## 📂 Project Structure

```
langchainprj/
├── src/                # Core source code
│   ├── main_gradio.py      # UI Entry point
│   ├── main_vectordb.py    # DB Builder
│   ├── _proj_rag.py        # RAG Logic
│   ├── _proj_vector_db.py  # DB Config
│   ├── _proj_embedding.py  # Embedding Config
│   └── _get_llm.py         # LLM Provider
├── eval/               # Evaluation suite
│   ├── main_eval.py        # Eval CLI
│   ├── retriever_eval.py   # Retrieval metrics
│   ├── llm_eval.py         # Judge logic
│   └── tests.jsonl         # Test cases
├── documents/          # Source PDFs/TXTs
└── vectors_db/         # Persisted ChromaDB
```

---

## 📝 Appendix: Sample Responses

### Performance on ICP Crises
- **Predictive Modeling**: Penn Medicine uses AI to identify ICP crises earlier in TBIs.
- **Reduced Lead-Time**: AI accelerates detection, decreasing intervention time.
- **Decision Support**: Supports data-driven clinical decisions.

### Domain Guardrail
If asked about topics outside MLOps/AI:
> "I don't know. My expertise is limited to MLOps and trustworthy AI for data leaders."

### Detailed Response Logs
<details>
<summary>View detailed sample responses</summary>

#### Response with context:
- **Predictive Modeling for Intracranial Hypertension (ICP Crises):** Penn Medicine uses AI-driven predictive modeling to identify and treat intracranial hypertension (ICP crises) earlier in patients with severe traumatic brain injuries (TBIs).
- **Reduced Lead-Time for Neurosurgeons:** AI accelerates the detection and intervention process, decreasing the time between identifying an ICP crisis and taking action.
- **Enhanced Treatment Decision-Making:** AI supports data-driven clinical decisions, helping clinicians optimize treatment strategies.

#### Response without context (General Knowledge):
- **Precision Medicine & Diagnostics**: Early disease detection through deep learning-based imaging analysis.
- **Predictive Analytics & Patient Care**: Predicting patient deterioration via real-time EHR analysis.
- **Drug Discovery & Research Acceleration**: Accelerating drug discovery using platforms like AlphaFold.
</details>
