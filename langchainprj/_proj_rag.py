from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from _proj_vector_db import get_retriever

load_dotenv()

# Configuration
MODEL = "gemini-2.5-flash-lite"
RETRIEVAL_K = 10

# Lazy-initialized LLM
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    return _llm


def run_rag_pipeline(query: str) -> tuple[str, list]:
    """
    Run the complete RAG pipeline:
    1. Multi-query expansion
    2. Retrieval and deduplication
    3. Answer generation

    Returns:
        Tuple of (generated_answer, list of retrieved Document objects)
    """
    llm = get_llm()
    retriever = get_retriever(k=RETRIEVAL_K)

    # 1. Multi-Query Expansion
    expansion_prompt = f"""
    Given the user's question, generate 2 additional search queries to find context.
    Focus on finding the identity/role of any people mentioned.
    
    Original Question: {query}
    
    Output ONLY the 2 queries, one per line.
    """
    expansion_response = llm.invoke([HumanMessage(content=expansion_prompt)]).content.strip()
    queries = [query] + [q.strip() for q in expansion_response.split("\n") if q.strip()]

    # 2. Retrieval
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))

    # 3. Deduplication
    seen_contents = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)

    # Limit to top-k unique chunks
    relevant_docs = unique_docs[:RETRIEVAL_K]

    # 4. Answer Generation
    context_parts = []
    for i, doc in enumerate(relevant_docs):
        context_parts.append(f"Content from Document {i + 1}:\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    final_prompt = (
        "Be concise. Answer in 1-2 sentences maximum.\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n"
        f"Using only the context above, answer the question: {query}\n"
        "If the answer is not in the context, say 'I don't know.'"
    )

    response = llm.invoke([HumanMessage(content=final_prompt)])
    return response.content.strip(), relevant_docs
