import re

from langchain_core.messages import HumanMessage


def rerank_chunks(query: str, documents: list, llm, top_k: int = 5) -> list:
    """
    Reranks a list of documents based on relevance to the query using an LLM.
    Returns the top_k most relevant documents.
    """
    if not documents:
        return []
    if len(documents) <= 1:
        return documents[:top_k]

    rerank_prompt = f"""
You are an expert relevance ranker. Given the user's original query and a list of retrieved documents, 
rank the documents by how relevant they are to answering the query. Focus on finding documents that either 
directly answer the query, or provide supporting context (like full names/titles) for entities mentioned in the answer.

User Query: {query}

Documents:
"""
    for i, doc in enumerate(documents):
        rerank_prompt += f"[Document {i}]\n{doc.page_content}\n\n"

    rerank_prompt += f"""
Analyze the documents and identify the most relevant ones to answer the User Query and provide entity context.
Output ONLY a comma-separated list of the top {min(top_k, len(documents))} most relevant Document IDs (integers).
Do not include any other text, brackets, or explanation.
Example output: 3, 0, 5, 1
"""
    try:
        response = llm.invoke([HumanMessage(content=rerank_prompt)])
        rerank_response = response.content.strip()

        # Parse output safely
        ranked_indices = [int(idx) for idx in re.findall(r"\d+", rerank_response)]

        # Remove duplicates while preserving order
        seen_indices = set()
        unique_ranked_indices = []
        for idx in ranked_indices:
            if idx not in seen_indices and 0 <= idx < len(documents):
                unique_ranked_indices.append(idx)
                seen_indices.add(idx)

        # Reorder documents and take top_k (though prompt already limited it)
        reranked_docs = [documents[i] for i in unique_ranked_indices]

        # If the LLM failed to return enough or any, return the original order up to top_k
        if not reranked_docs:
            return documents[:top_k]

        return reranked_docs[:top_k]

    except Exception as e:
        print(f"Reranking error: {e}")
        # Fallback to original retrieval order
        return documents[:top_k]
