from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from _get_llm import get_llm
from _proj_vector_db import get_retriever

load_dotenv()


RETRIEVAL_K = 10


def run_rag_pipeline(query: str, history: str = "") -> tuple[str, list]:
    """
    Run the complete RAG pipeline:
    1. Query Rewriting (incorporating history)
    2. Retrieval (Pass 1)
    3. Multi-Hop Entity extraction and Pass 2 retrieval
    4. Answer generation

    Returns:
        Tuple of (generated_answer, list of retrieved Document objects)
    """
    llm = get_llm()
    retriever = get_retriever(k=RETRIEVAL_K)

    # 1. Query Rewriting (Front door)
    rewrite_message = f"""
You are in a conversation with a user, answering questions about MLOps and Trustworthy AI.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history if history else "No history yet."}

And this is the user's current question:
{query}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    rewritten_query = llm.invoke([HumanMessage(content=rewrite_message)]).content.strip()
    print("\n--- Query Rewriting ---")
    print(f"Original: {query}")
    print(f"Rewritten: {rewritten_query}")
    print("-----------------------\n")

    # 2. Retrieval (Pass 1)
    all_docs = retriever.invoke(rewritten_query)

    # 3. Deduplication
    seen_contents = set()
    pass1_unique = []
    for doc in all_docs:
        if doc.page_content not in seen_contents:
            pass1_unique.append(doc)
            seen_contents.add(doc.page_content)

    # Limit to top-k unique chunks for pass 1
    relevant_docs = pass1_unique[:RETRIEVAL_K]

    # --- 3.5. Multi-Hop: Entity Extraction & Pass 2 ---
    pass1_context = "\n\n".join([d.page_content for d in relevant_docs])
    entity_extraction_prompt = f"""
    Analyze the following context and extract the names of any people mentioned.
    Output ONLY a comma-separated list of names. If no people are mentioned, output "NONE".
    
    Context:
    {pass1_context}
    """
    entities_str = llm.invoke([HumanMessage(content=entity_extraction_prompt)]).content.strip()

    if entities_str and entities_str.upper() != "NONE" and "NONE" not in entities_str.upper():
        entities = [e.strip() for e in entities_str.split(",") if e.strip()]
        pass2_docs = []
        for entity in entities:
            # Query specifically for the entity
            entity_query = entity
            entity_docs = retriever.invoke(entity_query)
            print(f"\n--- Entity: {entity} ---")
            print(f"Retrieved {len(entity_docs)} documents for entity: {entity}")
            for doc in entity_docs:
                print(f"  - {doc.page_content}")
            print("-----------------------")

            pass2_docs.extend(entity_docs)

        # Deduplicate pass 2 docs against pass 1
        for doc in pass2_docs:
            if doc.page_content not in seen_contents:
                relevant_docs.append(doc)
                seen_contents.add(doc.page_content)

    # Limit final total to avoid context bloat before reranking (e.g., 30 chunks maximum)
    relevant_docs = relevant_docs[:30]

    # --- 3.75 LLM-based Reranking ---
    if len(relevant_docs) > 1:
        rerank_prompt = f"""
You are an expert relevance ranker. Given the user's original query and a list of retrieved documents, 
rank the documents by how relevant they are to answering the query. Focus on finding documents that either 
directly answer the query, or provide supporting context (like full names/titles) for entities mentioned in the answer.

User Query: {query}

Documents:
"""
        for i, doc in enumerate(relevant_docs):
            rerank_prompt += f"[Document {i}]\n{doc.page_content}\n\n"

        rerank_prompt += f"""
Analyze the documents and identify the most relevant ones to answer the User Query and provide entity context.
Output ONLY a comma-separated list of the top {min(10, len(relevant_docs))} most relevant Document IDs (integers).
Do not include any other text, brackets, or explanation.
Example output: 3, 0, 5, 1
"""
        rerank_response = llm.invoke([HumanMessage(content=rerank_prompt)]).content.strip()
        print("\n--- LLM Reranking ---")
        print(f"Reranker output: {rerank_response}")

        # Parse output safely
        try:
            import re

            ranked_indices = [int(idx) for idx in re.findall(r"\d+", rerank_response)]

            # Remove duplicates while preserving order
            seen_indices = set()
            unique_ranked_indices = []
            for idx in ranked_indices:
                if idx not in seen_indices and 0 <= idx < len(relevant_docs):
                    unique_ranked_indices.append(idx)
                    seen_indices.add(idx)

            # Reorder relevant_docs
            reranked_docs = [relevant_docs[i] for i in unique_ranked_indices]

            # Append any unranked docs at the end
            for i in range(len(relevant_docs)):
                if i not in seen_indices:
                    reranked_docs.append(relevant_docs[i])

            relevant_docs = reranked_docs
        except Exception as e:
            print(f"Reranking parse error: {e}")

    # Limit to top 10 chunks as requested
    relevant_docs = relevant_docs[:10]

    # 4. Answer Generation
    context_parts = []
    for i, doc in enumerate(relevant_docs):
        context_parts.append(f"Content from Document {i + 1}:\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    final_prompt = (
        "Be concise. Answer in 1-2 sentences maximum.\n"
        "Always use the full names and titles of any persons mentioned, if available in the context.\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n"
        f"Using only the context above, answer the question: {query}\n"
        "If the answer is not in the context, say 'I don't know.'"
    )

    response = llm.invoke([HumanMessage(content=final_prompt)])
    return response.content.strip(), relevant_docs
