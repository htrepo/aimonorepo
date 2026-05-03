from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from _get_llm import get_llm
from _proj_vector_db import get_retriever
from _rerank_chunks import rerank_chunks

load_dotenv()


RETRIEVAL_K = 10


def run_rag_pipeline(query: str, history: str = "") -> tuple[str, list]:
    """
    Run the complete RAG pipeline:
    1. Query Rewriting (incorporating history)
    2. Retrieval (Pass 1)
    3. Multi-Hop Entity extraction and Pass 2 retrieval
    4. Answer generation
`
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

    # Rerank Pass 1 immediately to get the best context for entity extraction
    relevant_docs: list = rerank_chunks(query, pass1_unique, llm, top_k=5)
    print(f"length of reranked docs = {len(relevant_docs)}")

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
        pass2_unique = []
        for entity in entities:
            # Query specifically for the entity
            entity_query = entity
            entity_docs = retriever.invoke(entity_query)
            print(f"\n--- Entity: {entity} ---")
            print(f"Retrieved {len(entity_docs)} documents for entity: {entity}")
            for doc in entity_docs:
                print(f"  - {doc.page_content}")
            print("-----------------------")

            # Deduplicate against pass 1 and previous entities
            for doc in entity_docs:
                if doc.page_content not in seen_contents:
                    pass2_unique.append(doc)
                    seen_contents.add(doc.page_content)

            # --- 3.75 LLM-based Reranking ---
            print("\n--- Reranking Pass 2 ---")
            # relevant_docs is already reranked Pass 1 (top 5)
            top_pass2 = rerank_chunks(query, pass2_unique, llm, top_k=5)
            print(f"length of pass2_unique = {len(pass2_unique)}")
            print(f"length of top_pass2 = {len(top_pass2)}")
            relevant_docs = relevant_docs + top_pass2
            print(f"length of relevant docs = {len(relevant_docs)}")
    # else: If no entities, relevant_docs is already the reranked Pass 1

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

    print(f"\n--- Final Prompt ---\n{final_prompt}\n----------------------\n")

    response = llm.invoke([HumanMessage(content=final_prompt)])
    return response.content.strip(), relevant_docs
