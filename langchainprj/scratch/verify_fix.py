from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

MODEL = "llama3.2:latest"
DB_NAME = "vectors_db"

SYSTEM_PROMPT_TEMPLATE = """
You are an expert in AI and Machine Learning. 
You are chatting with a user about MLOps and trustworthy AI for data leaders.
The following context is extracted from a document on MLOps and Trustworthy AI.
Use the context to answer the user's question. If the question is about a person or organization mentioned in the context, provide the details available.
If the answer is not in the context, say "I don't know."

Context:
{context}
"""

def test_query():
    embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)
    # Use k=10 as in the updated main_gradio.py
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatOllama(model=MODEL, temperature=0)

    # Use the specific short query that failed before
    message = "maryann"
    
    print(f"Querying: {message}")
    relevant_docs = retriever.invoke(message)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print(f"Retrieved {len(relevant_docs)} documents.")
    if "MaryAnn Fleming" in context:
        print("SUCCESS: 'MaryAnn Fleming' found in retrieved context.")
    else:
        print("FAILURE: 'MaryAnn Fleming' NOT found in retrieved context.")

    # Call LLM
    system_message = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context))
    messages = [system_message, HumanMessage(content=message)]
    
    print("\n--- LLM RESPONSE ---")
    response = llm.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    test_query()
