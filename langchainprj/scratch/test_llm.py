from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

MODEL = "llama3.2:latest"
DB_NAME = "vectors_db"

SYSTEM_PROMPT_TEMPLATE = """
You are an expert in AI and Machine Learning. 
You are chatting with a user about MLOps and trustworthy AI for data leaders.
If relevant, use the given context to answer the user's question.
Do not answer any questions that are not related to MLOps and trustworthy AI for data leaders.
Answer the user's question based on the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}
"""

def test_query():
    embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=MODEL, temperature=0)

    message = "Who is MaryAnn Fleming?"
    
    # 1. Retrieve
    relevant_docs = retriever.invoke(message)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print("--- RETRIEVED CONTEXT ---")
    print(context)
    print("-------------------------")

    # 2. Call LLM
    system_message = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context))
    messages = [system_message, HumanMessage(content=message)]
    
    print("\n--- LLM RESPONSE ---")
    response = llm.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    test_query()
