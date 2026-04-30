
import sys
import io
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Force UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MODEL = "mistral:7b"
DB_NAME = "vectors_db"
embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
llm = ChatOllama(model=MODEL, temperature=0)

def test_final_simpler(message):
    print(f"\n=== TESTING SIMPLER PROMPT: {message} ===")
    
    docs = retriever.invoke(message)
    context_parts = []
    for i, doc in enumerate(docs[:10]):
        context_parts.append(f"Content from Document {i+1}:\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    final_prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the question: {message}
    
    Rules to follow:
    - If any chunk mentions "MaryAnn" being paid, it refers to "MaryAnn Fleming".
    - You MUST include the salary/payment info in your answer if it is mentioned in the context.
    - If you don't know the answer, say "I don't know."
    """

    response = llm.invoke([HumanMessage(content=final_prompt)])
    print("\n--- FINAL ANSWER ---")
    print(response.content)
    print("--------------------")

if __name__ == "__main__":
    test_final_simpler("how much maryann is paid")
