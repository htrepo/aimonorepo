import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Set encoding to utf-8 for printing
sys.stdout.reconfigure(encoding='utf-8')

DB_NAME = "vectors_db"

def test_query(query):
    print(f"\n--- Testing Query: '{query}' ---")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)
    
    results = vectorstore.similarity_search(query, k=5)
    
    for i, res in enumerate(results):
        has_name = "MaryAnn" in res.page_content
        print(f"Result {i+1} (Has 'MaryAnn': {has_name}): {res.page_content[:100].replace('\n', ' ')}...")

if __name__ == "__main__":
    test_query("Who is MaryAnn Fleming?")
    test_query("maryann")
