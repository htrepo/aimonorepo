import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Set encoding to utf-8 for printing
sys.stdout.reconfigure(encoding='utf-8')

DB_NAME = "vectors_db"

def find_rank(query, target_string):
    print(f"\n--- Finding rank for query: '{query}' ---")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)
    
    # Get more results to find it
    results = vectorstore.similarity_search(query, k=50)
    
    found_at = -1
    for i, res in enumerate(results):
        if target_string.lower() in res.page_content.lower():
            print(f"Found '{target_string}' at rank {i+1}")
            found_at = i + 1
            break
            
    if found_at == -1:
        print(f"'{target_string}' not found in top 50 results.")

if __name__ == "__main__":
    find_rank("maryann", "MaryAnn Fleming")
