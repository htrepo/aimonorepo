import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Set encoding to utf-8 for printing
sys.stdout.reconfigure(encoding='utf-8')

DB_NAME = "vectors_db"
PDF_PATH = "mlops-and-trustworthy-ai-for-data-leaders.pdf"

def debug_pdf():
    print(f"Loading PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    found = False
    for i, page in enumerate(pages):
        if "MaryAnn" in page.page_content or "Fleming" in page.page_content:
            print(f"Found 'MaryAnn' or 'Fleming' on page {i+1}")
            start_idx = page.page_content.find("Fleming")
            if start_idx == -1:
                start_idx = page.page_content.find("MaryAnn")
            
            ctx_start = max(0, start_idx - 150)
            ctx_end = min(len(page.page_content), start_idx + 150)
            print(f"Context: {page.page_content[ctx_start:ctx_end]}")
            found = True
    if not found:
        print("'MaryAnn Fleming' not found in PDF text.")

def debug_retrieval():
    print("\n--- DEBUG RETRIEVAL ---")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)
    
    query = "Who is MaryAnn Fleming?"
    print(f"Querying: {query}")
    results = vectorstore.similarity_search(query, k=5)
    
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        # Find if name is in content
        has_name = "MaryAnn" in res.page_content and "Fleming" in res.page_content
        print(f"Has name: {has_name}")
        print(f"Content snippet: {res.page_content[:300].replace('\n', ' ')}")
        if has_name:
            print("--- FULL CONTENT OF MATCHING CHUNK ---")
            print(res.page_content)
            print("--------------------------------------")

if __name__ == "__main__":
    debug_pdf()
    debug_retrieval()
