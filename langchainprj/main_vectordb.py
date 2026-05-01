from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from _proj_embedding import embeddings_model
from _proj_vector_db import DB_NAME

load_dotenv()

# The tokenizer used for estimating chunk limits
TOKENIZER_MODEL = "mistralai/Mistral-7B-v0.1"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60


# read pdf and return the list of documents (one per page)
def read_pdf(file_path: str) -> list[Document]:
    print(f"Loading PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Total pages loaded: {len(pages)}")
    return pages


# create Document type from content
def create_document(content: str) -> Document:
    return Document(page_content=content)


# split documents into chunks
def split_documents(documents: list[Document], tokenizer) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,  # Increased for better context
        chunk_overlap=CHUNK_OVERLAP,  # Increased overlap
    )
    chunks: list[Document] = splitter.split_documents(documents)
    print(f"number of chunks : {len(chunks)}")
    return chunks


# get embeddings for MODEL
def get_embeddings_model() -> HuggingFaceEmbeddings:
    return embeddings_model


# save vectors in chromadb
def save_vectors(chunks: list[Document], embeddings_model: HuggingFaceEmbeddings, persist_dir: str) -> Chroma:
    # clean previous data
    import shutil

    shutil.rmtree(persist_dir, ignore_errors=True)
    print(f"vectors deleted from {persist_dir}")

    vectorstore = Chroma.from_documents(chunks, embeddings_model, persist_directory=persist_dir)
    print(f"vectors saved in {persist_dir}")

    # print sample vector and dimension
    collection = vectorstore._collection
    count = collection.count()
    sample = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimension = len(sample)
    print(f"vector count: {count}")
    print(f"sample vector first 10 values : {sample[:10]}...")
    print(f"dimension:{dimension}")
    return vectorstore


if __name__ == "__main__":
    import os

    docs_dir = "documents"
    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
    txt_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]

    print("reading documents - START")
    pages = []
    for pdf in pdf_files:
        pages.extend(read_pdf(os.path.join(docs_dir, pdf)))

    seed_documents = []
    for txt in txt_files:
        with open(os.path.join(docs_dir, txt), "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                if line.strip():
                    doc = create_document(line)
                    doc.metadata = {"source": os.path.join(docs_dir, txt), "page": 0}
                    seed_documents.append(doc)

    print(f"Loaded {len(pages)} PDF pages and {len(seed_documents)} text documents.")
    print("reading documents - END\n")

    print("loading tokenizer - START")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    print("loading tokenizer - END\n")

    print("splitting documents - START")
    documents = pages + seed_documents
    chunks = split_documents(documents=documents, tokenizer=tokenizer)
    print("splitting documents - END\n")
    print("\n\n\n")

    print("getting embeddings for MODEL - START")
    embeddings_model = get_embeddings_model()
    print("getting embeddings for MODEL - END")
    print("\n\n\n")

    print("saving vectors in chromadb - START")
    vectorstore = save_vectors(chunks=chunks, embeddings_model=embeddings_model, persist_dir=DB_NAME)
    print("saving vectors in chromadb - END")
    print("\n\n\n")
