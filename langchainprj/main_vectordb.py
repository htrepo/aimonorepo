import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

load_dotenv()

# The tokenizer used for estimating chunk limits
TOKENIZER_MODEL = "mistralai/Mistral-7B-v0.1"
DB_NAME = "vectors_db"


# read pdf and return the list of documents (one per page)
def read_pdf(file_path: str) -> list[Document]:
    print(f"Loading PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Total pages loaded: {len(pages)}")
    return pages


def create_documents_from_list(contents: list[str]) -> list[Document]:
    documents = []
    for content in contents:
        documents.append(create_document(content))
    return documents


# encode content and return tokens
def tokens_from_content(content: str, tokenizer) -> list[int]:
    tokens = tokenizer.encode(content)
    print(f"\nTokens: {tokens[:10]}...\n")
    return tokens


# create Document type from content
def create_document(content: str) -> Document:
    return Document(page_content=content)


# split documents into chunks
def split_documents(documents: list[Document], tokenizer) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n\n", "\n", ".", " "],
        chunk_size=500,  # Increased for better context
        chunk_overlap=100,  # Increased overlap
    )
    chunks: list[Document] = splitter.split_documents(documents)
    print(f"number of chunks : {len(chunks)}")
    return chunks


# create embeddings for MODEL
def create_embeddings() -> HuggingFaceEmbeddings:
    # fast, lightweight, good general-purpose (384 dims)
    embeddings1 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # higher accuracy, slower (768 dims)
    embeddings2 = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    # print dimension
    sample_embedding = embeddings1.embed_query("test")
    print(f"dimension all-MiniLM-L6-v2: {len(sample_embedding)}")
    sample_embedding = embeddings2.embed_query("test")
    print(f"dimension all-mpnet-base-v2: {len(sample_embedding)}")
    # print few embeddings from embeddings1
    sample_embeddings: list[list[float]] = embeddings1.embed_documents(["test"])
    print(f"sample embeddings all-MiniLM-L6-v2: {sample_embeddings[0][:10]}...")
    # print few embeddings from embeddings2
    sample_embeddings = embeddings2.embed_documents(["test"])
    print(f"sample embeddings all-mpnet-base-v2: {sample_embeddings[0][:10]}...")
    # return embedding. manually switch
    return embeddings2


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


# visualize vectors using plotly
def visualize_vectors(vectorstore: Chroma):
    collection = vectorstore._collection
    coll_data = collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.array(coll_data["embeddings"])
    # print useful information
    metadata = coll_data["metadatas"]
    documents = coll_data["documents"]
    print(f"metadata : {metadata[0]}")
    print(f"document : {documents[0]}")
    # Reduce dimensionality using t-SNE - simple parameters
    tsne = TSNE(n_components=3, random_state=42, perplexity=10)
    vectors_3d = tsne.fit_transform(vectors)
    fig = go.Figure(
        data=go.Scatter3d(
            x=vectors_3d[:, 0],
            y=vectors_3d[:, 1],
            z=vectors_3d[:, 2],
            mode="markers",
            marker=dict(
                colorscale="Viridis",  # or 'Plotly3', 'Hot', 'Jet', etc.
                size=5,
                line=dict(width=1, color="Black"),
            ),
            text=documents,  # hover text
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="3D t-SNE Visualization of Document Embeddings",
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3",
        ),
    )
    fig.show()


if __name__ == "__main__":
    pdf_path_1 = "mlops-and-trustworthy-ai-for-data-leaders.pdf"
    pdf_path_2 = "LeadershipintheAIEra-Navigatingandshapingthefutureoforganizationalguidance.pdf"
    print("reading pdf - START")
    pages_1 = read_pdf(pdf_path_1)
    pages_2 = read_pdf(pdf_path_2)
    pages = pages_1 + pages_2
    print("reading pdf - END\n")

    seed_data = []
    with open("maryann_seed_data.txt", "r") as f:
        seed_data = f.read().splitlines()
    seed_documents = create_documents_from_list(seed_data)
    print("seed data: ", seed_data)
    print("seed data END\n")

    print("loading tokenizer - START")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    print("loading tokenizer - END\n")

    print("splitting documents - START")
    documents = pages + seed_documents
    chunks = split_documents(documents=documents, tokenizer=tokenizer)
    print("splitting documents - END\n")
    print("\n\n\n")

    print("creating embeddings for MODEL - START")
    embeddings_model = create_embeddings()
    print("creating embeddings for MODEL - END")
    print("\n\n\n")

    print("saving vectors in chromadb - START")
    vectorstore = save_vectors(chunks=chunks, embeddings_model=embeddings_model, persist_dir=DB_NAME)
    print("saving vectors in chromadb - END")
    print("\n\n\n")

    # print("visualizing vectors - START")
    # visualize_vectors(vectorstore=vectorstore)
    # print("visualizing vectors - END")
