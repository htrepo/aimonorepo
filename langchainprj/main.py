from langchain_community.document_loaders import PyPDFLoader
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from langchain_core.documents import Document
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
MODEL = "ministral-3:14b"


# read pdf and return the text content
def read_pdf(file_path: str) -> str:
    print(f"Loading PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Total pages: {len(pages)}\n")
    # iterate pages and create content str
    content = ""
    for page in pages:
        content += page.page_content
    return content


# encode content and return tokens
def tokens_from_content(content: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokens = tokenizer.encode(content)
    print(f"\nTokens: {tokens[:10]}...\n")
    return tokens


# create Document type from content
def create_document(content: str) -> Document:
    return Document(page_content=content)


# split documents into chunks
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks: list[Document] = splitter.split_documents(documents)
    print(f"number of chunks : {len(chunks)}")
    return chunks


# create embeddings for MODEL
def create_embeddings() -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# save vectors in chromadb
def save_vectors(
    chunks: list[Document], embeddings_model: HuggingFaceEmbeddings, persist_dir: str
) -> Chroma:
    vectorstore = Chroma.from_documents(
        chunks, embeddings_model, persist_directory=persist_dir
    )
    # clean previous data
    vectorstore.delete_collection()
    print("vectors deleted from {persist_dir}")
    vectorstore = Chroma.from_documents(
        chunks, embeddings_model, persist_directory=persist_dir
    )
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
    pdf_path = "mlops-and-trustworthy-ai-for-data-leaders.pdf"
    print(f"reading pdf:{pdf_path} - START")
    content = read_pdf(pdf_path)
    print(f"reading pdf:{pdf_path} - END")

    print("tokenizing content from pdf - START")
    tokens = tokens_from_content(content=content)
    print("tokenizing content from pdf - END")

    print("creating document from content - START")
    doc = create_document(content=content)
    print("creating document from content - END")

    print("splitting documents - START")
    chunks = split_documents(documents=[doc])
    print("splitting documents - END")

    print("creating embeddings for MODEL - START")
    embeddings_model = create_embeddings()
    print("creating embeddings for MODEL - END")

    print("saving vectors in chromadb - START")
    vectorstore = save_vectors(
        chunks=chunks, embeddings_model=embeddings_model, persist_dir="chroma_db"
    )
    print("saving vectors in chromadb - END")

    print("visualizing vectors - START")
    visualize_vectors(vectorstore=vectorstore)
    print("visualizing vectors - END")

    print("\n\n\n")
