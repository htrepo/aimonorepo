from langchain_chroma import Chroma

from _proj_embedding import embeddings_model

# Centralized vector database configuration
DB_NAME = "vectors_db"


def get_vectorstore():
    """Returns the Chroma vectorstore instance."""
    return Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)


def get_retriever(k=10):
    """Returns a retriever from the vectorstore with specified k search parameter."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
