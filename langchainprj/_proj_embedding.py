from langchain_huggingface import HuggingFaceEmbeddings

# Centralized embedding model to ensure consistency across the project
# model_name="all-mpnet-base-v2" is used for higher accuracy (768 dims)
embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
