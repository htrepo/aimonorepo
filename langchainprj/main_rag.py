from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

MODEL = "ministral-3:14b"
DB_NAME = "vectors_db"
load_dotenv()


if __name__ == "__main__":
    embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=MODEL, temperature=0)
    # invoke
    question = "How is AI being used at Penn Medicine? Provide answer in three bullet points."
    relevant_docs = retriever.invoke(question)
    # print relevant docs
    for doc in relevant_docs:
        print("-" * 40, "doc page content", "-" * 40)
        print(doc.page_content[:50])
    print("-" * 40)

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
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context="\n\n".join([doc.page_content for doc in relevant_docs])
    )
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    print("response with context:")
    print(response.content)
    
    # add random unrelated question
    question_unrelated = "How did Russia-Ukraine war start?"
    response_unrelated = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question_unrelated)])
    print("response unrelated with context:")
    print(response_unrelated.content)
    
    # add code to show llm response with context
    response_without_context = llm.invoke([HumanMessage(content=question)])
    print("response without context:")
    print(response_without_context.content)

   
