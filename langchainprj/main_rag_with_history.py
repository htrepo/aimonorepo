from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

MODEL = "llama3.2:latest"
DB_NAME = "vectors_db"
load_dotenv()


if __name__ == "__main__":
    embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=MODEL, temperature=0)

    chat_history = []  # Maintain conversation history

    # Prompt to help the LLM contextualize the question for retrieval
    CONTEXTUALIZE_SYSTEM_PROMPT = """
    Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """

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

    print("-" * 50)
    print("Welcome! Chat about MLOps and Trustworthy AI. (Type 'exit' to stop)")
    print("-" * 50)

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not query.strip():
            continue

        # 0. Contextualize the question if history exists
        # This makes sure the search query is descriptive (e.g., changing "tell me more" to "tell me more about MLOps")
        standalone_query = query
        if chat_history:
            contextualize_messages = [
                SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT)
            ] + chat_history + [HumanMessage(content=query)]
            standalone_query = llm.invoke(contextualize_messages).content

        # 1. Retrieve relevant documents for the standalone question
        relevant_docs = retriever.invoke(standalone_query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 2. Update the system prompt with retrieved context
        system_message = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context))

        # 3. Create the message list for the LLM
        messages = [system_message] + chat_history + [HumanMessage(content=query)]

        # 4. Get response from LLM
        response = llm.invoke(messages)
        ai_response = response.content

        # 5. Print and update history
        print(f"\nAI: {ai_response}")
        
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=ai_response))
