import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# MODEL = "gpt-4o-mini"
MODEL = "gemini-2.5-flash-lite"
DB_NAME = "vectors_db"
load_dotenv()

# Initialize models and vectorstore
embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)


def process_interaction(message, history):
    # Convert Gradio history to LangChain messages
    chat_history = []
    for msg in history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    # 0. Contextualize the question if history exists
    standalone_query = message
    if chat_history:
        # Format chat history for the prompt
        history_str = ""
        for msg in chat_history:
            prefix = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_str += f"{prefix}: {msg.content}\n"

        contextualize_prompt = f"""
        Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the conversation history.
        
        CRITICAL RULES:
        1. DO NOT answer the question.
        2. DO NOT provide any preamble or explanation.
        3. ONLY output the rephrased question.
        4. If the question is already standalone, return it exactly as is.

        Chat History:
        {history_str}

        Follow-up Question: {message}
        Standalone Question:"""

        print("\n\n--- Contextualization Task ---")
        print(f"History context length: {len(history_str)} chars")
        standalone_query = llm.invoke([HumanMessage(content=contextualize_prompt)]).content.strip()
        print(f"Original: {message}")
        print(f"Standalone: {standalone_query}")
        print("------------------------------\n")

    # 1. Retrieve relevant documents using multiple perspectives (Multi-Query)
    # We use the original query + 2 entity-focused expansions to bridge semantic gaps
    print(f"\n--- Multi-Query Expansion ---")
    expansion_prompt = f"""
    Given the user's question, generate 2 additional search queries to find context.
    Focus on finding the identity/role of any people mentioned.
    
    Original Question: {standalone_query}
    
    Output ONLY the 2 queries, one per line.
    """

    expansion_response = llm.invoke([HumanMessage(content=expansion_prompt)]).content.strip()
    # Ensure the original query is first to prioritize its results
    queries = [standalone_query] + [q.strip() for q in expansion_response.split("\n") if q.strip()]

    print(f"Total queries for retrieval: {queries}")

    all_docs = []
    for q in queries:
        # Retrieve 10 docs for each query to ensure we catch distant but relevant info
        all_docs.extend(retriever.invoke(q))

    # Deduplicate docs by content while maintaining priority order
    seen_contents = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)

    # Provide a focused context window (10 unique chunks)
    relevant_docs = unique_docs[:10]

    # Format context clearly for the LLM
    context_parts = []
    for i, doc in enumerate(relevant_docs):
        context_parts.append(f"Content from Document {i + 1}:\n{doc.page_content}")

    context = "\n\n".join(context_parts)
    print(f"Retrieved {len(relevant_docs)} unique chunks.")
    print("------------------------------\n")

    # 2. Create the final prompt for the LLM
    # Generic RAG prompt: context first, then question, then a simple fallback rule.
    # No person-specific rules needed - the LLM synthesises information from the context naturally.
    final_prompt = (
        "Be concise. Answer in 1-2 sentences maximum.\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n"
        f"Using only the context above, answer the question: {message}\n"
        "If the answer is not in the context, say 'I don't know.'"
    )

    # Debug: Print the first chunk to ensure salary info is being passed
    if relevant_docs:
        print(f"DEBUG: First context chunk: {relevant_docs[0].page_content[:100]}...")

    # 3. Get response from LLM
    response = llm.invoke([HumanMessage(content=final_prompt)])

    # Update history for Gradio
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response.content})

    return history, "", context


# Custom CSS for premium look and feel
custom_css = """
footer {visibility: hidden}
.context-box {
    border-radius: 10px;
    background-color: #1a1b1e;
    border: 1px solid #2d2e32;
}
"""

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🛡️ MLOps & Trustworthy AI Assistant
        *Expert guidance for data leaders on operationalizing AI with trust.*
        """
    )

    with gr.Row():
        # Left Column: Chat Interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=AI"),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about MLOps, Model Governance, Monitoring...",
                    label="Your Question",
                    scale=9,
                    container=False,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "What is MLOps?",
                    "How to ensure AI is trustworthy?",
                    "What are the key pillars of data leadership?",
                ],
                inputs=msg,
            )

        # Right Column: Context Panel
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Retrieved Context")
            context_display = gr.Textbox(
                label="Supporting Documents",
                placeholder="Retrieved context will appear here...",
                lines=25,
                interactive=False,
                elem_classes="context-box",
            )

    # Event handlers
    submit_btn.click(process_interaction, inputs=[msg, chatbot], outputs=[chatbot, msg, context_display])
    msg.submit(process_interaction, inputs=[msg, chatbot], outputs=[chatbot, msg, context_display])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"), css=custom_css)
