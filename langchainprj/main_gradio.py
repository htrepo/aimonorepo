import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

MODEL = "mistral:7b"
DB_NAME = "vectors_db"
load_dotenv()

# Initialize models and vectorstore
embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
llm = ChatOllama(model=MODEL, temperature=0)

# Prompts
CONTEXTUALIZE_SYSTEM_PROMPT = """
Combine the chat history and the latest question into a single standalone question.
If the question is already standalone, output it exactly as is.
Output ONLY the question text. Do not provide any explanation, preamble, or refusal.

Examples:
- History: [User: What is Watson?] | Question: Is it good? -> Is IBM Watson a good product?
- History: [User: Tell me about MLOps] | Question: who is MaryAnn? -> Who is MaryAnn?
"""

SYSTEM_PROMPT_TEMPLATE = """
You are an expert in AI and Machine Learning. 
You are chatting with a user about MLOps and trustworthy AI for data leaders.
The following context is extracted from a document on MLOps and Trustworthy AI.
Use the context to answer the user's question. If the question is about a person or organization mentioned in the context, provide the details available.
If the answer is not in the context, say "I don't know."

Context:
{context}
"""

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
        contextualize_messages = [
            SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT)
        ] + chat_history + [HumanMessage(content=message)]
        print("\n\nContextualizing question: ", contextualize_messages, "\n")
        standalone_query = llm.invoke(contextualize_messages).content
        print("Standalone query: ", standalone_query, "\n\n")

    # 1. Retrieve relevant documents for the standalone question
    relevant_docs = retriever.invoke(standalone_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 2. Update the system prompt with retrieved context
    system_message = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context))

    # 3. Create the message list for the LLM
    messages = [system_message] + chat_history + [HumanMessage(content=message)]

    # 4. Get response from LLM
    response = llm.invoke(messages)
    
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
                    container=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            gr.Examples(
                examples=["What is MLOps?", "How to ensure AI is trustworthy?", "What are the key pillars of data leadership?"],
                inputs=msg
            )

        # Right Column: Context Panel
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Retrieved Context")
            context_display = gr.Textbox(
                label="Supporting Documents",
                placeholder="Retrieved context will appear here...",
                lines=25,
                interactive=False,
                elem_classes="context-box"
            )

    # Event handlers
    submit_btn.click(
        process_interaction, 
        inputs=[msg, chatbot], 
        outputs=[chatbot, msg, context_display]
    )
    msg.submit(
        process_interaction, 
        inputs=[msg, chatbot], 
        outputs=[chatbot, msg, context_display]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"), css=custom_css)
