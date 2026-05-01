import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from _proj_rag import run_rag_pipeline

# MODEL = "gpt-4o-mini"
MODEL = "gemini-2.5-flash-lite"
load_dotenv()

# Initialize LLM for contextualization
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
        Given the following conversation history and a follow-up question, 
        rephrase the follow-up question to be a standalone question that can be understood
        without the conversation history.
        
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

    # 1. Run the RAG pipeline
    print(f"\n--- Running RAG Pipeline for: {standalone_query} ---")
    answer, relevant_docs = run_rag_pipeline(standalone_query)
    print(f"Retrieved {len(relevant_docs)} unique chunks.")
    print(f"Generated Answer: {answer}")
    print("------------------------------\n")

    # Update history for Gradio
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    # Format context for display
    context_display = "\n\n".join([f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])

    return history, "", context_display


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
