import gradio as gr


def greet(name, message):
    return "Hello " + name + "! " + message


demo = gr.Interface(
    fn=greet,
    inputs=[gr.Text(label="Please enter your name:"), gr.TextArea(label="Please enter your message:")],
    outputs=gr.Text(label="Output:"),
)
demo.launch()
