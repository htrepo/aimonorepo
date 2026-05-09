import gradio as gr
from PIL import Image


def greet(name, image_pixels):
    print(type(image_pixels))
    print(image_pixels.shape)

    img = Image.fromarray(image_pixels)
    # img.save("uploaded.jpg")
    print(img)

    return "Hello " + name + "!"


demo = gr.Interface(
    fn=greet, inputs=[gr.Text(label="Please enter your name:"), gr.Image()], outputs=gr.Text(label="Output:")
)
demo.launch()
