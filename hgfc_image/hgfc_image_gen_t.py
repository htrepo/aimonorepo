import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import time
import os

def generate_image_from_source(image_path: str, prompt: str, output_path: str = "output.png"):
    print(f"Loading model using AutoPipelineForImage2Image...")
    start_time = time.time()
    
    model_id = "Lykon/dreamshaper-8"    
    
    # AutoPipelineForImage2Image automatically selects the correct pipeline
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    # Load and prepare the initial image
    print(f"Loading input image: {image_path}...")
    init_image = load_image(image_path).convert("RGB")
    init_image = init_image.resize((512, 512))

    print(f"Generating image for prompt: '{prompt}'")
    start_time = time.time()
    
    # Generate the image
    image = pipe(
        prompt=prompt, 
        image=init_image, 
        strength=0.75, 
        num_inference_steps=30
    ).images[0]
    
    print(f"Image generated in {time.time() - start_time:.2f} seconds.")
    
    # Save the image
    image.save(output_path)
    print(f"Image saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Image generation will be very slow on CPU.")
    else:
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")

    input_photo = "tej.jpg"
    
    sample_prompt = """
    create cartoon from supplied image. make it look super beautiful, anime style, pixar style.
    ensure features match the supplied image.
    """
    
    if os.path.exists(input_photo):
        generate_image_from_source(input_photo, sample_prompt, "tej_bengaluru_car_gen.png")
    else:
        print(f"Error: {input_photo} not found. Please ensure the file exists.")

