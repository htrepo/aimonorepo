import torch
from diffusers import AutoPipelineForText2Image
import time
import os

def generate_image(prompt: str, output_path: str = "output.png"):
    print(f"Loading model using AutoPipelineForText2Image... (this may take a bit on first run)")
    start_time = time.time()
    
    # Load the model from Hugging Face using the generic AutoPipeline
    # Best Quality: Lykon/dreamshaper-8 (Drop-in replacement for v1.5).
    # Fastest Speed: stabilityai/sdxl-turbo (Generates in 1-4 steps, nearly instant).
    # Best Realism: SG_161222/RealVisXL_V4.0_Lightning.

    model_id = "Lykon/dreamshaper-8"    
    
    # AutoPipelineForText2Image will automatically select the correct pipeline class (StableDiffusionPipeline in this case)
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # Move pipeline to GPU
    pipe = pipe.to("cuda")
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    print(f"Generating image for prompt: '{prompt}'")
    start_time = time.time()
    
    # Generate the image
    image = pipe(prompt).images[0]
    
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

    sample_prompt = """
    a man with his black hyundai verna car.
    location - bengaluru.
    weather - hot and sweaty.
    time - afternoon.
    """
    generate_image(sample_prompt, "man_in_car_driver_window.png")
