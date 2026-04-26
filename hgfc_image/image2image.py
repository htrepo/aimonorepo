import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import time
import os

def generate_from_photo(image_path: str, prompt: str, strength=0.6, output_path: str = "tej_in_car.png"):
    print(f"--- Image-to-Image Processing ---")
    start_time = time.time()
    
    # We use DreamShaper 8 (based on SD 1.5) for high quality and compatibility with 6GB VRAM
    model_id = "Lykon/dreamshaper-8"
    
    print(f"Loading model: {model_id}...")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    
    # Load and prepare the initial image
    print(f"Loading input image: {image_path}...")
    init_image = load_image(image_path).convert("RGB")
    
    # Resize to 512x512 as SD 1.5 works best at this resolution
    init_image = init_image.resize((512, 512))

    # Detailed negative prompt to ensure quality
    negative_prompt = "blurry, low quality, distorted, extra limbs, bad anatomy, deformed, cartoon, anime, drawing"

    print(f"Generating image with prompt: '{prompt}'")
    print(f"Strength: {strength} (higher = more AI, lower = more like original)")
    
    gen_start = time.time()
    
    # Generate the image
    # Note: height and width are taken from the init_image in Img2Img
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=40,
        guidance_scale=7.5,
    ).images[0]
    
    print(f"Image generated in {time.time() - gen_start:.2f} seconds.")
    
    # Save output
    image.save(output_path)
    print(f"Result saved to: {os.path.abspath(output_path)}")
    print(f"Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA not found. This will be very slow on CPU.")
    
    # Input file from your directory
    input_photo = "tej.jpg"
    
    # Custom prompt to put her in the Bengaluru car scene
    # We describe the person to help the AI keep the likeness
    target_prompt = (
        "a woman with long dark hair and dark eyes sitting inside a black hyundai verna car, "
        "hand on chin, looking out of the driver window, Bengaluru street background, "
        "sunny hot afternoon, cinematic lighting, highly detailed, photorealistic, 8k"
    )
    
    if os.path.exists(input_photo):
        generate_from_photo(
            image_path=input_photo, 
            prompt=target_prompt, 
            strength=0.9, # Adjusted strength for a good balance
            output_path="tej_bengaluru_car.png"
        )
    else:
        print(f"File not found: {input_photo}")
