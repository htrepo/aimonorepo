import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

# Model name from Hugging Face
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

def main():
    # Configure quantization for running 7B models locally on consumer GPUs
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Ensure pad token is set to eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        quantization_config=quant_config,
        torch_dtype=torch.float16
    )

    # Prepare the messages for the chat template
    messages = [
        {"role": "system", "content": "You are a helpful and knowledgeable AI assistant expert in Large Language Models."},
        {"role": "user", "content": "key skills to crack LLM interivew in bullet points."}
    ]

    print("Formatting inputs using chat template...")
    # Apply chat template and move to CUDA
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_dict=True,
        return_tensors="pt", 
        add_generation_prompt=True
    ).to("cuda")

    # Set up the streamer to yield text tokens as they are generated
    # Using skip_prompt=True so we only see the newly generated response
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\n--- Generating Response ---\n")
    # Generate the output iteratively passing through the streamer
    outputs = model.generate(
        **inputs, 
        max_new_tokens=2000, 
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    print("\n---------------------------\n")

if __name__ == "__main__":
    main()
