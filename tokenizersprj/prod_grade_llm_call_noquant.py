import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model name from Hugging Face
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def main():

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Ensure pad token is set to eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
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

    print("\n--- Generating Response ---\n")
    # Generate the output
    outputs = model.generate(
        **inputs, 
        max_new_tokens=2000, 
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Decode and print the generated text (excluding the prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(response)
    print("\n---------------------------\n")

if __name__ == "__main__":
    main()


