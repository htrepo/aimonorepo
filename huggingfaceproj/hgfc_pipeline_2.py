from transformers import pipeline

def main():
    # Initialize the text-generation pipeline with a small model that easily fits on an RTX 3060 (6GB VRAM)
    # The 'summarization' task is deprecated in newer versions of transformers in favor of 'text-generation'
    #model_name1 = "Qwen/Qwen2.5-0.5B-Instruct"
    model_name2 = "mistralai/Ministral-3-14B-Instruct-2512"
    print(f"Loading text-generation pipeline with {model_name2} on CUDA...")
    text_generator = pipeline("text-generation", model=model_name2, device="cuda")

    # Sample text to summarize
    text = """
    if there is one thing i want you to remember about using hugging face pipelines, it is ..."
    """
    # Run the summarization using the instruction-tuned model
    print("completing the text...")
    prompt = f"Complete the following text:\n\n{text.strip()}\n\nCompleted:"
    
    # We use return_full_text=False so it only returns the newly generated text
    summary = text_generator(prompt, max_new_tokens=130, do_sample=False, return_full_text=False)

    print("\nCompleted Text:")
    print("-" * 50)
    print(summary[0]['generated_text'].strip())

if __name__ == "__main__":
    main()
