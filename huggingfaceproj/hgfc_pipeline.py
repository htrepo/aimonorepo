from transformers import pipeline

def main():
    # Initialize the text-generation pipeline with a small model that easily fits on an RTX 3060 (6GB VRAM)
    # The 'summarization' task is deprecated in newer versions of transformers in favor of 'text-generation'
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading text-generation pipeline with {model_name} on CUDA...")
    summarizer = pipeline("text-generation", model=model_name, device="cuda")

    # Sample text to summarize
    text = """
    New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. 
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband. 
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, 
    sometimes only within two weeks of each other. In 2010, she married once more, this time in the Bronx. 
    In an application for a marriage license, she stated it was her "first and only" marriage. 
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," 
    referring to her false statements on the 2010 marriage license application, according to court documents. 
    Prosecutors said the marriages were part of an immigration scam. 
    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, 
    who declined to comment further. After leaving court, Barrientos was arrested and charged with theft of service 
    and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective 
    Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her 
    marriages occurring between 1999 and 2002. All occurred either in Westchester County, Long Island, New Jersey 
    or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, 
    prosecutors say.
    """

    # print("Original Text:")
    # print("-" * 50)
    # print(text.strip())
    # print("\n")

    # Run the summarization using the instruction-tuned model
    print("Summarizing text...")
    prompt = f"Summarize the following text in a few short sentences:\n\n{text.strip()}\n\nSummary:"
    
    # We use return_full_text=False so it only returns the newly generated text
    summary = summarizer(prompt, max_new_tokens=130, do_sample=False, return_full_text=False)

    print("\nSummary:")
    print("-" * 50)
    print(summary[0]['generated_text'].strip())

if __name__ == "__main__":
    main()
