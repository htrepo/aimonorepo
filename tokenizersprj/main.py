from huggingface_hub import login
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
from IPython import get_ipython

def main():
    # Log in to Hugging Face
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token.startswith("hf_"):
        print("HF key looks good so far")
    else:
        print("HF key is not set - please click the key in the left sidebar")
    login(hf_token, add_to_git_credential=True)

    # nvidia info
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("No GPU detected by PyTorch.")
    except ImportError:
        print("PyTorch not installed, skipping GPU check.")

    # Load Tokenizer
    print("Loading Tokenizer")
    tokenizer_llama = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)
    print(f"Tokenizer: {type(tokenizer_llama)}")
    print(f"Vocab Size: {tokenizer_llama.vocab_size}")
    print(f"Num Special Tokens: {len(tokenizer_llama.special_tokens_map)}")

    # Sample Text
    print("\nTokenizing Sample Text")
    sample_text = "Tokenization is the process of splitting text into smaller units called tokens."

    tokens = tokenizer_llama.tokenize(sample_text)

    print(f"\nOriginal Text: {sample_text}")
    print(f"\nTokens: {tokens}")
    print(f"\nNumber of Tokens: {len(tokens)}")

    # Check Special Tokens
    special_tokens = tokenizer_llama.special_tokens_map
    print("\nSpecial Tokens:")
    for key, value in special_tokens.items():
        print(f"{key}: {value}")

    # Convert to IDs
    print("\nConverting to IDs")
    token_ids = tokenizer_llama.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids}")

    # Decode IDs
    print("\nConverting back to Text")
    decoded_text = tokenizer_llama.decode(token_ids)
    print(f"Decoded Text: {decoded_text}")

    # Verify Match
    print("\nVerification:")
    print(f"Original matches decoded: {sample_text == decoded_text}")

    # Special Case: EOT Token
    print("\nTesting End-of-Sequence Token:")
    print(f"EOT Token: {tokenizer_llama.eos_token}")
    print(f"EOT Token ID: {tokenizer_llama.eos_token_id}")

    # Test with EOT
    test_text = sample_text + tokenizer_llama.eos_token
    test_tokens = tokenizer_llama.tokenize(test_text)
    print(f"\n{test_text}")
    print(f"Tokens with EOT: {test_tokens}")

    # Decode with EOT
    test_token_ids = tokenizer_llama.convert_tokens_to_ids(test_tokens)
    test_decoded = tokenizer_llama.decode(test_token_ids)
    print(f"\nDecoded with EOT: {test_decoded}")
    print(f"Matches original: {test_decoded == sample_text}")

    print("\nDone!")

if __name__ == "__main__":
    main()
