import torch
from transformers import BarkModel, BarkProcessor
import scipy.io.wavfile as wavfile
import os
import numpy as np
import warnings
from transformers import logging

# Suppress warnings and background thread chatter
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def main():
    print("--- Hugging Face Text-to-Audio (Bark High Quality) ---")
    
    # 1. Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Initialize Processor and Model
    print("Loading full model (suno/bark)... This might take a moment.")
    
    # Bark processor handles text and voice presets
    processor = BarkProcessor.from_pretrained("suno/bark")
    
    # Load model in float16 to fit in 6GB VRAM
    model = BarkModel.from_pretrained(
        "suno/bark", 
        torch_dtype=torch.float16,
        use_safetensors=False
    ).to(device)

    # 3. Define the text and voice preset
    # Note: User updated this text in the editor
    text = "You are lovely human being and I am so glad that i met you !"
    voice_preset = "v2/en_speaker_9"
    
    print(f"Synthesizing: \"{text}\"")
    print(f"Voice Preset: {voice_preset}")
    
    # 4. Process inputs
    inputs = processor(text, voice_preset=voice_preset).to(device)

    # 5. Generate audio
    print("Generating audio... (Full Bark model)")
    with torch.no_grad():
        audio_array = model.generate(**inputs, do_sample=True)
    
    # 6. Post-process and Save
    audio_data = audio_array.cpu().numpy().squeeze()
    sampling_rate = 24000
    
    # Normalize
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV
    output_filename = "output.wav"
    wavfile.write(output_filename, rate=sampling_rate, data=audio_data)
    
    print(f"\nSuccess! Audio saved to: {os.path.abspath(output_filename)}")
    print("The WAV file is high-quality and fully compatible with Android phones.")

if __name__ == "__main__":
    main()
