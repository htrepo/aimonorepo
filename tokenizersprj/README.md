# LLM Tokenizer & Inference Project

This project explores production-grade implementations for calling Large Language Models (LLMs) locally using Hugging Face's `transformers` library.

## Key Learnings

### 1. Model Selection & Resource Management
- **Size vs. Performance**: Choosing a model like `Qwen2.5-1.5B-Instruct` (~3GB) over `Mistral-7B-Instruct` (~14GB) significantly reduces download time and VRAM usage while maintaining high inference speed.
- **VRAM Constraints**: 7B parameter models in `float16` require ~14-15GB of VRAM. Smaller models (1.5B - 3B) are better suited for consumer GPUs (like RTX 3060 12GB).

### 2. Efficient Loading
- **Device Mapping**: Using `device_map="auto"` allows `accelerate` to handle weight distribution across GPU and CPU automatically.
- **Dtype Selection**: Loading models in `torch.float16` or `bfloat16` is standard for modern GPUs to halve memory usage compared to the default `float32`.

### 3. Proper Input Formatting
- **Chat Templates**: Always use `tokenizer.apply_chat_template()` with `add_generation_prompt=True` to ensure the model receives inputs in the format it was trained on (e.g., ChatML for Qwen).
- **Special Tokens**: Explicitly setting `pad_token` to `eos_token` if not defined is crucial for batching and generation consistency.

### 4. Inference Techniques
- **Streaming vs. Non-Streaming**:
    - **Streaming** (`TextStreamer`) provides a better UX for interactive applications by showing tokens as they generate.
    - **Non-Streaming** is simpler for scripts where the full response is needed at once for post-processing or logging.
- **Generation Parameters**: Tuning `temperature`, `top_p`, and `max_new_tokens` is essential for controlling the creativity and length of the response.

## Scripts
- `prod_grade_llm_call_noquant.py`: A clean, non-quantized implementation focusing on speed and simplicity using the Qwen 1.5B model.
