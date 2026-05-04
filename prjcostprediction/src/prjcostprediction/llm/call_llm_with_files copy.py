from pathlib import Path

from google import genai
from google.genai.types import Batch, UploadedFile

from prjcostprediction import config

MODEL = "gemini/gemini-3.1-flash-lite-preview"
google_ai_client = genai.Client()


def call_llm():
    """Calls the LLM with the files."""
    files: list[Path] = config.MODEL_INPUT_JSONL_FILES_BASE_DIR.glob("*.jsonl")
    batch_ids = []
    # list file info
    for file in files:
        print(f"File name: {file.name}")
        print(f"File size: {file.stat().st_size / 1024 / 1024:.2f} MB")
        # Upload the file
        with open(file, "rb") as f:
            uploaded_file: UploadedFile = google_ai_client.files.create(file=f, purpose="batch")

        # Create a batch request
        batch: Batch = google_ai_client.batches.create(
            model=MODEL,
            input_file=uploaded_file.name,
            config={
                "temperature": 0.0,
            },
        )
        print("Batch job created:", batch.id)
        batch_ids.append(batch.id)

    # Wait for the batch job to complete and get the results.
    output_file_number = 1
    for batch_id in batch_ids:
        print(f"Waiting for batch job {batch_id} to complete...")
        batch: Batch = google_ai_client.batches.wait(batch_id)
        print("Batch job completed.")
        results: Batch = google_ai_client.batches.get_results(batch)

        print("Batch job results:", results.output_file)
        # The results object contains the generated text for each input.
        if results.output_file:
            output_file = config.MODEL_OUTPUT_JSONL_FILES_BASE_DIR / f"batchoutput_{output_file_number}.jsonl"
            with open(output_file, "w") as f:
                for response in results.output_file.read_lines():
                    f.write(response.text + "\n")
        output_file_number += 1


def main():
    call_llm()


if __name__ == "__main__":
    main()
