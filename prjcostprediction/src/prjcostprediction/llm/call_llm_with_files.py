import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from prjcostprediction import config

MODEL = "gemini-2.5-flash"
POLL_SECONDS = 30
JOBS_MANIFEST = config.MODEL_OUTPUT_JSONL_FILES_BASE_DIR / "batch_jobs.json"
SUCCESS_STATES = {"ACTIVE", "JOB_STATE_SUCCEEDED"}
TERMINAL_STATES = SUCCESS_STATES | {"FAILED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


# .venv\Scripts\python.exe src\prjcostprediction\llm\call_llm_with_files.py --status
# .venv\Scripts\python.exe src\prjcostprediction\llm\call_llm_with_files.py --download
# .venv\Scripts\python.exe src\prjcostprediction\llm\call_llm_with_files.py --wait


def _state_name(state: Any) -> str:
    """Return a comparable SDK state name across enum/string API responses."""
    return getattr(state, "name", None) or getattr(state, "value", None) or str(state)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def _openai_line_to_gemini_line(line: dict[str, Any]) -> dict[str, Any]:
    """Convert this repo's OpenAI-style batch JSONL line to Gemini Batch API JSONL."""
    body = line.get("body")
    if not isinstance(body, dict) or "messages" not in body:
        raise ValueError("expected either Gemini batch JSONL or OpenAI-style JSONL with body.messages")

    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in body["messages"]:
        role = message.get("role")
        text = _message_content_to_text(message.get("content", ""))
        if role == "system":
            system_parts.append(text)
        elif text:
            user_parts.append(text)

    if not user_parts:
        raise ValueError("OpenAI-style batch line has no user content to send to Gemini")

    request: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "\n\n".join(user_parts)}],
            }
        ]
    }
    if system_parts:
        request["system_instruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    return {"key": str(line.get("custom_id", "")), "request": request}


def _prepare_batch_file(input_path: Path, output_dir: Path) -> Path:
    """Create a Gemini-compatible JSONL file from either Gemini or OpenAI batch JSONL."""
    output_path = output_dir / input_path.name
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, raw_line in enumerate(source, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            parsed_line = json.loads(raw_line)
            if "key" in parsed_line and "request" in parsed_line:
                gemini_line = parsed_line
            else:
                try:
                    gemini_line = _openai_line_to_gemini_line(parsed_line)
                except ValueError as exc:
                    raise ValueError(f"{input_path}:{line_number}: {exc}") from exc

            target.write(json.dumps(gemini_line) + "\n")

    return output_path


def _load_manifest() -> list[dict[str, Any]]:
    if not JOBS_MANIFEST.exists():
        return []
    return json.loads(JOBS_MANIFEST.read_text(encoding="utf-8"))


def _save_manifest(jobs: list[dict[str, Any]]) -> None:
    config.MODEL_OUTPUT_JSONL_FILES_BASE_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_MANIFEST.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def _jobs_from_ids(job_ids: list[str]) -> list[dict[str, Any]]:
    return [{"batch_id": job_id, "input_file": None} for job_id in job_ids]


def submit_batches() -> list[dict[str, Any]]:
    """Upload JSONL files and create Gemini batch jobs without waiting for completion."""
    files: list[Path] = list(config.MODEL_INPUT_JSONL_FILES_BASE_DIR.glob("*.jsonl"))
    if not files:
        print(f"No JSONL files found in {config.MODEL_INPUT_JSONL_FILES_BASE_DIR}")
        return []

    google_ai_client = genai.Client()
    jobs: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="gemini_batch_") as temp_dir_name:
        prepared_dir = Path(temp_dir_name)

        # Upload and create batches
        for file in files:
            prepared_file = _prepare_batch_file(file, prepared_dir)
            print(f"File name: {file.name}")
            print(f"File size: {file.stat().st_size / 1024 / 1024:.2f} MB")

            uploaded_file = google_ai_client.files.upload(
                file=prepared_file,
                config=types.UploadFileConfig(display_name=file.stem, mime_type="jsonl"),
            )

            batch = google_ai_client.batches.create(
                model=MODEL,
                src=uploaded_file.name,
                config={"display_name": file.stem},
            )
            if not batch.name:
                raise RuntimeError(f"Batch job was created without a name for {file}")

            print(f"Batch job created: {batch.name}")
            jobs.append(
                {
                    "batch_id": batch.name,
                    "input_file": str(file),
                    "state": _state_name(batch.state),
                    "output_file": None,
                }
            )

    _save_manifest(jobs)
    print(f"Saved batch job manifest to: {JOBS_MANIFEST}")
    print("Run this script with --status or --download later to collect results.")
    return jobs


def _download_batch_output(
    google_ai_client: genai.Client,
    batch: types.BatchJob,
    output_file_number: int,
) -> Path | None:
    if _state_name(batch.state) not in SUCCESS_STATES:
        return None

    print("Batch job results info:", batch.output_info or batch.dest)
    output_file_name: str | None = batch.dest.file_name if batch.dest is not None else None
    if not output_file_name:
        print(
            f"No downloadable output file found for batch {output_file_number}. "
            "Check your Google Cloud Console or the output_info above."
        )
        return None

    print(f"Downloading output file '{output_file_name}' for batch {output_file_number}...")
    output_bytes: bytes = google_ai_client.files.download(file=output_file_name)

    config.MODEL_OUTPUT_JSONL_FILES_BASE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.MODEL_OUTPUT_JSONL_FILES_BASE_DIR / f"batch_output_{output_file_number}.jsonl"
    output_path.write_bytes(output_bytes)
    print(f"Saved batch output to: {output_path}")
    return output_path


def check_batches(
    jobs: list[dict[str, Any]],
    *,
    download: bool = False,
    wait: bool = False,
) -> list[dict[str, Any]]:
    """Check saved batch jobs and optionally wait for/download completed results."""
    if not jobs:
        print(f"No batch jobs found. Submit first or pass --job-id. Manifest path: {JOBS_MANIFEST}")
        return []

    google_ai_client = genai.Client()
    updated_jobs: list[dict[str, Any]] = []
    for output_file_number, job in enumerate(jobs, start=1):
        batch_id = job["batch_id"]
        print(f"Checking batch job {batch_id}...")

        batch = google_ai_client.batches.get(name=batch_id)
        while wait and _state_name(batch.state) not in TERMINAL_STATES:
            print(f"Batch job {batch_id} is in state {_state_name(batch.state)}. Waiting...")
            time.sleep(POLL_SECONDS)
            batch = google_ai_client.batches.get(name=batch_id)

        state = _state_name(batch.state)
        print(f"Batch job {batch_id} state: {state}")
        if batch.error:
            print(f"Batch job {batch_id} error: {batch.error}")

        updated_job = {**job, "state": state}
        if download and state in SUCCESS_STATES and not updated_job.get("output_file"):
            output_path = _download_batch_output(google_ai_client, batch, output_file_number)
            if output_path:
                updated_job["output_file"] = str(output_path)

        updated_jobs.append(updated_job)

    _save_manifest(updated_jobs)
    return updated_jobs


def call_llm(*, wait: bool = False) -> None:
    """Submit batch jobs, optionally waiting for completion."""
    jobs = submit_batches()
    if wait:
        check_batches(jobs, download=True, wait=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit/check/download Gemini batch jobs.")
    parser.add_argument("--status", action="store_true", help="Check existing batch job status from the manifest.")
    parser.add_argument("--download", action="store_true", help="Download completed batch outputs.")
    parser.add_argument("--wait", action="store_true", help="Poll until jobs reach a terminal state.")
    parser.add_argument(
        "--job-id", action="append", default=[], help="Batch job ID to check/download, e.g. batches/abc."
    )
    args = parser.parse_args()

    jobs = _jobs_from_ids(args.job_id) if args.job_id else _load_manifest()
    if args.status or args.download or args.job_id:
        check_batches(jobs, download=args.download, wait=args.wait)
    else:
        call_llm(wait=args.wait)


if __name__ == "__main__":
    main()
