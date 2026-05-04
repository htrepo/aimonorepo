import asyncio
import json
import time
from typing import Any, Dict, List

import pandas as pd
from litellm import acompletion
from tqdm.asyncio import tqdm

from prjcostprediction import config
from prjcostprediction.domain.items import Item

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


async def process_item(
    item: Item,
    model_name: str,
    system_message: Dict[str, str],
) -> Dict[str, Any]:
    """Processes a single item using the LLM."""
    messages = [system_message, {"role": "user", "content": item.full}]
    try:
        start_time = time.time()
        response = await acompletion(model=model_name, messages=messages, api_base="http://localhost:11434")
        end_time = time.time()

        content = response.choices[0].message.content
        duration = end_time - start_time

        return {
            "id": item.id,
            "response": content,
            "time_taken": duration,
            "tokens": response.usage.total_tokens,
            "status": "success",
        }
    except Exception as e:
        return {"id": item.id, "error": str(e), "status": "error"}


async def run_inference(
    items: List[Item], model_name: str, concurrency: int = 20, output_file: str = "llm_results_async.json"
) -> List[Dict[str, Any]]:
    """Runs inference on a list of items using a worker pool to manage concurrency."""
    queue = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)

    results = []
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    pbar = tqdm(total=len(items), desc=f"Processing (concurrency={concurrency})")

    async def worker():
        while not queue.empty():
            item = await queue.get()
            result = await process_item(item, model_name, system_message)
            results.append(result)

            # Incremental save every 10 items
            if len(results) % 10 == 0:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)

            pbar.update(1)
            queue.task_done()

    # Create worker tasks
    workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(items)))]

    # Wait for all workers to finish
    await asyncio.gather(*workers)
    pbar.close()

    return results


def load_data(max_items: int = 10) -> List[Item]:
    """Loads and prepares item data."""
    train_df = pd.read_parquet(config.TRAIN_DATA_FILE)
    val_df = pd.read_parquet(config.VAL_DATA_FILE)
    test_df = pd.read_parquet(config.TEST_DATA_FILE)

    full_df = pd.concat([train_df, val_df, test_df])
    print(f"Total available items: {len(full_df)}")

    # Assign ID
    full_df["id"] = range(len(full_df))

    # Take sample
    sample_df = full_df.head(max_items)

    # Convert to Item objects
    return sample_df.apply(lambda row: Item(**row.to_dict()), axis=1).tolist()


async def main():
    # Configuration
    MODEL_NAME = "ollama/mistral:7b"
    MAX_ITEMS = 10
    CONCURRENCY = 20  # Adjust based on your hardware capabilities
    OUTPUT_FILE = "llm_results_async.json"

    # 1. Load data
    items = load_data(max_items=MAX_ITEMS)

    # 2. Run inference
    start_total = time.time()
    results = await run_inference(items, MODEL_NAME, concurrency=CONCURRENCY)
    end_total = time.time()

    # 3. Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # 4. Statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    print("\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(results)}")
    print(f"Total time: {end_total - start_total:.2f}s")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
