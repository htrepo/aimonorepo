import json
import time
from typing import Any, Dict, List

import pandas as pd
from litellm import completion
from tqdm import tqdm

from prjcostprediction import config
from prjcostprediction.domain.items import Item

SYSTEM_PROMPT = """You will be provided with a product description. Create a concise summary for it. 
Respond ONLY in the following format. Do not include part numbers or introductory text.

Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


def process_item(
    item: Item,
    model_name: str,
    system_message: Dict[str, str],
) -> Dict[str, Any]:
    """Processes a single item using the LLM."""
    # Construct user message with the item details
    user_content = f"Product ID {item.id}:\n{item.full}"
    messages = [system_message, {"role": "user", "content": user_content}]

    try:
        start_time = time.time()
        # Note: Using Gemini API via LiteLLM. Ensure GOOGLE_API_KEY is set in your environment.
        response = completion(model=model_name, messages=messages)
        end_time = time.time()

        content = response.choices[0].message.content
        duration = end_time - start_time

        return {
            "id": item.id,
            "response": content.strip(),
            "time_taken": duration,
            "tokens": response.usage.total_tokens,
            "status": "success",
        }
    except Exception as e:
        return {
            "id": item.id, 
            "error": str(e), 
            "status": "error",
            "message": "Make sure GOOGLE_API_KEY is set for Gemini models."
        }


def run_inference(items: List[Item], model_name: str, output_file: str = "llm_results.json") -> List[Dict[str, Any]]:
    """Runs inference on items sequentially one by one."""
    results: list[dict[str, Any]] = []
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    # Progress bar tracks number of items
    pbar = tqdm(total=len(items), desc="Processing items")

    for item in items:
        result = process_item(item, model_name, system_message)
        results.append(result)

        # Incremental save every item
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        pbar.update(1)

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


def main():
    # Configuration
    # Using gemini-1.5-flash as the best available equivalent to "nano" for API usage.
    MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"
    MAX_ITEMS = 20
    OUTPUT_FILE = "llm_results.json"

    # 1. Load data
    items = load_data(max_items=MAX_ITEMS)

    # 2. Run inference
    start_total = time.time()
    results = run_inference(items, MODEL_NAME, output_file=OUTPUT_FILE)
    end_total = time.time()

    # 3. Save final results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # 4. Statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(results)}")
    print(f"Total time: {end_total - start_total:.2f}s")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
