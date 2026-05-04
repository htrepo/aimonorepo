import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from prjcostprediction import config
from prjcostprediction.domain.items import Item

SYSTEM_PROMPT = """You will be provided with a product description. Create a concise summary for it. 
Respond ONLY in the following format. Do not include part numbers or introductory text.

Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


def run_inference() -> None:
    """Runs inference on items sequentially one by one."""
    # send all jsonl files from config.MODEL_INPUT_JSONL_FILES_BASE_DIR to MODEL
    files: List[Path] = config.MODEL_INPUT_JSONL_FILES_BASE_DIR.glob("*.jsonl")
    # list file info
    for file in files:
        print(f"File name: {file.name}")
        print(f"File size: {file.stat().st_size / 1024 / 1024:.2f} MB")


def make_jsonl(item: Item) -> Dict[str, Any]:
    if item.id is None:
        raise ValueError("Item must have an id before creating Gemini batch JSONL.")
    if item.full is None:
        raise ValueError(f"Item {item.id} has no full product description.")

    return {
        "key": str(item.id),
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": item.full}],
                }
            ],
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        },
    }


def create_jsonl_files(items: List[Item]) -> None:
    """Creates jsonl files for all items."""
    file_num = 1
    item_count = 0

    output_file_dir = config.MODEL_INPUT_JSONL_FILES_BASE_DIR
    output_file_dir.mkdir(parents=True, exist_ok=True)
    # delete existing files
    for file in output_file_dir.glob("*.jsonl"):
        file.unlink()

    # create new files
    for item in items:
        output_file = output_file_dir / f"batchinput_{file_num}.jsonl"
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(make_jsonl(item)) + "\n")
        item_count += 1
        if item_count % 1000 == 0:
            print(f"Processed {item_count} items")
            file_num += 1


def load_data() -> List[Item]:
    """Loads and prepares item data."""
    train_df = pd.read_parquet(config.TRAIN_DATA_FILE)
    val_df = pd.read_parquet(config.VAL_DATA_FILE)
    test_df = pd.read_parquet(config.TEST_DATA_FILE)

    full_df = pd.concat([train_df, val_df, test_df])
    print(f"Total available items: {len(full_df)}")

    # Assign ID
    full_df["id"] = range(len(full_df))

    # Convert to Item objects
    return full_df.apply(lambda row: Item(**row.to_dict()), axis=1).tolist()


def main():
    # Configuration

    # 1. Load data
    items = load_data()

    # create jsonl files for all items
    create_jsonl_files(items)

    # 2. Run inference
    run_inference()


if __name__ == "__main__":
    main()
