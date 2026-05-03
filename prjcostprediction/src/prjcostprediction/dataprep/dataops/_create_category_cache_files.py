import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
from tqdm import tqdm

from prjcostprediction import config
from prjcostprediction.dataprep.dataops.parser import parse

DEBUG = True


def _create_category_cache_file(category_name: str, input_filename: str) -> None:
    """
    This method processes a single category file and saves it to a parquet cache.
    Memory-optimized: Stores dictionaries instead of Pydantic objects.
    """
    print(f"Starting {category_name}...")
    filepath = config.UNZIPPED_FILES_DIR / f"{category_name}.jsonl"
    cache_filepath = config.PROCESSED_ITEMS_DIR / f"{category_name}.parquet"

    if cache_filepath.exists() and not config.FORCE_DATAPIPELINE:
        return

    if not filepath.exists():
        print(f"File not found: {filepath}. Skipping.")
        return

    category_items = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    datapoint = json.loads(line)
                    item = parse(datapoint, category_name)
                    if item is not None:
                        # Store as dict to save memory during parallel processing
                        category_items.append(item.model_dump())
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading {category_name}: {e}")
        return

    if not category_items:
        return

    config.PROCESSED_ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(category_items)
    df.write_parquet(cache_filepath)
    print(f"Finished {category_name}: {len(category_items)} items saved.")


def create_category_cache_files():
    """
    Creates category caches in parallel using a process pool.
    """
    if config.ITEMS_CACHE.exists() and not config.FORCE_DATAPIPELINE:
        print(f"{config.ITEMS_CACHE} exists. Skipping category cache creation.")
        return

    print(f"Processing {len(config.links)} categories in parallel (4 workers)...")

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_create_category_cache_file, category, filename): category
            for category, filename in config.links.items()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Total Progress"):
            category = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Category {category} generated an exception: {e}")


if __name__ == "__main__":
    create_category_cache_files()
