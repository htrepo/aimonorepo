import random
import sys

import polars as pl
from tqdm import tqdm

from prjcostprediction import config
from prjcostprediction.domain.items import Item


def combine_cache_files_create_final_file():
    # check if config.ITEMS_CACHE exists and not FORCE_DATAPIPELINE
    if config.ITEMS_CACHE.exists() and not config.FORCE_DATAPIPELINE:
        print(f"{config.ITEMS_CACHE} exists. Skipping combining parquet files.")
        return

    items = []
    for category in tqdm(config.links.keys(), desc="Combining parquet files"):
        processed_category_filepath = config.PROCESSED_ITEMS_DIR / f"{category}.parquet"
        if not processed_category_filepath.exists():
            print(f"Cache not found for {category}: {processed_category_filepath}")
            continue

        df = pl.read_parquet(processed_category_filepath)
        # remove duplicates from df based on full and title columns
        df = df.unique(subset=["full", "title"], keep="first")
        items.extend([Item.model_validate(d) for d in df.to_dicts()])

    print("Removing duplicates from all items")

    seen = set()
    items = [x for x in tqdm(items) if not (x.full in seen or seen.add(x.full))]
    del seen

    seen = set()
    items = [x for x in tqdm(items) if not (x.title in seen or seen.add(x.title))]
    del seen

    # Randomly shuffle the list of items

    random.seed(42)

    random.shuffle(items)

    # Save to cache

    print(f"Saving {len(items)} items to cache: {config.ITEMS_CACHE}")

    df = pl.DataFrame([item.model_dump() for item in items])

    df.write_parquet(config.ITEMS_CACHE)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    combine_cache_files_create_final_file()
    count = pl.scan_parquet(config.ITEMS_CACHE).select(pl.len()).collect().item()
    print(f"Successfully created cache at {config.ITEMS_CACHE}")
    print(f"Total items in combined cache: {count:,}")
