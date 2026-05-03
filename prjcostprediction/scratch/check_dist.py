import pandas as pd

from prjcostprediction import config


def check_distribution(path, name):
    if not path.exists():
        print(f"{name} does not exist at {path}")
        return
    df = pd.read_parquet(path)
    print(f"\n--- {name} Distribution ---")
    print(df["category"].value_counts())
    print(f"Total: {len(df)}")


if __name__ == "__main__":
    check_distribution(config.ITEMS_CACHE, "Original Cache")
    check_distribution(config.ITEMS_CACHE_WEIGHT_ADJUSTED, "Weight Adjusted Cache")
