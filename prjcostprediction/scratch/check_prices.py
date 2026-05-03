import pandas as pd

from prjcostprediction import config


def check_prices(path):
    df = pd.read_parquet(path)
    print("\n--- Mean Price by Category ---")
    print(df.groupby("category")["price"].mean().sort_values(ascending=False))
    print("\n--- Max Price by Category ---")
    print(df.groupby("category")["price"].max().sort_values(ascending=False))


if __name__ == "__main__":
    check_prices(config.ITEMS_CACHE)
