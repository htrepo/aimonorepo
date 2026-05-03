import random

import numpy as np
import pandas as pd

from prjcostprediction import config
from prjcostprediction.domain.items import Item

SIZE = 820_000


# weight adjust mechanism used below:
# - Use inverse category frequency to balance representation across all categories
# - Use normalized prices (linear) to favor higher prices without extreme skew
# - Apply aggressive penalties to historically dominant categories (Automotive/Tools)
# - Normalize weights to sum to 1 for probabilistic sampling


def _get_weight(items: list[Item]) -> list[float]:
    """get the weights for each item"""

    # 1. Collect all prices from the list of items into a numerical array for processing
    prices: np.ndarray[float] = np.array([item.price for item in items])

    # 2. Normalize the prices to a range between 0 and 1.
    # This makes it easier to compare items regardless of how expensive they are.
    # We add a tiny number (1e-9) at the end to prevent crashing if all prices are the same.
    p: np.ndarray[float] = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)

    # 3. Use normalized prices (linear) instead of squaring.
    # Squaring (p**2) was making expensive categories like Automotive too dominant.
    w: np.ndarray[float] = p

    # 4. Calculate inverse category frequency.
    # This gives smaller categories a boost and larger ones a penalty to balance the counts.
    categories: np.ndarray[str] = np.array([item.category for item in items])
    unique_cats, counts = np.unique(categories, return_counts=True)
    cat_count_dict = dict(zip(unique_cats, counts))
    # inv_freq will be calculated for each item based on the category it belongs to
    inv_freq: np.ndarray[float] = np.array([1.0 / cat_count_dict[cat] for cat in categories])
    # Combine price weight with inverse frequency. what this means is if a category has low count
    # then its weight will be increased and vice versa.
    w = w * inv_freq

    # 5. Apply aggressive reductions for historically dominant categories.
    # Even with inverse frequency, we want to actively suppress these.
    w[categories == "Tools_and_Home_Improvement"] *= 0.2
    w[categories == "Automotive"] *= 0.1

    # 6. Normalize weights to sum to 1.
    # w is available for each item based on the category it belongs to and its price
    # w.sum() gives the sum of all weights
    w: np.ndarray[float] = w / w.sum()

    return w


def weight_adjusted_items():
    # read the items cache
    df = pd.read_parquet(config.ITEMS_CACHE)
    items = [Item.model_validate(i) for i in df.to_dict(orient="records")]
    # get weight for each item
    w = _get_weight(items)
    # sample based on weight
    idx = np.random.choice(len(items), size=SIZE, replace=False, p=w)
    # create sample
    sample: list[Item] = [items[i] for i in idx]

    # shuffle data before writing
    random.seed(42)
    random.shuffle(sample)

    # create new file with weight adjusted items
    pd.DataFrame([item.model_dump() for item in sample]).to_parquet(config.ITEMS_CACHE_WEIGHT_ADJUSTED, index=False)
    print("Wrote to file")


if __name__ == "__main__":
    weight_adjusted_items()
