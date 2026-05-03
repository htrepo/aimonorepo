import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from prjcostprediction import config
from prjcostprediction.domain.items import Item


def plot_data(items: list[Item]):
    if False:
        # plot length of full text for each category
        lengths: list[int] = [len(item.full) for item in items]
        plt.figure(figsize=(15, 6))
        plt.title(f"Lengths: Avg {sum(lengths) / len(lengths):,.0f} and highest {max(lengths):,}\n")
        plt.xlabel("Length (chars)")
        plt.ylabel("Count")
        plt.hist(lengths, rwidth=0.7, color="lightblue", bins=range(0, 6000, 100))
        plt.show()

    if True:
        ## Plot the distribution of prices
        prices: list[float] = [item.price for item in items]
        plt.figure(figsize=(15, 6))
        plt.title(f"Prices: Avg {sum(prices) / len(prices):,.2f} and highest {max(prices):,}\n")
        plt.xlabel("Price ($)")
        plt.ylabel("Count")
        _, _, bars = plt.hist(prices, rwidth=0.7, color="orange", bins=range(0, 1000, 10))
        plt.bar_label(bars, padding=3, fontsize=8)
        plt.tight_layout()
        plt.show()

    if True:
        ## category distribution
        categories: dict[str, int] = {category: 0 for category in config.dataset_names}
        for item in items:
            categories[item.category] += 1
        plt.figure(figsize=(15, 6))
        plt.title(f"Category Distribution: Total {len(items)} items\n")
        plt.xlabel("Category")
        plt.ylabel("Count")
        bars = plt.bar(list(categories.keys()), list(categories.values()), color="green")
        plt.bar_label(bars)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # read items cache
    # pf = pq.ParquetFile(str(config.ITEMS_CACHE))
    pf = pq.ParquetFile(str(config.ITEMS_CACHE_WEIGHT_ADJUSTED))
    df = pf.read().to_pandas()
    items = [Item.model_validate(d) for d in df.to_dict(orient="records")]

    plot_data(items)
