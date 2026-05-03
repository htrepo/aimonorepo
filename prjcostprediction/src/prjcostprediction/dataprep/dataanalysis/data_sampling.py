import gzip
import json

from tqdm import tqdm

from prjcostprediction import config


def data_sampling():
    # load data\meta_Appliances.jsonl
    with gzip.open(config.DATA_DIR / "meta_Appliances.jsonl", "r") as f:
        data: list[str] = f.readlines()

    # pretty print record sample  row 6
    # print(json.dumps(json.loads(data[6]), indent=4))

    # iterate dataset to find max price
    max_price = 0
    max_item = None
    for line in tqdm(data):
        item = json.loads(line)
        if item["price"] is None:
            continue
        if item["price"] > max_price:
            max_price = item["price"]
            max_item = item
    print(f"Max price: {max_price}")
    print(f"Max item: {max_item}")


if __name__ == "__main__":
    data_sampling()
