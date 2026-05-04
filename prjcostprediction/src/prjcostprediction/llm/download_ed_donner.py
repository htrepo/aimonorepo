import pandas as pd

splits = {"train": "data/train-00000-of-00001.parquet", "validation": "data/validation-00000-of-00001.parquet", "test": "data/test-00000-of-00001.parquet"}


def download_and_save(split: str):
    df = pd.read_parquet("hf://datasets/ed-donner/items_lite/" + splits[split])

    # save the df to output file
    df.to_parquet("data/" + split + ".parquet", index=False)

download_and_save("train")
download_and_save("validation")
download_and_save("test")