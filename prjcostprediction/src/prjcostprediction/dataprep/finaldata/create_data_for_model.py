import pandas as pd
import pyarrow.parquet as pq

from prjcostprediction import config

FULL_MODE = [800_000, 10_000]
LITE_MODE = [20_000, 1000]
MODE = LITE_MODE


def split_data():
    # read weight adjusted items cache
    df_read = pd.read_parquet(config.ITEMS_CACHE_WEIGHT_ADJUSTED)
    # size of df
    print(f"dataframe with adjusted weights items has len(df):{len(df_read)}")

    # take only items required for train+val+test
    df = df_read.head(MODE[0] + MODE[1] + MODE[1])

    # take items from the df into train_file
    train_df = df.head(MODE[0])
    train_df.to_parquet(config.TRAIN_DATA_FILE)

    # take items into val_file
    val_df = df.iloc[MODE[0] : MODE[0] + MODE[1]]
    val_df.to_parquet(config.VAL_DATA_FILE)

    # take remaining items into test_file
    test_df = df.iloc[MODE[0] + MODE[1] :]
    test_df.to_parquet(config.TEST_DATA_FILE)

    # print statistics for all pq files using numpy as dataframe approach is very slow
    print("Train data:", pq.ParquetFile(config.TRAIN_DATA_FILE).metadata.num_rows)
    print("Val data:", pq.ParquetFile(config.VAL_DATA_FILE).metadata.num_rows)
    print("Test data:", pq.ParquetFile(config.TEST_DATA_FILE).metadata.num_rows)


if __name__ == "__main__":
    split_data()
