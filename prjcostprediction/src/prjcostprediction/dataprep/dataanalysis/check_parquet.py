import pyarrow.parquet as pq

from prjcostprediction import config


def check_parquet_size():
    # check parquet number of lines without reading it
    file_path = str(config.ITEMS_CACHE_WEIGHT_ADJUSTED)
    pf = pq.ParquetFile(file_path)
    num_rows = pf.metadata.num_rows
    print(num_rows)


def print_parquet():
    """
    This method prints 3 lines from the top of the parquet file.
    """
    # check parquet number of lines without reading it
    file_path = str(config.ITEMS_CACHE_WEIGHT_ADJUSTED)
    pf = pq.ParquetFile(file_path)
    print(pf.read().to_pandas().head(3))


if __name__ == "__main__":
    check_parquet_size()
    print_parquet()
