import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv(override=True)

FORCE_DATAPIPELINE = True


def login_to_hub():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token, add_to_git_credential=True)
    else:
        print("HF_TOKEN not found in environment.")


# Dataset
dataset = "McAuley-Lab/Amazon-Reviews-2023"
base_link: str = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"

# Dataset names
dataset_names = [
    "Automotive",
    "Electronics",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
]

# local paths
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
ORIG_GZ_FILES_DIR: Path = DATA_DIR / "orig_zipped_files"
UNZIPPED_FILES_DIR: Path = DATA_DIR / "unzipped_files"
PROCESSED_ITEMS_DIR: Path = DATA_DIR / "processed_items"
ITEMS_CACHE: Path = DATA_DIR / "items_cache.parquet"
ITEMS_CACHE_WEIGHT_ADJUSTED: Path = DATA_DIR / "items_cache_weight_adjusted.parquet"

# domain training test files , kept in data/model_files/train/
MODEL_FILES_BASE_DIR: Path = DATA_DIR / "model_files" / "train"
TRAIN_DATA_FILE: Path = MODEL_FILES_BASE_DIR / "train_data.parquet"
VAL_DATA_FILE: Path = MODEL_FILES_BASE_DIR / "val_data.parquet"
TEST_DATA_FILE: Path = MODEL_FILES_BASE_DIR / "test_data.parquet"

# links for the dataset, replace with the ones from orig_zipped_files
links = {name: f"meta_{name}.jsonl.gz" for name in dataset_names}
