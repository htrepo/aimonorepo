import sys

from prjcostprediction import config
from prjcostprediction.dataprep.dataops._create_category_cache_files import create_category_cache_files
from prjcostprediction.dataprep.dataops._create_deduped_final_file import combine_cache_files_create_final_file
from prjcostprediction.dataprep.dataops._hf_data_download import download_datasets
from prjcostprediction.dataprep.dataops._unzip_data_files import unzip_files
from prjcostprediction.dataprep.dataops.main_weighted_adjustment import weight_adjusted_items

# Ensure stdout can handle unicode characters on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
config.login_to_hub()

# call download script
download_datasets()
# unzip files
unzip_files()
# create cache file for each category
create_category_cache_files()
# remove duplicates and create final parquet file
combine_cache_files_create_final_file()
# Adjust weights and create new file with weight adjusted items
weight_adjusted_items()
