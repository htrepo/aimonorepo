import urllib.request
from pathlib import Path

from tqdm import tqdm

from prjcostprediction import config

# link - https://amazon-reviews-2023.github.io/

# download meta files for limited datasets


def download_file(url: str, dest_path: Path):
    """Download a file with a progress bar."""
    print(f"Downloading to {dest_path}...")

    # Custom reporthook for tqdm
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)


def download_datasets():
    config.ORIG_GZ_FILES_DIR.mkdir(parents=True, exist_ok=True)

    for key, filename in config.links.items():
        url = f"{config.base_link}{filename}"
        dest_path = config.ORIG_GZ_FILES_DIR / filename

        if dest_path.exists() and not config.FORCE_DATAPIPELINE:
            print(f"{filename} already exists. Skipping.")
            continue

        print(f"Downloading {key} from {url}")
        try:
            download_file(url, dest_path)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")


if __name__ == "__main__":
    download_datasets()
