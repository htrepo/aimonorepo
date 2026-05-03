import gzip

from tqdm import tqdm

from prjcostprediction import config


def unzip_files():
    config.UNZIPPED_FILES_DIR.mkdir(parents=True, exist_ok=True)

    for key, filename in config.links.items():
        filepath = config.ORIG_GZ_FILES_DIR / filename
        dest_path = config.UNZIPPED_FILES_DIR / f"{key}.jsonl"

        if dest_path.exists() and not config.FORCE_DATAPIPELINE:
            print(f"{filename} already exists. Skipping.")
            continue

        print(f"Unzipping {filename}...")
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                with open(dest_path, "w", encoding="utf-8") as out_file:
                    for line in tqdm(f, desc=f"Unzipping {filename}", unit="line"):
                        out_file.write(line)
            print(f"Successfully unzipped {filename}")
        except Exception as e:
            print(f"Failed to unzip {filename}: {e}")


if __name__ == "__main__":
    unzip_files()
