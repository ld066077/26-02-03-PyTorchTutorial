from pathlib import Path
import shutil
import os

import kagglehub


def main():
    target_dir = Path(__file__).resolve().parent / "titanic"
    target_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(__file__).resolve().parent / ".cache" / "kagglehub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    download_path = Path(kagglehub.competition_download("titanic"))
    print(f"Path to competition files: {download_path}")

    copied_files = []
    for file_path in download_path.iterdir():
        if file_path.is_file():
            destination = target_dir / file_path.name
            shutil.copy2(file_path, destination)
            copied_files.append(destination.name)

    print("Copied files:", copied_files)
    print(f"Titanic data directory: {target_dir}")


if __name__ == "__main__":
    main()
