from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests


URLS = {
    "images": "https://datasets-u2m.s3.eu-west-3.amazonaws.com/tomatOD_images.zip",
    "annotations": "https://datasets-u2m.s3.eu-west-3.amazonaws.com/tomatOD_annotations.zip",
}


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official tomatOD dataset.")
    parser.add_argument("--output-root", required=True, help="Directory to store downloaded files.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep downloaded zip files.")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    raw_dir = root / "downloads"
    extract_dir = root

    for name, url in URLS.items():
        zip_path = raw_dir / f"{name}.zip"
        print(f"[download] {url} -> {zip_path}")
        download_file(url, zip_path)
        print(f"[extract] {zip_path} -> {extract_dir}")
        extract_zip(zip_path, extract_dir)
        if not args.keep_zip:
            zip_path.unlink(missing_ok=True)

    print(f"[done] tomatOD extracted to: {root}")


if __name__ == "__main__":
    main()

