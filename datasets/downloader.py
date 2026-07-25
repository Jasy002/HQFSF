"""
Dataset Downloader

Downloads and manages public datasets used by HQFSF.
"""

from pathlib import Path
import urllib.request


class DatasetDownloader:
    """
    Download datasets from public URLs.
    """

    def __init__(self, download_dir="datasets/raw"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, filename: str):
        """
        Download dataset if not already available.
        """

        destination = self.download_dir / filename

        if destination.exists():
            print(f"[INFO] Dataset already exists:\n{destination}")
            return destination

        print(f"[INFO] Downloading dataset...")

        urllib.request.urlretrieve(url, destination)

        print(f"[SUCCESS] Dataset saved to:\n{destination}")

        return destination