"""
Dataset Downloader

Downloads and manages public datasets used by the
Hybrid Quantum Feature Selection Framework (HQFSF).
"""

from pathlib import Path
import urllib.request
import urllib.error


class DatasetDownloader:
    """
    Download and manage public datasets.
    """

    def __init__(self, download_dir="datasets/raw"):
        """
        Initialize the downloader.

        Parameters
        ----------
        download_dir : str
            Directory where datasets will be stored.
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, filename: str):
        """
        Download a dataset if it does not already exist.

        Parameters
        ----------
        url : str
            Dataset download URL.

        filename : str
            Name of the downloaded file.

        Returns
        -------
        pathlib.Path
            Path to the downloaded dataset.
        """

        destination = self.download_dir / filename

        # Skip download if file already exists
        if destination.exists():
            print("=" * 60)
            print("[INFO] Dataset already exists.")
            print(f"Location : {destination}")
            print("=" * 60)
            return destination

        print("=" * 60)
        print("Downloading Dataset...")
        print(f"Source      : {url}")
        print(f"Destination : {destination}")
        print("=" * 60)

        try:
            urllib.request.urlretrieve(url, destination)

            print()
            print("=" * 60)
            print("[SUCCESS] Dataset downloaded successfully.")
            print(f"Saved to : {destination}")
            print("=" * 60)

            return destination

        except urllib.error.HTTPError as error:
            print(f"[HTTP ERROR] {error.code} : {error.reason}")
            raise

        except urllib.error.URLError as error:
            print(f"[URL ERROR] {error.reason}")
            raise

        except Exception as error:
            print(f"[ERROR] {error}")
            raise

    def download_multiple(self, datasets: dict):
        """
        Download multiple datasets.

        Parameters
        ----------
        datasets : dict

        Example
        -------
        {
            "dataset.csv": "https://example.com/data.csv",
            "dataset2.csv": "https://example.com/data2.csv"
        }
        """

        downloaded_files = []

        print("=" * 60)
        print("Downloading Multiple Datasets")
        print("=" * 60)

        for filename, url in datasets.items():
            path = self.download(url, filename)
            downloaded_files.append(path)

        print()
        print("=" * 60)
        print("All downloads completed.")
        print("=" * 60)

        return downloaded_files

    def dataset_exists(self, filename: str):
        """
        Check whether a dataset exists.

        Parameters
        ----------
        filename : str

        Returns
        -------
        bool
        """
        return (self.download_dir / filename).exists()

    def list_datasets(self):
        """
        List all downloaded datasets.

        Returns
        -------
        list
        """
        return sorted(self.download_dir.glob("*"))

    def remove_dataset(self, filename: str):
        """
        Delete a dataset.

        Parameters
        ----------
        filename : str
        """

        file_path = self.download_dir / filename

        if file_path.exists():
            file_path.unlink()
            print(f"[SUCCESS] Removed {file_path}")
        else:
            print(f"[INFO] Dataset not found: {filename}")