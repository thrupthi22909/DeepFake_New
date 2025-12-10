import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset1():
    dataset = "manjilkarki/deepfake-and-real-images"
    base_dir = "dataset1"
    zip_path = os.path.join(base_dir, "deepfake-and-real-images.zip")

    os.makedirs(base_dir, exist_ok=True)

    print("⬇️ Downloading DeepFake dataset (≈1.8 GB) from Kaggle...")
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset, path=base_dir, unzip=False)

    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(base_dir)

    print("✅ Dataset successfully downloaded and extracted to:", base_dir)

if __name__ == "__main__":
    download_dataset1()
