import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Create dataset2 folder
os.makedirs("dataset2_raw", exist_ok=True)

api = KaggleApi()
api.authenticate()

print("📥 Downloading DFDC dataset from Kaggle...")
api.dataset_download_files("muhammedashiqkm/dfdc-dataset", path="dataset2_raw", unzip=True)
print("✅ Dataset downloaded and extracted to dataset2_raw/")
