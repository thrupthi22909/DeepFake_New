import cv2, os, numpy as np
from tqdm import tqdm

def preprocess_dataset(base_path, size=(128,128), batch_size=5000):
    save_dir = "preprocessed_batches"
    os.makedirs(save_dir, exist_ok=True)

    batch_data, batch_labels = [], []
    batch_count = 0
    total_images = 0

    for label, folder in enumerate(["real", "fake"]):
        path = os.path.join(base_path, folder)
        files = os.listdir(path)
        print(f"\n📂 Processing folder: {folder} ({len(files)} images)")

        for file in tqdm(files, desc=f"Processing {folder}"):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, size)
                batch_data.append(img / 255.0)
                batch_labels.append(label)
                total_images += 1

                # save every few thousand images
                if len(batch_data) >= batch_size:
                    np.savez_compressed(f"{save_dir}/batch_{batch_count}.npz",
                                        X=np.array(batch_data, dtype=np.float32),
                                        y=np.array(batch_labels))
                    print(f"💾 Saved batch {batch_count} ({len(batch_data)} images)")
                    batch_data, batch_labels = [], []
                    batch_count += 1

    # save remaining images
    if batch_data:
        np.savez_compressed(f"{save_dir}/batch_{batch_count}.npz",
                            X=np.array(batch_data, dtype=np.float32),
                            y=np.array(batch_labels))
        print(f"💾 Saved final batch {batch_count} ({len(batch_data)} images)")

    print(f"✅ Finished preprocessing {total_images} images in total!")

if __name__ == "__main__":
    preprocess_dataset("dataset1/train")
