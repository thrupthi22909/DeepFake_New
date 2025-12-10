import os
import numpy as np
import cv2
import imageio
from tqdm import tqdm

DATASET_ROOT = "dataset2_raw"
OUTPUT_FILE = "dataset2_processed.npz"

IMG_SIZE = 128
FRAME_COUNT = 10

def read_video_frames(video_path, num_frames=FRAME_COUNT, size=(IMG_SIZE, IMG_SIZE)):
    try:
        reader = imageio.get_reader(video_path, "ffmpeg")
        total = reader.count_frames()
        if total <= 0:
            reader.close()
            return None

        idxs = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        for i, frame in enumerate(reader):
            if i in idxs:
                frame = cv2.resize(frame, size)
                frame = frame.astype("float32") / 255.0
                frames.append(frame)
        reader.close()

        if len(frames) < num_frames:
            while len(frames) < num_frames:
                frames.append(frames[-1])
        return np.stack(frames)
    except Exception as e:
        print(f"⚠️ Could not read {video_path}: {e}")
        return None

def load_all_videos():
    X, y = [], []
    for source in ["Celeb-DF", "FF++"]:
        for label_name, label_value in [("real", 0), ("fake", 1)]:
            folder = os.path.join(DATASET_ROOT, source, label_name)
            if not os.path.exists(folder):
                continue
            print(f"📂 Loading {source}/{label_name} ...")
            for file in tqdm(os.listdir(folder)):
                if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    continue
                path = os.path.join(folder, file)
                frames = read_video_frames(path)
                if frames is not None:
                    X.append(frames)
                    y.append(label_value)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_all_videos()
    print("✅ Dataset loaded:", X.shape, y.shape)
    np.savez(OUTPUT_FILE, X=X, y=y)
    print(f"💾 Saved to {OUTPUT_FILE}")
