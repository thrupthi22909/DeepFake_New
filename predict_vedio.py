import os
import cv2
import numpy as np
import imageio
from tensorflow.keras.models import load_model

MODEL_PATH = "video_best_model.h5"
IMG_SIZE = 128
FRAME_COUNT = 10

model = load_model(MODEL_PATH)
print("✅ Loaded model:", MODEL_PATH)

def read_frames(path, num_frames=FRAME_COUNT):
    reader = imageio.get_reader(path, "ffmpeg")
    total = reader.count_frames()
    idxs = np.linspace(0, total-1, num_frames, dtype=int)
    frames = []
    for i, frame in enumerate(reader):
        if i in idxs:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame.astype("float32") / 255.0
            frames.append(frame)
    reader.close()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.expand_dims(np.array(frames), axis=0)

while True:
    path = input("\n🎬 Enter video path (or 'exit'): ").strip()
    if path.lower() == "exit":
        break
    if not os.path.exists(path):
        print("❌ File not found.")
        continue

    print(f"🎞 Analyzing: {path}")
    X = read_frames(path)
    pred = model.predict(X)[0][0]
    label = "🟢 REAL VIDEO" if pred < 0.5 else "🔴 FAKE VIDEO"
    confidence = (1 - pred if pred < 0.5 else pred) * 100
    print(f"Raw model output: {pred:.4f}")
    print(f"✅ Prediction: {label}")
    print(f"🎯 Confidence: {confidence:.2f}%")

    # Play video (optional)
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(label, frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
