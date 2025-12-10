import cv2, numpy as np, os
from tensorflow.keras.models import load_model

model = load_model("best_model.h5")
print("✅ Model loaded successfully!")

path = input("Enter image path: ").strip().strip('"')
if not os.path.exists(path):
    print("❌ File not found.")
    exit()

img = cv2.imread(path)
img = cv2.resize(img, (128,128)) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]
if pred > 0.5:
    print(f"🧠 Prediction: FAKE ({pred*100:.2f}% confidence)")
else:
    print(f"✅ Prediction: REAL ({(1-pred)*100:.2f}% confidence)")
