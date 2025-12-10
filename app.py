import os
import time
import cv2
import base64
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
FRAME_FOLDER = "static/frames"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"}

IMAGE_MODEL_PATH = "best_model.h5"
VIDEO_MODEL_CANDIDATES = ["video_best_model.h5"]

FAKE_NEWS_MODEL_PATH = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

SIGMOID_OUTPUT_IS_FAKE = True
VIDEO_FALLBACK_FRAME_COUNT = 10

# ---------------- INIT APP ----------------
app = Flask(__name__)
app.secret_key = "deepfake_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# ---------------- LOAD MODELS ----------------
def try_load_image_model(path):
    if os.path.exists(path):
        print(f"✅ Loading image model from: {path}")
        return load_model(path)
    print(f"⚠️ Image model not found: {path}")
    return None


def try_find_and_load_video_model(candidates):
    for p in candidates:
        if os.path.exists(p):
            print(f"✅ Loading video model from: {p}")
            return load_model(p)
    print("⚠️ No video model file found among candidates.")
    return None


def try_load_fake_news_model():
    if os.path.exists(FAKE_NEWS_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(FAKE_NEWS_MODEL_PATH, "rb") as fm:
            fake_news_model = pickle.load(fm)
        with open(VECTORIZER_PATH, "rb") as vf:
            vectorizer = pickle.load(vf)
        print("✅ Fake News model loaded.")
        return fake_news_model, vectorizer
    print("⚠️ Fake News model/vectorizer not found.")
    return None, None


image_model = try_load_image_model(IMAGE_MODEL_PATH)
video_model = try_find_and_load_video_model(VIDEO_MODEL_CANDIDATES)
fake_news_model, vectorizer = try_load_fake_news_model()

# ---------------- HELPERS ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_image_input_size(model, default=(128, 128)):
    if model is None:
        return default
    try:
        shape = model.input_shape
        if len(shape) >= 4:
            return (int(shape[2]), int(shape[1]))
    except Exception:
        pass
    return default


def preprocess_image_file(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def extract_uniform_frames(video_path, num_frames=10, target_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, target_size)
        frames.append(resized)
    cap.release()
    return frames


def frames_to_base64(frames):
    """Convert extracted frames to Base64 for HTML display."""
    frame_images = []
    for f in frames:
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buffer).decode("utf-8")
        frame_images.append(b64)
    return frame_images


def interpret_raw_prediction(raw_pred):
    arr = np.array(raw_pred)
    if arr.ndim == 2 and arr.shape[1] == 1:
        val = float(arr[0, 0])
        return (1 - val, val) if SIGMOID_OUTPUT_IS_FAKE else (val, 1 - val)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return float(arr[0, 0]), float(arr[0, 1])
    val = float(np.ravel(arr)[0])
    prob_fake = val if SIGMOID_OUTPUT_IS_FAKE else (1 - val)
    return 1 - prob_fake, prob_fake


def clean_text(text):
    import re
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


def fetch_article_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return " ".join(p.get_text() for p in soup.find_all("p"))
    except Exception as e:
        print("Error fetching URL:", e)
        return ""


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detection")
def detection():
    return render_template("detection.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/fake_news")
def fake_news():
    return render_template("fake_news.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file uploaded")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected")
        return redirect(request.url)
    if not allowed_file(file.filename):
        return "❌ Invalid file type"

    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    filename = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    ext = filename.split(".")[-1].lower()
    is_image = ext in {"png", "jpg", "jpeg"}
    is_video = ext in {"mp4", "avi", "mov", "mkv"}

    image_path = f"uploads/{filename}" if is_image else None
    video_path = f"uploads/{filename}" if is_video else None

    try:
        if is_image:
            target_w, target_h = _get_image_input_size(image_model)
            data = preprocess_image_file(save_path, target_size=(target_w, target_h))
            raw = image_model.predict(data)
            prob_real, prob_fake = interpret_raw_prediction(raw)
            label = "Real" if prob_fake >= 0.5 else "Fake"
            confidence = max(prob_real, prob_fake) * 100
            frames_b64 = None

        elif is_video:
            frames = extract_uniform_frames(save_path, VIDEO_FALLBACK_FRAME_COUNT)
            if video_model:
                # Predict with video model if available
                data = np.expand_dims(np.array(frames) / 255.0, axis=0)
                raw = video_model.predict(data)
                prob_real, prob_fake = interpret_raw_prediction(raw)
            else:
                preds = [image_model.predict(np.expand_dims(f / 255.0, axis=0)) for f in frames]
                prob_fakes = [interpret_raw_prediction(p)[1] for p in preds]
                prob_fake = float(np.mean(prob_fakes))
                prob_real = 1 - prob_fake
            label = "Fake" if prob_fake >= 0.5 else "Real"
            confidence = max(prob_real, prob_fake) * 100
            frames_b64 = frames_to_base64(frames)

        else:
            return "❌ Unsupported file type."

        return render_template(
            "result.html",
            result=label,
            confidence=confidence,
            image_path=image_path,
            video_path=video_path,
            frames=frames_b64
        )

    except Exception as e:
        print("❌ Prediction Error:", e)
        return f"❌ Prediction Error: {e}"


@app.route("/predict_news", methods=["POST"])
def predict_news():
    input_type = request.form.get("input_type")
    text_input = request.form.get("news_input", "").strip()
    if not text_input:
        return "❌ Please enter text or URL."

    if input_type == "url":
        text = fetch_article_text(text_input)
    else:
        text = text_input

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = fake_news_model.predict(features)[0]
    prob = fake_news_model.predict_proba(features)[0]
    confidence = max(prob) * 100
    label = "Fake News Detected" if pred == 1 else "Real News"

    return render_template("fake_news_result.html", result=label, confidence=confidence, input_text=text_input)


if __name__ == "__main__":
    app.run(debug=True)
