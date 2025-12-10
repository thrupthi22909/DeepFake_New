import os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Demo fake news samples
texts = [
    "Breaking news! Drinking bleach cures COVID-19.",
    "Scientists discovered a new treatment for cancer.",
    "Aliens have landed in Paris last night!",
    "Government approves new education policy to help students.",
    "NASA confirms water found on Mars surface",
    "COVID-19 vaccine causes people to turn into zombies",
    "The Prime Minister announced new education reforms today",
    "Drinking bleach cures coronavirus",
    "World Health Organization approves new malaria vaccine",
    "Aliens built the pyramids, scientists finally admit",
]
labels = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # 1 = fake, 0 = real

vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

os.makedirs("model", exist_ok=True)
with open("model/fake_news_model.pkl", "wb") as fm:
    pickle.dump(model, fm)
with open("model/tfidf_vectorizer.pkl", "wb") as vf:
    pickle.dump(vectorizer, vf)

print("✅ Demo Fake News model and vectorizer created!")
