
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK resources (safe if already downloaded)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load artifacts
model = joblib.load("anxiety_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing (MUST match Phase 1)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)

def predict_anxiety(text):
    clean = preprocess_text(text)
    vector = tfidf.transform([clean])   # ⚠️ transform only
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][1]
    label = "Anxiety" if pred == 1 else "Non-Anxiety"
    return label, round(prob, 3)

# 🔍 Test with sample inputs
samples = [
    "I feel my heart racing and I can't breathe",
    "I am calm and enjoying my day",
    "Suddenly I feel panicked and dizzy"
]

for s in samples:
    label, prob = predict_anxiety(s)
    print(f"Text: {s}")
    print(f"Prediction: {label} | Anxiety probability: {prob}\n")