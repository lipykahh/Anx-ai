import speech_recognition as sr
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK resources (safe if already downloaded)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load trained artifacts
model = joblib.load("anxiety_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing (same as training)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)

def predict_from_speech(text):
    clean = preprocess_text(text)
    vector = tfidf.transform([clean])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][1]
    label = "Anxiety" if pred == 1 else "Non-Anxiety"
    return label, round(prob, 3)
def panic_intervention(probability):
    if probability >= 0.7:
        print("\n🛑 High anxiety detected.")
        print("🧘 Let's slow things down together.")
        print("👉 Inhale slowly for 4 seconds...")
        print("👉 Hold for 4 seconds...")
        print("👉 Exhale gently for 6 seconds...")
        print("💬 You are safe. Focus on your breathing.\n")

    elif probability >= 0.4:
        print("\n⚠️ Mild anxiety detected.")
        print("💡 Try taking a deep breath and relaxing your shoulders.\n")

    else:
        print("\n✅ Anxiety level appears normal.\n")

# 🎙️ Speech Recognition
recognizer = sr.Recognizer()
mic = sr.Microphone()

print("🎙️ Speak now (Ctrl+C to stop)...")

with mic as source:
    recognizer.adjust_for_ambient_noise(source)

while True:
    try:
        with mic as source:
            audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio)
        print(f"\n📝 You said: {text}")

        label, prob = predict_from_speech(text)
        print(f"🔍 Prediction: {label} | Anxiety probability: {prob}")

        panic_intervention(prob)

    except sr.UnknownValueError:
        print("❌ Could not understand audio")
    except KeyboardInterrupt:
        print("\n🛑 Stopped")
        break