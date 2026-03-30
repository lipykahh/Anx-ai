#Anx-ai

A small prototype to detect anxiety from short text/speech using a pre-trained scikit-learn model.

Contents
offline_inference.py — CLI-style script to run model predictions on sample texts.
realtime_speech.py — speech-based inference (requires microphone/audio setup).
app.py — Streamlit app UI (if present).
anxiety_model.pkl, tfidf_vectorizer.pkl, y.pkl — model and artifacts (binary).
anxiety_mini_1466k.csv — example dataset (if included).
requirements.txt — Python dependencies.
Requirements
Python 3.8+ (macOS)
zsh (default shell)
Recommended: create a virtual environment
Setup (macOS / zsh)
Create and activate venv:
  python3 -m venv .venv
  source .venv/bin/activate

Upgrade pip and install deps:
  pip install --upgrade pip
  pip install -r requirements.txt
Download NLTK data (if not already):
  python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
Run offline inference
The offline_inference.py file includes a small sample test. Run:

  python offline_inference.py
ou should see printed sample predictions and probabilities.

Run the Streamlit app
If app.py is present and uses Streamlit:

  streamlit run app.py

Notes about large/binary files
GitHub rejects files >100MB. For models or large CSVs consider using Git LFS:
  brew install git-lfs
  git lfs install
  git lfs track "*.pkl" "*.csv"
  git add .gitattributes
If large files are already committed, re-commit them into LFS before pushing.
