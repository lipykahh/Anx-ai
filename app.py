import streamlit as st
import joblib
import speech_recognition as sr
import time

# Load model
model = joblib.load("anxiety_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

recognizer = sr.Recognizer()

st.title("🧠 Anxiety Detection AI")
st.write("Real-time speech based anxiety detection")

# Speech capture
if st.button("Start Listening"):

    with sr.Microphone() as source:
        st.write("🎙️ Speak now...")
        start_time = time.time()

        audio = recognizer.listen(source)

        end_time = time.time()
        duration = end_time - start_time

        try:
            text = recognizer.recognize_google(audio)

            st.write("📝 You said:", text)

            # ML prediction
            vector = vectorizer.transform([text])
            prediction = model.predict(vector)[0]
            prob = model.predict_proba(vector).max()

            st.write("🔍 Prediction:", prediction)
            st.write("📊 Anxiety probability:", round(prob,2))

            # Speech rate
            words = len(text.split())
            rate = words/duration

            st.write("🗣️ Speech rate:", round(rate,2),"words/sec")

            # Intervention logic
            if prob >= 0.7:
                st.error("⚠️ High anxiety detected")

                st.write("🧘 Breathing exercise:")
                st.write("Inhale for 4 seconds")
                st.write("Hold for 4 seconds")
                st.write("Exhale for 6 seconds")

            elif prob >= 0.4:
                st.warning("Mild anxiety detected. Try relaxing.")

            else:
                st.success("Anxiety level appears normal")

        except:
            st.write("❌ Could not understand audio")