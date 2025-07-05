import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# تحميل الموديل والفيكتورايزر
model = joblib.load("model/best_logistic_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# دالة الـpreprocess
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", text)
    tokens = text.split()
    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(cleaned)

# واجهة Streamlit
st.title("Amazon Reviews Sentiment Classifier")

review = st.text_area("Enter a product review:")

if st.button("Analyze"):
    clean = preprocess(review)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0].max()
    label = "Positive 😊" if pred == 1 else "Negative ☹️"
    st.success(f"Prediction: {label} (Confidence: {prob:.2%})")
