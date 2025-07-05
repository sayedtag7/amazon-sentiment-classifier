import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# تحميل stopwords لو مش موجودة
nltk.download("stopwords")

# تحميل الموديل والفيكتورايزر
model = joblib.load("model/best_logistic_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# تهيئة الـ stopwords والـ stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# دالة التنظيف
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", text)
    tokens = text.split()
    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(cleaned)

# واجهة Streamlit
st.set_page_config(
    page_title="Amazon Sentiment Classifier",
    page_icon="🛒",
    layout="centered"
)

st.title("🛍️ Amazon Reviews Sentiment Classifier")
st.write("Enter your product review below to predict whether it's Positive or Negative.")

# Text area for input
review = st.text_area("✍️ **Review:**", height=200)

if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review text.")
    else:
        clean = preprocess(review)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0].max()
        label = "✅ Positive 😊" if pred == 1 else "❌ Negative ☹️"
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {prob:.2%}")
