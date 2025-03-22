import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# โหลด NLTK resources (รอบแรกเท่านั้น)
nltk.download('stopwords')

# โหลดโมเดลและเวกเตอร์
model = joblib.load("model_nb.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ฟังก์ชันล้างข้อความ
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    cleaned = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned)

# UI
st.title("Fake News Detector")
st.markdown("ใส่ข่าวแล้วดูว่าเป็น **ข่าวจริงหรือข่าวปลอม**")

input_text = st.text_area("พิมพ์ข่าวที่ต้องการตรวจสอบ:")

if st.button("ตรวจสอบข่าว"):
    cleaned = preprocess(input_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    
    if prediction == "FAKE":
        st.error("ข่าวนี้น่าจะเป็น **ข่าวปลอม**")
    else:
        st.success("ข่าวนี้น่าจะเป็น **ข่าวจริง**")
