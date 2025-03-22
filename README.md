# 📰 Fake News Detection (Streamlit App)

A simple and powerful web app that detects **fake news** using **Natural Language Processing (NLP)** and **Machine Learning (Naive Bayes)**, built with **Streamlit**.

## Demo
[Try it Online](https://news-verify-42.streamlit.app/)

## Features

- Detects whether news is **FAKE** or **REAL**
- Clean and preprocess input text
- Uses `TfidfVectorizer` for text transformation
- Trained using `Multinomial Naive Bayes`
- Fast prediction on any custom news input
- Simple & responsive UI via Streamlit

---
## Dataset

- Dataset used for training:
> [Fake News Detection Datasets by Emine Yetim (Kaggle)](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets?resource=download)

- ชุดข้อมูลนี้มี 2 ไฟล์:
  - `Fake.csv` — ข่าวปลอม
  - `True.csv` — ข่าวจริง

- ใช้รวมกัน แล้วติดป้าย label เป็น `'FAKE'` หรือ `'REAL'` ก่อนส่งเข้าโมเดล

---
## Tech Stack

- Python
- Scikit-learn
- Joblib
- NLTK 
- Streamlit

---
## Sample Input

> `"NASA confirms aliens arrived in Thailand"`  
> → ❌ **FAKE**

---

> `"Local hero saves child from fire"`  
> → ✅ **REAL**
---

## 🔧 Model Training (Colab)

You can view or reproduce the model training process via Google Colab:  
👉 [Train Fake News Classifier on Colab](https://colab.research.google.com/drive/1sZVTB-yNfzFfAC0C4dp9fecrFvqXmIti?usp=sharing)

This Colab notebook includes:
- Loading & preprocessing the dataset (`Fake.csv`, `True.csv`)
- Text cleaning with `nltk`
- TF-IDF vectorization
- Model training using `MultinomialNB`
- Accuracy evaluation & model export (`joblib`)

## Project Structure
fake-news-detection/ ├── app.py # 🖥️ Main Streamlit web app ├── model_nb.pkl # 🤖 Trained Naive Bayes model (with scikit-learn) ├── vectorizer.pkl # 🔠 TF-IDF Vectorizer used during training ├── requirements.txt # 📦 List of all Python dependencies └── README.md # 📘 Project overview and instructions

## 🙋‍♂️ Author
Made with ❤️ by [ncqxm](https://github.com/ncqxm)
