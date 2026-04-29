# 💳 Credit Card Fraud Detection System

## 🚀 Overview

This project is an end-to-end Machine Learning system to detect fraudulent credit card transactions using classification models and an interactive Streamlit application.

---

## 🎯 Problem Statement

Credit card fraud leads to significant financial losses. The objective is to accurately detect fraudulent transactions while minimizing false positives.

---

## 🛠️ Solution

* Performed data preprocessing and cleaning
* Conducted Exploratory Data Analysis (EDA)
* Built and compared models:

  * Logistic Regression (baseline)
  * Random Forest (advanced)
* Evaluated using Precision, Recall, F1-score, ROC-AUC
* Developed a Streamlit UI for real-time fraud prediction

---

## 📊 Model Performance

* Logistic Regression ROC-AUC: ~0.99
* Random Forest ROC-AUC: ~0.99

---

## 📊 Exploratory Data Analysis

### 🔹 Correlation Heatmap

![Heatmap](images/heatmap.png)

### 🔹 Transaction Amount Distribution

![Amount Distribution](images/amount.png)

---

## 🌐 Application Interface

![Streamlit App](images/app.png)

---

## 📂 Dataset

Dataset is not included due to GitHub size limitations.

Download from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place it in:
data/transactions.parquet

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Key Learning

Fraud detection is not just about accuracy — it requires balancing precision, recall, and decision thresholds based on business needs.

---

## 👨‍💻 Author

Jatin Gujarathi
