import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/random_forest.pkl")

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1 style='text-align: center;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered fraud prediction system</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Transaction Settings")

amount = st.sidebar.slider("Transaction Amount", 0, 20000, 1000)

threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.7)

st.sidebar.markdown("### ℹ️ Info")
st.sidebar.write("Higher threshold → stricter fraud detection")

# -----------------------------
# Main Layout (Columns)
# -----------------------------
st.subheader("📊 Enter Transaction Features")

col1, col2, col3 = st.columns(3)

features = {}

for i in range(1, 29):
    if i <= 10:
        features[f"V{i}"] = col1.slider(f"V{i}", -5.0, 5.0, 0.0)
    elif i <= 20:
        features[f"V{i}"] = col2.slider(f"V{i}", -5.0, 5.0, 0.0)
    else:
        features[f"V{i}"] = col3.slider(f"V{i}", -5.0, 5.0, 0.0)

# -----------------------------
# Predict Button
# -----------------------------
st.divider()

if st.button("🚀 Predict Fraud", use_container_width=True):

    input_data = pd.DataFrame([{
        **features,
        "amount": amount
    }])

    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability > threshold else 0

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"🚨 Fraud Detected!\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Normal Transaction\n\nProbability: {probability:.2f}")

    st.info(f"Threshold used: {threshold}")