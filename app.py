import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="CardGuard AI",
    page_icon="💳",
    layout="wide"
)

# Title Section
st.title("💳 CardGuard AI")

st.markdown("""
Welcome to **CardGuard AI** 🚀

An intelligent Machine Learning powered system for detecting fraudulent credit card transactions in real time.
""")

# Load Models
@st.cache_resource
def load_models():
    logistic_model = joblib.load('models/logistic_model.pkl')
    random_forest_model = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')

    return logistic_model, random_forest_model, scaler


# Try Loading Models
try:
    logistic_model, random_forest_model, scaler = load_models()
    st.success("✅ Models Loaded Successfully")

except Exception as e:
    st.error(f"❌ Error Loading Models: {e}")
    st.stop()


# Sidebar
st.sidebar.header("⚙️ Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ["Logistic Regression", "Random Forest"]
)

# Main Section
st.header("📝 Enter Transaction Details")

# Input Columns
col1, col2, col3 = st.columns(3)

with col1:
    time = st.number_input("Time", min_value=0.0, value=10000.0)
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)
    v6 = st.number_input("V6", value=0.0)
    v7 = st.number_input("V7", value=0.0)
    v8 = st.number_input("V8", value=0.0)
    v9 = st.number_input("V9", value=0.0)

with col2:
    v10 = st.number_input("V10", value=0.0)
    v11 = st.number_input("V11", value=0.0)
    v12 = st.number_input("V12", value=0.0)
    v13 = st.number_input("V13", value=0.0)
    v14 = st.number_input("V14", value=0.0)
    v15 = st.number_input("V15", value=0.0)
    v16 = st.number_input("V16", value=0.0)
    v17 = st.number_input("V17", value=0.0)
    v18 = st.number_input("V18", value=0.0)
    v19 = st.number_input("V19", value=0.0)

with col3:
    v20 = st.number_input("V20", value=0.0)
    v21 = st.number_input("V21", value=0.0)
    v22 = st.number_input("V22", value=0.0)
    v23 = st.number_input("V23", value=0.0)
    v24 = st.number_input("V24", value=0.0)
    v25 = st.number_input("V25", value=0.0)
    v26 = st.number_input("V26", value=0.0)
    v27 = st.number_input("V27", value=0.0)
    v28 = st.number_input("V28", value=0.0)
    amount = st.number_input("Amount", min_value=0.0, value=100.0)

# Prediction Button
if st.button("🔍 Predict Transaction"):

    try:
        input_data = np.array([[
            time, v1, v2, v3, v4, v5, v6, v7, v8, v9,
            v10, v11, v12, v13, v14, v15, v16, v17,
            v18, v19, v20, v21, v22, v23, v24, v25,
            v26, v27, v28, amount
        ]])

        # Scale Input Data
        scaled_data = scaler.transform(input_data)

        # Prediction
        if model_choice == "Logistic Regression":
            prediction = logistic_model.predict(scaled_data)
            probability = logistic_model.predict_proba(scaled_data)[0][1]

        else:
            prediction = random_forest_model.predict(scaled_data)
            probability = random_forest_model.predict_proba(scaled_data)[0][1]

        # Result
        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")

        else:
            st.success("✅ Legitimate Transaction")

        st.write(f"Fraud Probability: {probability:.2%}")

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# CSV Upload Section
st.header("📂 Batch Prediction Using CSV")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        if st.button("🚀 Run Batch Prediction"):

            scaled_batch = scaler.transform(data)

            if model_choice == "Logistic Regression":
                predictions = logistic_model.predict(scaled_batch)

            else:
                predictions = random_forest_model.predict(scaled_batch)

            data["Prediction"] = predictions

            data["Prediction"] = data["Prediction"].map({
                0: "Legitimate",
                1: "Fraud"
            })

            st.success("✅ Batch Prediction Completed")

            st.dataframe(data.head())

            # Download Predictions
            csv = data.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ CSV Processing Error: {e}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")