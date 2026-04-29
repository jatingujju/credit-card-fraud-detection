import pandas as pd
import joblib

print("🔮 Fraud Prediction System")

# -----------------------------
# 1. LOAD MODEL
# -----------------------------
model = joblib.load("models/random_forest.pkl")

# -----------------------------
# 2. SAMPLE INPUT
# -----------------------------
sample = pd.DataFrame([{
    'V1': 0.1, 'V2': -0.2, 'V3': 1.2, 'V4': 0.5, 'V5': -0.3,
    'V6': 0.7, 'V7': -0.1, 'V8': 0.2, 'V9': -0.5, 'V10': 0.3,
    'V11': -0.2, 'V12': 0.4, 'V13': -0.1, 'V14': 0.6, 'V15': -0.3,
    'V16': 0.2, 'V17': -0.4, 'V18': 0.1, 'V19': -0.2, 'V20': 0.3,
    'V21': -0.1, 'V22': 0.2, 'V23': -0.3, 'V24': 0.4, 'V25': -0.2,
    'V26': 0.1, 'V27': -0.3, 'V28': 0.2,
    'amount': 5000
}])

# -----------------------------
# 3. PREDICT PROBABILITY
# -----------------------------
probability = model.predict_proba(sample)[0][1]

# -----------------------------
# 4. APPLY THRESHOLD
# -----------------------------
threshold = 0.7  # stricter fraud detection

prediction = 1 if probability > threshold else 0

# -----------------------------
# 5. OUTPUT
# -----------------------------
print("\nPrediction:", "🚨 FRAUD" if prediction == 1 else "✅ NORMAL")
print("Fraud Probability:", round(probability, 4))