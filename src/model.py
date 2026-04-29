# ==========================================
# CREDIT CARD FRAUD - MODEL TRAINING
# ==========================================

import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("📥 Loading data...")

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_parquet("data/transactions.parquet")

print("✅ Data Loaded:", df.shape)

# -----------------------------
# 2. FEATURES & TARGET
# -----------------------------
X = df.drop(columns=["is_fraud", "id"])
y = df["is_fraud"]

print("\n📊 Features shape:", X.shape)
print("📊 Target shape:", y.shape)

# -----------------------------
# 3. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n✅ Split Done")
print("Train:", X_train.shape)
print("Test:", X_test.shape)

# -----------------------------
# 4. SCALING (for Logistic Regression)
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. LOGISTIC REGRESSION
# -----------------------------
print("\n🤖 Training Logistic Regression...")

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\n📊 ROC-AUC Score:", roc_auc_score(y_test, y_prob_lr))

# -----------------------------
# 6. RANDOM FOREST
# -----------------------------
print("\n🌲 Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n📊 Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

print("\n📊 Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\n📊 Random Forest ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# -----------------------------
# 7. SAVE MODELS
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(lr_model, "models/logistic_model.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n💾 Models saved in 'models/' folder")

# -----------------------------
# 8. DONE
# -----------------------------
print("\n🎯 MODEL TRAINING COMPLETED SUCCESSFULLY!")