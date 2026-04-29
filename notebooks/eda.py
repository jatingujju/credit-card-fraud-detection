# ==========================================
# CREDIT CARD FRAUD - EDA SCRIPT
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 0. SETUP
# -----------------------------
print("📥 Loading data...")

# Ensure images folder exists
os.makedirs("images", exist_ok=True)

# Load parquet file
df = pd.read_parquet("data/transactions.parquet")

print("✅ Data Loaded")
print("Shape:", df.shape)

# -----------------------------
# 1. BASIC INFO
# -----------------------------
print("\n📊 First 5 rows:")
print(df.head())

print("\n📊 Columns:")
print(df.columns)

print("\n📊 Missing values:")
print(df.isnull().sum())

# -----------------------------
# 2. FRAUD DISTRIBUTION
# -----------------------------
print("\n🚨 Fraud Distribution:")
print(df['is_fraud'].value_counts())

print("\n🚨 Fraud Percentage:")
print(df['is_fraud'].value_counts(normalize=True))

# -----------------------------
# 3. HANDLE IMBALANCE (if needed)
# -----------------------------
# If dataset is 50-50, make it realistic
fraud_ratio = df['is_fraud'].mean()

if fraud_ratio > 0.1:
    print("\n⚙️ Adjusting dataset to realistic imbalance...")

    df_fraud = df[df['is_fraud'] == 1]
    df_normal = df[df['is_fraud'] == 0].sample(n=len(df_fraud)*20, random_state=42)

    df = pd.concat([df_normal, df_fraud]).sample(frac=1, random_state=42)

    print("New Shape:", df.shape)

# -----------------------------
# 4. FRAUD COUNT PLOT
# -----------------------------
plt.figure()
sns.countplot(x='is_fraud', data=df)
plt.title("Fraud vs Non-Fraud")
plt.savefig("images/fraud_distribution.png")
plt.close()

# -----------------------------
# 5. AMOUNT DISTRIBUTION
# -----------------------------
plt.figure()
sns.histplot(data=df, x='amount', hue='is_fraud', bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.savefig("images/amount_distribution.png")
plt.close()

# -----------------------------
# 6. CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.close()

# -----------------------------
# 7. TOP FEATURES CORRELATED WITH FRAUD
# -----------------------------
corr = df.corr()['is_fraud'].sort_values(ascending=False)

print("\n📊 Top correlated features with fraud:")
print(corr.head(10))

print("\n📊 Least correlated features:")
print(corr.tail(10))

# -----------------------------
# 8. SUMMARY
# -----------------------------
print("\n🎯 EDA Completed Successfully!")
print("📁 Plots saved in 'images/' folder")