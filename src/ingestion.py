import pandas as pd
import os

BASE_DIR = os.getcwd()

INPUT_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "transactions.parquet")

def load_data():
    print("\n📥 Loading CSV...")
    df = pd.read_csv(INPUT_PATH)
    print("✅ Loaded shape:", df.shape)
    return df

def clean_data(df):
    print("\n🧹 Cleaning data...")

    # Rename columns
    df = df.rename(columns={
        "Amount": "amount",
        "Class": "is_fraud"
    })

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove invalid amounts
    df = df[df["amount"] > 0]

    print("✅ Cleaned shape:", df.shape)
    return df

def sort_data(df):
    print("\n⏳ No timestamp column → skipping sorting")
    return df

def save_data(df):
    print("\n💾 Saving Parquet...")
    df.to_parquet(OUTPUT_PATH, index=False)
    print("✅ Saved at:", OUTPUT_PATH)

def run_pipeline():
    df = load_data()
    df = clean_data(df)
    df = sort_data(df)
    save_data(df)

    print("\n🎯 Ingestion Completed Successfully!")

if __name__ == "__main__":
    run_pipeline()