def clean_data(df):
    print("\n🧹 Cleaning data...")

    # Rename columns for consistency
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