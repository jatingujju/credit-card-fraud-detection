import pandas as pd

df = pd.read_parquet("data/transactions.parquet")
df.sample(1000).to_csv("data/sample.csv", index=False)

print("✅ Sample dataset created!")