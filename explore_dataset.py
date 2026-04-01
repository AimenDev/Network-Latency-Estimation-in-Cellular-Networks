
import pandas as pd
# =========================
# Load Dataset
# =========================
df = pd.read_csv("signal_metrics.csv")


print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== LAST 5 ROWS =====")
print(df.tail())

# =========================
# Basic Info
# =========================
print("\n===== DATASET INFO =====")
df.info()

print("\n===== DATASET SHAPE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# =========================
# Statistics
# =========================
print("\n===== NUMERICAL STATS =====")
print(df.describe())

print("\n===== CATEGORICAL STATS =====")
print(df.describe(include="object"))

# =========================
# Missing Values
# =========================
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# =========================
# Unique Values Per Column
# =========================
print("\n===== UNIQUE VALUES =====")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# =========================
# Value Counts for Categorical Columns
# =========================
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())