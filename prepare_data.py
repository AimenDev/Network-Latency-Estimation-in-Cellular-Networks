import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "data"
RAW_FILE = "signal_metrics.csv"
TARGET = "Latency (ms)"

os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv(RAW_FILE)
print(f"📂 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------
# Feature Engineering
# ----------------------------
# Extract time features from Timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["hour"] = df["Timestamp"].dt.hour
df["day_of_week"] = df["Timestamp"].dt.dayofweek

# Drop Signal Quality — all values are 0, carries no information
df.drop(columns=["Signal Quality (%)"], inplace=True)

# One-Hot Encode Network Type (avoids false numeric ordering)
df = pd.get_dummies(df, columns=["Network Type"], prefix="net")

# ----------------------------
# Select Features
# ----------------------------
# Automatically grab all one-hot encoded net_ columns
net_cols = [c for c in df.columns if c.startswith("net_")]

features = [
    "Signal Strength (dBm)",
    "Data Throughput (Mbps)",
    "BB60C Measurement (dBm)",
    "srsRAN Measurement (dBm)",
    "BladeRFxA9 Measurement (dBm)",
    "hour",
    "day_of_week",
] + net_cols

X = df[features].copy()
y = df[TARGET].copy()

print(f"✅ Features selected: {len(features)}")
print(f"   → {features}")

# ----------------------------
# Train / Validation / Test Split
# ----------------------------
# First split: 85% train+val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Second split: 70% train, 15% val (0.176 of 85% ≈ 15% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42
)

print(f"\n📊 Split sizes:")
print(f"   Train : {len(X_train):,} rows ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Val   : {len(X_val):,} rows ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test  : {len(X_test):,} rows ({len(X_test)/len(X)*100:.1f}%)")

# ----------------------------
# Feature Scaling (FIT ONLY ON TRAIN)
# ----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Save scaler for later use in model inference
joblib.dump(scaler, f"{DATA_DIR}/scaler.pkl")
print(f"\n💾 Scaler saved → {DATA_DIR}/scaler.pkl")

# Verify scaling worked correctly
scaled_df = pd.DataFrame(X_train_scaled, columns=features)
print(f"\n📐 Scaling check (should be ~mean=0, std=1):")
print(scaled_df.describe().loc[["mean", "std"]].round(2))

# ----------------------------
# Save Processed Data
# ----------------------------
def save_split(X_scaled, y_vals, name):
    split_df = pd.DataFrame(X_scaled, columns=features)
    split_df[TARGET] = y_vals.values
    path = f"{DATA_DIR}/{name}.csv"
    split_df.to_csv(path, index=False)
    print(f"   Saved {name}.csv → {split_df.shape}")

print("\n💾 Saving splits...")
save_split(X_train_scaled, y_train, "train")
save_split(X_val_scaled,   y_val,   "val")
save_split(X_test_scaled,  y_test,  "test")

# Save feature names for use in model.py
with open(f"{DATA_DIR}/feature_names.json", "w") as f:
    json.dump(features, f, indent=4)

print("\n✅ Data preparation complete!")