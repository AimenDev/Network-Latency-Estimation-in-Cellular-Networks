import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import build_model

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# ----------------------------
# Config
# ----------------------------
DATA_DIR    = "data"
PLOTS_DIR   = "plots"
TARGET      = "Latency (ms)"
BATCH_SIZE  = 32
MAX_EPOCHS  = 100
LR          = 0.001

os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------
# Load Data
# ----------------------------
print("📂 Loading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df   = pd.read_csv(f"{DATA_DIR}/val.csv")

with open(f"{DATA_DIR}/feature_names.json") as f:
    feature_names = json.load(f)

X_train = train_df[feature_names].values
y_train = train_df[TARGET].values

X_val   = val_df[feature_names].values
y_val   = val_df[TARGET].values

print(f"   Train : {X_train.shape}")
print(f"   Val   : {X_val.shape}")

# ----------------------------
# Build Model
# ----------------------------
print("\n🔧 Building model...")
model = build_model(num_features=len(feature_names), learning_rate=LR)

# ----------------------------
# Callbacks
# ----------------------------
callbacks = [

    # Stop training when val_loss stops improving for 10 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate when stuck for 5 epochs
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),

    # Save best model automatically
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{DATA_DIR}/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    # TensorBoard logs
    tf.keras.callbacks.TensorBoard(
        log_dir="logs",
        histogram_freq=1
    )

]

# ----------------------------
# Train
# ----------------------------
print("\n🚀 Starting training...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ----------------------------
# Save Training History
# ----------------------------
history_df = pd.DataFrame(history.history)
history_df.to_csv(f"{DATA_DIR}/training_history.csv", index=False)
print(f"\n💾 Training history saved → {DATA_DIR}/training_history.csv")

# ----------------------------
# Plot Loss Curve
# ----------------------------
print("📊 Saving plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Training Results", fontsize=14, fontweight="bold")

# Plot 1 — Loss (MSE)
axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True)

# Plot 2 — MAE
axes[1].plot(history.history["mae"],     label="Train MAE")
axes[1].plot(history.history["val_mae"], label="Val MAE")
axes[1].set_title("Mean Absolute Error (ms)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE (ms)")
axes[1].legend()
axes[1].grid(True)

# Plot 3 — RMSE
axes[2].plot(history.history["rmse"],     label="Train RMSE")
axes[2].plot(history.history["val_rmse"], label="Val RMSE")
axes[2].set_title("RMSE (ms)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("RMSE (ms)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/training_curves.png", dpi=150)
plt.close()
print(f"   Saved → {PLOTS_DIR}/training_curves.png")

# ----------------------------
# Final Summary
# ----------------------------
best_epoch = np.argmin(history.history["val_loss"]) + 1
best_val_mae  = min(history.history["val_mae"])
best_val_rmse = min(history.history["val_rmse"])

print(f"""
========================================
✅ Training Complete!
========================================
   Best epoch     : {best_epoch}
   Best Val MAE   : {best_val_mae:.2f} ms
   Best Val RMSE  : {best_val_rmse:.2f} ms
   Model saved    → {DATA_DIR}/best_model.keras
   Plots saved    → {PLOTS_DIR}/training_curves.png
========================================
""")

print("💡 To view TensorBoard run:")
print("   tensorboard --logdir logs")