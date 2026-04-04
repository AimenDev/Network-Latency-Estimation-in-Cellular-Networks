import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Config
# ----------------------------
DATA_DIR  = "data"
PLOTS_DIR = "plots"
TARGET    = "Latency (ms)"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------
# Load Test Data
# ----------------------------
print("📂 Loading test data...")
test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")

with open(f"{DATA_DIR}/feature_names.json") as f:
    feature_names = json.load(f)

X_test  = test_df[feature_names].values
y_test  = test_df[TARGET].values
X_train = train_df[feature_names].values
y_train = train_df[TARGET].values

print(f"   Test samples  : {len(X_test)}")
print(f"   Train samples : {len(X_train)}")

# ----------------------------
# Load & Evaluate Neural Network
# ----------------------------
print("\n🧠 Evaluating Neural Network...")
nn_model  = tf.keras.models.load_model(f"{DATA_DIR}/best_model.keras")
y_pred_nn = nn_model.predict(X_test, verbose=0).flatten()

mae_nn  = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn   = r2_score(y_test, y_pred_nn)

print(f"   MAE  : {mae_nn:.2f} ms")
print(f"   RMSE : {rmse_nn:.2f} ms")
print(f"   R²   : {r2_nn:.4f}")

# ----------------------------
# Baseline 1 — Linear Regression
# ----------------------------
print("\n📏 Evaluating Linear Regression baseline...")
lr_model  = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)

print(f"   MAE  : {mae_lr:.2f} ms")
print(f"   RMSE : {rmse_lr:.2f} ms")
print(f"   R²   : {r2_lr:.4f}")

# ----------------------------
# Baseline 2 — Random Forest
# ----------------------------
print("\n🌲 Evaluating Random Forest baseline...")
rf_model  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)

print(f"   MAE  : {mae_rf:.2f} ms")
print(f"   RMSE : {rmse_rf:.2f} ms")
print(f"   R²   : {r2_rf:.4f}")

# ----------------------------
# Comparison Table
# ----------------------------
results = pd.DataFrame({
    "Model"    : ["Neural Network", "Linear Regression", "Random Forest"],
    "MAE (ms)" : [round(mae_nn,2),  round(mae_lr,2),     round(mae_rf,2)],
    "RMSE (ms)": [round(rmse_nn,2), round(rmse_lr,2),    round(rmse_rf,2)],
    "R² Score" : [round(r2_nn,4),   round(r2_lr,4),      round(r2_rf,4)],
})

print("\n" + "="*55)
print("📊 MODEL COMPARISON")
print("="*55)
print(results.to_string(index=False))
print("="*55)

results.to_csv(f"{DATA_DIR}/model_comparison.csv", index=False)

# ----------------------------
# Plots — 6 Panel Evaluation
# ----------------------------
print("\n📊 Generating evaluation plots...")
errors_nn = y_test - y_pred_nn
errors_rf = y_test - y_pred_rf
errors_lr = y_test - y_pred_lr

fig = plt.figure(figsize=(20, 14))
fig.suptitle("Network Latency Estimation — Evaluation Results",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1 — Predicted vs Actual (Neural Network)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred_nn, alpha=0.3, s=10, color="steelblue")
ax1.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")
ax1.set_title("Neural Network\nPredicted vs Actual")
ax1.set_xlabel("Actual Latency (ms)")
ax1.set_ylabel("Predicted Latency (ms)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2 — Predicted vs Actual (Random Forest)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, y_pred_rf, alpha=0.3, s=10, color="forestgreen")
ax2.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")
ax2.set_title("Random Forest\nPredicted vs Actual")
ax2.set_xlabel("Actual Latency (ms)")
ax2.set_ylabel("Predicted Latency (ms)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3 — Predicted vs Actual (Linear Regression)
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(y_test, y_pred_lr, alpha=0.3, s=10, color="darkorange")
ax3.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")
ax3.set_title("Linear Regression\nPredicted vs Actual")
ax3.set_xlabel("Actual Latency (ms)")
ax3.set_ylabel("Predicted Latency (ms)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4 — Error Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(errors_nn, bins=50, color="steelblue",   alpha=0.6, label=f"NN (MAE={mae_nn:.1f}ms)")
ax4.hist(errors_rf, bins=50, color="forestgreen", alpha=0.6, label=f"RF (MAE={mae_rf:.1f}ms)")
ax4.hist(errors_lr, bins=50, color="darkorange",  alpha=0.6, label=f"LR (MAE={mae_lr:.1f}ms)")
ax4.axvline(0, color="red", linestyle="--", lw=2)
ax4.set_title("Error Distribution\n(Actual - Predicted)")
ax4.set_xlabel("Error (ms)")
ax4.set_ylabel("Count")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5 — MAE Comparison Bar Chart
ax5 = fig.add_subplot(gs[1, 1])
models_list = ["Neural\nNetwork", "Random\nForest", "Linear\nRegression"]
mae_values  = [mae_nn, mae_rf, mae_lr]
colors      = ["steelblue", "forestgreen", "darkorange"]
bars = ax5.bar(models_list, mae_values, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, mae_values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax5.set_title("MAE Comparison\n(lower is better)")
ax5.set_ylabel("MAE (ms)")
ax5.grid(True, alpha=0.3, axis="y")

# Plot 6 — R² Score Comparison
ax6 = fig.add_subplot(gs[1, 2])
r2_values = [r2_nn, r2_rf, r2_lr]
bars2 = ax6.bar(models_list, r2_values, color=colors, alpha=0.85,
                edgecolor="black", linewidth=0.8)
for bar, val in zip(bars2, r2_values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax6.set_title("R² Score Comparison\n(higher is better)")
ax6.set_ylabel("R² Score")
ax6.set_ylim(0, 1.05)
ax6.grid(True, alpha=0.3, axis="y")

plt.savefig(f"{PLOTS_DIR}/evaluation_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {PLOTS_DIR}/evaluation_results.png")

# ----------------------------
# Sample Predicted vs Actual Table
# ----------------------------
print("\n📋 Generating sample prediction table...")

def recover_network_type(row):
    if row["net_5G"]  > 0.5: return "5G"
    if row["net_4G"]  > 0.5: return "4G"
    if row["net_LTE"] > 0.5: return "LTE"
    if row["net_3G"]  > 0.5: return "3G"
    return "Unknown"

test_df["Network Type"]      = test_df.apply(recover_network_type, axis=1)
test_df["Actual (ms)"]       = y_test.round(2)
test_df["NN Predicted (ms)"] = y_pred_nn.round(2)
test_df["Error (ms)"]        = (y_pred_nn - y_test).round(2)
test_df["Abs Error (ms)"]    = abs(y_pred_nn - y_test).round(2)

# Pick 5 samples per network type
sample_table = (
    test_df.groupby("Network Type", group_keys=False)
    .apply(lambda g: g.sample(5, random_state=42))
    .reset_index(drop=True)
)[["Network Type", "Actual (ms)", "NN Predicted (ms)",
   "Error (ms)", "Abs Error (ms)"]].sort_values("Network Type").reset_index(drop=True)

# Print to terminal
print("\n" + "="*65)
print("📋 SAMPLE PREDICTED VS ACTUAL (5 per Network Type)")
print("="*65)
print(sample_table.to_string(index=False))
print("="*65)

# Save as CSV
sample_table.to_csv(f"{DATA_DIR}/sample_predictions.csv", index=False)
print(f"\n💾 Saved → {DATA_DIR}/sample_predictions.csv")

# Save as Image
fig_table, ax_table = plt.subplots(figsize=(12, 6))
ax_table.axis("off")

table = ax_table.table(
    cellText=sample_table.values,
    colLabels=sample_table.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Header row styling
for col_idx in range(len(sample_table.columns)):
    table[0, col_idx].set_facecolor("#1F4E79")
    table[0, col_idx].set_text_props(color="white", fontweight="bold")

# Row colors by network type
row_colors = {
    "5G" : "#D6EAF8",
    "4G" : "#D5F5E3",
    "LTE": "#FDEBD0",
    "3G" : "#FADBD8"
}
for row_idx in range(1, len(sample_table) + 1):
    nt = sample_table.iloc[row_idx - 1]["Network Type"]
    for col_idx in range(len(sample_table.columns)):
        table[row_idx, col_idx].set_facecolor(row_colors.get(nt, "white"))

fig_table.suptitle(
    "Sample Predicted vs Actual Latency (5 per Network Type)",
    fontsize=13, fontweight="bold", y=0.98
)

plt.savefig(f"{PLOTS_DIR}/sample_predictions_table.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {PLOTS_DIR}/sample_predictions_table.png")

# ----------------------------
# Final Summary
# ----------------------------
best_model = results.loc[results["MAE (ms)"].idxmin(), "Model"]
print(f"""
========================================
✅ Evaluation Complete!
========================================
   Best model     : {best_model}
   Results saved  → {DATA_DIR}/model_comparison.csv
   Plot 1 saved   → {PLOTS_DIR}/evaluation_results.png
   Plot 2 saved   → {PLOTS_DIR}/sample_predictions_table.png
========================================
""")