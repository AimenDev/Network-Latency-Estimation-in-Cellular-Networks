import json
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# ----------------------------
# Load feature names (auto-detects input size)
# ----------------------------
with open("data/feature_names.json", "r") as f:
    feature_names = json.load(f)

NUM_FEATURES = len(feature_names)

# ----------------------------
# Build Model
# ----------------------------
def build_model(num_features=NUM_FEATURES, learning_rate=0.001):
    """
    Feedforward Neural Network for Latency Regression.
    
    Architecture:
        Input(11) → Dense(128) → Dense(64) → Dense(32) → Output(1)
    
    Each hidden layer uses:
        - ReLU activation
        - Batch Normalization (stabilizes training)
        - Dropout (prevents overfitting)
    """

    model = models.Sequential([

        # --- Input Layer ---
        layers.Input(shape=(num_features,), name="input"),

        # --- Hidden Layer 1 ---
        layers.Dense(
            128,
            kernel_regularizer=regularizers.l2(1e-4),
            name="dense_1"
        ),
        layers.BatchNormalization(name="bn_1"),
        layers.Activation("relu", name="relu_1"),
        layers.Dropout(0.3, name="dropout_1"),

        # --- Hidden Layer 2 ---
        layers.Dense(
            64,
            kernel_regularizer=regularizers.l2(1e-4),
            name="dense_2"
        ),
        layers.BatchNormalization(name="bn_2"),
        layers.Activation("relu", name="relu_2"),
        layers.Dropout(0.2, name="dropout_2"),

        # --- Hidden Layer 3 ---
        layers.Dense(
            32,
            activation="relu",
            name="dense_3"
        ),

        # --- Output Layer (single value = latency in ms) ---
        layers.Dense(1, name="output")

    ], name="LatencyEstimator")

    # --- Compile ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",                        # Mean Squared Error for regression
        metrics=[
            "mae",                         # Mean Absolute Error (in ms)
            tf.keras.metrics.RootMeanSquaredError(name="rmse")
        ]
    )

    return model


# ----------------------------
# Quick test when run directly
# ----------------------------
if __name__ == "__main__":
    model = build_model()
    model.summary()

    print(f"\n📐 Input size  : {NUM_FEATURES} features")
    print(f"📋 Features    : {feature_names}")
    print(f"\n✅ Model built successfully — ready for training!")