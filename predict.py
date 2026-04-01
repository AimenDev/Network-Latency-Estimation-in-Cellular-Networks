import json
import joblib
import pandas as pd
import tensorflow as tf

# ----------------------------
# Load saved artifacts
# ----------------------------
model  = tf.keras.models.load_model("data/best_model.keras")
scaler = joblib.load("data/scaler.pkl")

with open("data/feature_names.json") as f:
    feature_names = json.load(f)

# ----------------------------
# Prediction Function
# ----------------------------
def predict_latency(
    signal_strength_dbm,
    throughput_mbps,
    bb60c_dbm,
    srsran_dbm,
    bladerfxa9_dbm,
    network_type,
    hour=12,
    day_of_week=0
):
    """
    Predict network latency given cellular network conditions.

    Parameters:
        signal_strength_dbm  : float  e.g. -85.0
        throughput_mbps      : float  e.g. 25.0
        bb60c_dbm            : float  e.g. -90.0
        srsran_dbm           : float  e.g. -95.0
        bladerfxa9_dbm       : float  e.g. -88.0
        network_type         : str    "3G" / "4G" / "5G" / "LTE"
        hour                 : int    0-23
        day_of_week          : int    0=Monday ... 6=Sunday

    Returns:
        Predicted latency in milliseconds (float)
    """

    # Build one-hot for network type
    net_3G  = 1 if network_type == "3G"  else 0
    net_4G  = 1 if network_type == "4G"  else 0
    net_5G  = 1 if network_type == "5G"  else 0
    net_LTE = 1 if network_type == "LTE" else 0

    # Assemble as DataFrame (fixes sklearn feature name warning)
    sample = pd.DataFrame([[
        signal_strength_dbm,
        throughput_mbps,
        bb60c_dbm,
        srsran_dbm,
        bladerfxa9_dbm,
        hour,
        day_of_week,
        net_3G,
        net_4G,
        net_5G,
        net_LTE
    ]], columns=feature_names)

    # Scale using the same scaler from training
    sample_scaled = scaler.transform(sample)

    # Predict
    prediction = model.predict(sample_scaled, verbose=0)
    return float(prediction[0][0])


# ----------------------------
# Demo — try different scenarios
# ----------------------------
if __name__ == "__main__":

    print("=" * 50)
    print("  Network Latency Estimator")
    print("=" * 50)

    scenarios = [
        {
            "name"               : "5G — Real Measurement",
            "signal_strength_dbm": -94.14315859405015,
            "throughput_mbps"    : 68.59693229517501,
            "bb60c_dbm"          : -90.64277281615216,
            "srsran_dbm"         : -101.89590522335914,
            "bladerfxa9_dbm"     : -96.57069789510768,
            "network_type"       : "5G",
        },
        {
            "name"               : "4G — Medium Signal",
            "signal_strength_dbm": -90.0,
            "throughput_mbps"    : 25.0,
            "bb60c_dbm"          : -92.0,
            "srsran_dbm"         : -95.0,
            "bladerfxa9_dbm"     : -91.0,
            "network_type"       : "4G",
        },
        {
            "name"               : "3G — Weak Signal",
            "signal_strength_dbm": -110.0,
            "throughput_mbps"    : 2.0,
            "bb60c_dbm"          : -108.0,
            "srsran_dbm"         : -112.0,
            "bladerfxa9_dbm"     : -109.0,
            "network_type"       : "3G",
        },
        {
            "name"               : "LTE — Medium Signal",
            "signal_strength_dbm": -88.0,
            "throughput_mbps"    : 15.0,
            "bb60c_dbm"          : -90.0,
            "srsran_dbm"         : -93.0,
            "bladerfxa9_dbm"     : -89.0,
            "network_type"       : "LTE",
        },
    ]

    for s in scenarios:
        latency = predict_latency(
            signal_strength_dbm = s["signal_strength_dbm"],
            throughput_mbps     = s["throughput_mbps"],
            bb60c_dbm           = s["bb60c_dbm"],
            srsran_dbm          = s["srsran_dbm"],
            bladerfxa9_dbm      = s["bladerfxa9_dbm"],
            network_type        = s["network_type"],
        )
        print(f"\n📡 Scenario  : {s['name']}")
        print(f"   Network   : {s['network_type']}")
        print(f"   Signal    : {s['signal_strength_dbm']} dBm")
        print(f"   Speed     : {s['throughput_mbps']} Mbps")
        print(f"   Predicted : {latency:.2f} ms")

    print("\n" + "=" * 50)
    print("✅ Done! Model is ready for deployment.")
    print("=" * 50)