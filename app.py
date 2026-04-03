import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ----------------------------
# 1. Page Config & Custom CSS
# ----------------------------
st.set_page_config(
    page_title="Network Latency AI",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stExpander"] {
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# 2. Load Artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model  = tf.keras.models.load_model("data/best_model.keras")
        scaler = joblib.load("data/scaler.pkl")
        with open("data/feature_names.json") as f:
            feature_names = json.load(f)
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

# Load the artifacts once
model, scaler, feature_names = load_artifacts()

# ----------------------------
# 3. Helper Functions
# ----------------------------
def predict_latency(signal_strength, throughput, bb60c, srsran, bladerfxa9,
                    network_type, hour=12, day_of_week=0):
    
    # Mapping for one-hot encoding
    net_map = {"3G": [1,0,0,0], "4G": [0,1,0,0], "5G": [0,0,1,0], "LTE": [0,0,0,1]}
    encoding = net_map.get(network_type, [0,0,0,0])

    # Ensure feature order matches the trained model
    sample = pd.DataFrame([[
        signal_strength, throughput, bb60c, srsran, bladerfxa9,
        hour, day_of_week] + encoding
    ], columns=feature_names)

    scaled = scaler.transform(sample)
    prediction = model.predict(scaled, verbose=0)
    return float(prediction[0][0])

def get_quality_info(latency):
    if latency < 40:
        return "Excellent", "🟢", "Your connection is lightning fast. Perfect for cloud gaming and VR.", "green"
    elif latency < 80:
        return "Good", "🟡", "Responsive enough for HD video calls and smooth streaming.", "blue"
    elif latency < 130:
        return "Fair", "🟠", "Functional for browsing, but expect minor delays in gaming.", "orange"
    else:
        return "Poor", "🔴", "Significant lag detected. Unsuitable for real-time apps.", "red"

def get_network_defaults(network_type):
    defaults = {
        "5G" : {"signal": -88.0, "throughput": 65.0, "bb60c": -90.0,  "srsran": -95.0,  "blade": -91.0},
        "4G" : {"signal": -92.0, "throughput": 25.0, "bb60c": -93.0,  "srsran": -98.0,  "blade": -94.0},
        "LTE": {"signal": -95.0, "throughput": 15.0, "bb60c": -96.0,  "srsran": -100.0, "blade": -97.0},
        "3G" : {"signal": -105.0,"throughput": 2.0,  "bb60c": -106.0, "srsran": -110.0, "blade": -107.0},
    }
    return defaults[network_type]

# ----------------------------
# 4. UI — Sidebar
# ----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/antenna.png", width=80)
    st.title("Project Dashboard")
    st.info("**Model Accuracy:** 84.3% ($R^2$)\n\n**MAE:** 18.0 ms")
    
    st.divider()
    st.markdown("### Settings")
    # Added unique key for the sidebar slider
    sim_hour = st.slider("Simulated Hour", 0, 23, 12, key="side_hour")
    sim_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=0)
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

# ----------------------------
# 5. UI — Main Content
# ----------------------------
st.title("📡 Neural Latency Estimator")
st.write("An intelligent system for real-time cellular network performance estimation.")

tab1, tab2 = st.tabs(["🚀 Latency Predictor", "📊 Benchmark & Comparison"])

with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("Input Measurements")
        network_type = st.radio("Select Network Technology", ["5G", "4G", "LTE", "3G"], horizontal=True)
        
        defaults = get_network_defaults(network_type)
        
        with st.container(border=True):
            st.markdown("**Core Metrics**")
            # Added unique key for the main signal slider
            signal_strength = st.slider("Signal Strength (dBm)", -120.0, -50.0, defaults["signal"], step=0.1, key="main_signal")
            throughput = st.number_input("Throughput (Mbps)", 0.1, 500.0, defaults["throughput"])
        
        with st.expander("Advanced Hardware Sensors"):
            st.caption("Readings from professional SDR equipment")
            bb60c = st.number_input("BB60C (dBm)", -120.0, 0.0, defaults["bb60c"])
            srsran = st.number_input("srsRAN (dBm)", -120.0, 0.0, defaults["srsran"])
            bladerfxa9 = st.number_input("BladeRF (dBm)", -120.0, 0.0, defaults["blade"])

        predict_btn = st.button("Calculate Latency", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Estimation Result")
        if predict_btn:
            with st.spinner("Analyzing signals..."):
                lat = predict_latency(signal_strength, throughput, bb60c, srsran, bladerfxa9, network_type, sim_hour, day_map[sim_day])
                label, icon, desc, color = get_quality_info(lat)
                
                st.metric("Estimated Latency", f"{lat:.2f} ms", delta=f"{lat - 101:.1f} ms from avg", delta_color="inverse")
                
                norm_lat = min(lat / 200, 1.0)
                st.write(f"{icon} **Quality Status: {label}**")
                st.progress(norm_lat)
                st.success(desc)
                
                chart_data = pd.DataFrame({
                    "Category": ["Predicted", "Avg (101ms)"],
                    "Latency (ms)": [lat, 101]
                })
                st.bar_chart(chart_data, x="Category", y="Latency (ms)", color="#4A90E2")
        else:
            st.info("Adjust parameters and click 'Calculate' to see the AI estimation.")

with tab2:
    st.subheader("Technology Comparison")
    st.write("Simulating current signal parameters across all network generations.")
    
    if st.button("Run Benchmarks", use_container_width=True):
        results = []
        for nt in ["5G", "4G", "LTE", "3G"]:
            d = get_network_defaults(nt)
            l = predict_latency(d["signal"], d["throughput"], d["bb60c"], d["srsran"], d["blade"], nt, sim_hour, day_map[sim_day])
            results.append({"Type": nt, "Latency (ms)": round(l, 2), "Status": get_quality_info(l)[0]})
        
        comp_df = pd.DataFrame(results)
        st.table(comp_df)
        st.line_chart(comp_df.set_index("Type")["Latency (ms)"])

st.divider()
st.caption("Network Latency Estimation Project 2026")