# app.py  —  run with:  streamlit run app.py

import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from twin.engine import DigitalTwin, FEATURES, THRESHOLD

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cooling Tower Digital Twin",
    page_icon="🏭",
    layout="wide",
)

# ── LOAD MODELS (cached so they don't reload on every rerun) ─────────────────


@st.cache_resource
def load_all():
    lasso = joblib.load("model/lasso.pkl")
    elastic = joblib.load("model/elastic.pkl")
    scaler_X = joblib.load("model/scaler_X.pkl")
    scaler_y = joblib.load("model/scaler_y.pkl")
    return lasso, elastic, scaler_X, scaler_y, None


lasso, elastic, scaler_X, scaler_y, lstm = load_all()

# ── LOAD DATA ────────────────────────────────────────────────────────────────


@st.cache_data
def load_data():
    df = pd.read_csv("data/cooling_tower.csv")
    coef = pd.read_csv("model/coefficients.csv")
    return df, coef


df, coef_df = load_data()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
speed = st.sidebar.slider("Simulation speed (sec/reading)", 0.1, 2.0, 0.5, 0.1)
start_idx = st.sidebar.slider("Start from row", 0, len(df) - 100, 0)
run_sim = st.sidebar.button("▶️  Start Simulation", use_container_width=True)
stop_sim = st.sidebar.button("⏹  Stop", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.markdown(
    "This Digital Twin monitors a cooling tower in real time. "
    "It uses **Lasso**, **Elastic Net**, and **LSTM** to predict outlet temperature, "
    "then applies **trend forecasting** to warn before the threshold is breached."
)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("🏭 Cooling Tower Digital Twin")
st.markdown("**Real-time performance monitoring with predictive maintenance**")
st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📡 Live Simulation", "📊 Model Performance", "🔬 What-If Simulator"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIMULATION
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    # Status banner
    status_box = st.empty()

    col1, col2, col3, col4 = st.columns(4)
    metric_temp = col1.empty()
    metric_forecast = col2.empty()
    metric_breach = col3.empty()
    metric_step = col4.empty()

    st.markdown("#### Predicted Outlet Temperature Over Time")
    chart_area = st.empty()

    st.markdown("#### Fault Diagnosis")
    fault_area = st.empty()

    # ── Session state to persist data across reruns ──────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = {
            "steps": [], "actual": [], "lasso": [],
            "elastic": [], "lstm": [], "forecast": [],
        }
    if "running" not in st.session_state:
        st.session_state.running = False

    if stop_sim:
        st.session_state.running = False

    if run_sim:
        st.session_state.running = True
        st.session_state.history = {
            "steps": [], "actual": [], "lasso": [],
            "elastic": [], "lstm": [], "forecast": [],
        }

        twin = DigitalTwin(lasso, elastic, lstm, scaler_X, scaler_y)

        subset = df.iloc[start_idx:start_idx + 200].reset_index(drop=True)

        for i, row in subset.iterrows():
            if not st.session_state.running:
                break

            raw = {f: row[f] for f in FEATURES}
            result = twin.ingest(raw)
            actual = row["Water Outlet Temp (°C)"]

            h = st.session_state.history
            h["steps"].append(result["step"])
            h["actual"].append(actual)
            h["lasso"].append(result["pred_lasso"])
            h["elastic"].append(result["pred_elastic"])
            h["lstm"].append(result["pred_lstm"])
            h["forecast"].append(result["forecast_temp"])

            # ── Status banner ────────────────────────────────────────────────
            color_map = {"NORMAL": "green",
                         "WARNING": "orange", "ALERT": "red"}
            color = color_map[result["status"]]
            status_box.markdown(
                f'<div style="background:{color};padding:12px;border-radius:8px;'
                f'color:white;font-size:16px;font-weight:bold">{result["warning_msg"]}</div>',
                unsafe_allow_html=True,
            )

            # ── Metrics ──────────────────────────────────────────────────────
            metric_temp.metric("Current Predicted (°C)",
                               f"{result['primary_pred']:.2f}")
            metric_forecast.metric(
                "Forecast (1.5 hrs ahead)",
                f"{result['forecast_temp']:.2f}°C" if result["forecast_temp"] else "—"
            )
            metric_breach.metric(
                "Est. Time to Breach",
                f"{result['minutes_to_breach']:.0f} min" if result["minutes_to_breach"] else "—"
            )
            metric_step.metric("Readings Processed", result["step"])

            # ── Chart ────────────────────────────────────────────────────────
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=h["steps"], y=h["actual"],
                name="Actual", line=dict(color="#2196F3", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=h["steps"], y=h["lasso"],
                name="Lasso", line=dict(color="#4CAF50", dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=h["steps"], y=h["elastic"],
                name="Elastic Net", line=dict(color="#FF9800", dash="dot")
            ))

            lstm_valid = [(s, p) for s, p in zip(
                h["steps"], h["lstm"]) if p is not None]
            if lstm_valid:
                sx, sy = zip(*lstm_valid)
                fig.add_trace(go.Scatter(
                    x=list(sx), y=list(sy),
                    name="LSTM", line=dict(color="#9C27B0", width=2)
                ))

            # Threshold line
            fig.add_hline(
                y=THRESHOLD, line_dash="dash", line_color="red",
                annotation_text=f"Threshold ({THRESHOLD}°C)", annotation_position="top left"
            )

            fig.update_layout(
                xaxis_title="Reading #",
                yaxis_title="Outlet Temperature (°C)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=400, margin=dict(l=0, r=0, t=30, b=0),
            )
            chart_area.plotly_chart(fig, use_container_width=True)

            # ── Fault diagnosis ───────────────────────────────────────────────
            if result["fault"]:
                fault_area.warning(f"🔧 **Diagnosis:** {result['fault']}")
            else:
                fault_area.success("All parameters within normal range.")

            time.sleep(speed)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Evaluation on Test Set")

    results = pd.read_csv("model/test_results.csv")

    from sklearn.metrics import mean_squared_error, r2_score
    metrics = []
    for col in ["Lasso", "Elastic", "LSTM"]:
        valid = results[["Actual", col]].dropna()
        mse = mean_squared_error(valid["Actual"], valid[col])
        r2 = r2_score(valid["Actual"], valid[col])
        metrics.append({"Model": col, "RMSE": round(
            mse**0.5, 4), "R²": round(r2, 4)})

    st.dataframe(pd.DataFrame(metrics), use_container_width=True)

    st.markdown("### Actual vs Predicted (Test Set)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=results["Actual"].values,  name="Actual",  line=dict(color="#2196F3")))
    fig2.add_trace(go.Scatter(y=results["Lasso"].values,   name="Lasso",   line=dict(
        color="#4CAF50", dash="dash")))
    fig2.add_trace(go.Scatter(y=results["Elastic"].values, name="Elastic", line=dict(
        color="#FF9800", dash="dot")))
    fig2.add_hline(y=THRESHOLD, line_dash="dash", line_color="red",
                   annotation_text=f"Threshold ({THRESHOLD}°C)")
    fig2.update_layout(xaxis_title="Test Sample",
                       yaxis_title="Outlet Temp (°C)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Feature Importance (Lasso Coefficients)")
    st.markdown(
        "A higher absolute coefficient means that feature has stronger influence on outlet temperature. "
        "**Water Inlet Temp dominates** — physically intuitive, and matches your thesis result (coef ≈ 2.93)."
    )
    coef_sorted = coef_df.reindex(
        coef_df["Lasso Coef"].abs().sort_values(ascending=True).index)
    fig3 = go.Figure(go.Bar(
        x=coef_sorted["Lasso Coef"],
        y=coef_sorted["Feature"],
        orientation="h",
        marker_color=["#F44336" if v >
                      0 else "#2196F3" for v in coef_sorted["Lasso Coef"]],
    ))
    fig3.update_layout(xaxis_title="Coefficient",
                       height=400, margin=dict(l=0, r=0))
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### What-If Scenario Simulator")
    st.markdown(
        "Adjust sensor parameters and see how the models predict outlet temperature. "
        "This is the **risk-free testing** capability that justifies the Digital Twin label."
    )

    c1, c2 = st.columns(2)
    with c1:
        wi_temp = st.slider("Water Inlet Temp (°C)",   25.0, 35.0, 30.0, 0.1)
        out_temp = st.slider("Outdoor Temp (°C)",        20.0, 30.0, 25.0, 0.1)
        humidity = st.slider("Outdoor Humidity (%)",     40.0, 90.0, 60.0, 1.0)
        wind = st.slider("Wind Speed (m/s)",          1.0,  6.0,  3.0, 0.1)
        air_temp = st.slider("Air Temp (°C)",            20.0, 30.0, 24.0, 0.1)
    with c2:
        flow = st.slider("Water Flow Rate (L/s)",   35.0, 55.0, 45.0, 0.5)
        air_vel = st.slider("Air Velocity (m/s)",       1.0,  4.0,  2.5, 0.1)
        eff = st.slider("Cooling Efficiency (%)",  60.0, 95.0, 80.0, 0.5)
        cap = st.slider("Cooling Capacity (kW)",  400.0, 550.0, 480.0, 1.0)
        energy = st.slider("Energy Consumption (kWh)", 90.0, 140.0, 115.0, 0.5)

    raw_whatif = {
        "Outdoor Temp (°C)":            out_temp,
        "Outdoor Humidity (%)":         humidity,
        "Wind Speed (m/s)":             wind,
        "Water Inlet Temp (°C)":        wi_temp,
        "Air Temp (°C)":                air_temp,
        "Water Flow Rate (L/s)":        flow,
        "Air Velocity (m/s)":           air_vel,
        "Cooling Tower Efficiency (%)": eff,
        "Cooling Capacity (kW)":        cap,
        "Energy Consumption (kWh)":     energy,
    }

    feat_arr = np.array([[raw_whatif[f] for f in FEATURES]])
    scaled = scaler_X.transform(feat_arr)

    p_lasso = scaler_y.inverse_transform(
        lasso.predict(scaled).reshape(-1, 1))[0][0]
    p_elastic = scaler_y.inverse_transform(
        elastic.predict(scaled).reshape(-1, 1))[0][0]

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Lasso Prediction",       f"{p_lasso:.2f} °C",
                 delta=f"{'⚠️ Above threshold!' if p_lasso >= THRESHOLD else 'Within safe range'}")
    col_b.metric("Elastic Net Prediction", f"{p_elastic:.2f} °C",
                 delta=f"{'⚠️ Above threshold!' if p_elastic >= THRESHOLD else 'Within safe range'}")
    col_c.metric("Threshold", f"{THRESHOLD} °C")

    if p_lasso >= THRESHOLD:
        st.error(f"🔴 These conditions would cause outlet temp to exceed {THRESHOLD}°C! "
                 "Try increasing air velocity or reducing water inlet temperature.")
    elif p_lasso >= THRESHOLD - 2:
        st.warning(
            f"🟡 Close to threshold. Monitor carefully under these conditions.")
    else:
        st.success("🟢 These conditions are within safe operating range.")
