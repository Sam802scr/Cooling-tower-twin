# twin/engine.py
# This is the core of the digital twin.
# It handles: prediction, rolling trend analysis, early warnings, fault diagnosis.

import numpy as np
from collections import deque

# ── CONSTANTS ────────────────────────────────────────────────────────────────
THRESHOLD = 28.0   # °C — alert if outlet temp crosses this
WINDOW_SIZE = 10     # number of past predictions to track trend on
FORECAST_STEPS = 6      # predict this many steps ahead (6 × 15min = 1.5 hrs)
MIN_WINDOW = 5      # need at least this many readings before forecasting

FEATURES = [
    "Outdoor Temp (°C)",
    "Outdoor Humidity (%)",
    "Wind Speed (m/s)",
    "Water Inlet Temp (°C)",
    "Air Temp (°C)",
    "Water Flow Rate (L/s)",
    "Air Velocity (m/s)",
    "Cooling Tower Efficiency (%)",
    "Cooling Capacity (kW)",
    "Energy Consumption (kWh)",
]

# Baseline "normal" values from dataset statistics (computed during training)
# Used by fault diagnosis to detect which parameter is anomalous
NORMAL_RANGES = {
    "Air Velocity (m/s)":            (1.5, 3.5),
    "Water Flow Rate (L/s)":         (35.0, 55.0),
    "Cooling Tower Efficiency (%)":  (70.0, 90.0),
    "Water Inlet Temp (°C)":         (25.0, 32.0),
    "Outdoor Temp (°C)":             (20.0, 30.0),
}


class DigitalTwin:
    """
    The Digital Twin engine.

    Wraps the trained models and adds:
    - Rolling prediction history
    - Trend-based forecasting (predict future temp BEFORE it breaches threshold)
    - Fault diagnosis using Lasso feature coefficients
    """

    def __init__(self, lasso, elastic, lstm, scaler_X, scaler_y, timesteps=10):
        self.lasso = lasso
        self.elastic = elastic
        self.lstm = lstm
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.timesteps = timesteps

        # Rolling history of raw feature rows — needed for LSTM sequences
        self.feature_buffer = deque(maxlen=timesteps)

        # Rolling history of predicted outlet temperatures — used for trend analysis
        self.prediction_history = deque(maxlen=WINDOW_SIZE)

        self.step = 0  # counts how many readings have been processed

    def ingest(self, raw_row: dict) -> dict:
        """
        Feed one new sensor reading (as a dict) into the twin.
        Returns a full status report.

        raw_row keys must match FEATURES list.
        """
        self.step += 1

        # Extract feature values in correct order
        feature_values = np.array([[raw_row[f] for f in FEATURES]])

        # Scale features
        scaled = self.scaler_X.transform(feature_values)

        # ── Lasso & Elastic Net predictions ─────────────────────────────────
        pred_lasso_s = self.lasso.predict(scaled)[0]
        pred_elastic_s = self.elastic.predict(scaled)[0]

        pred_lasso = self.scaler_y.inverse_transform([[pred_lasso_s]])[0][0]
        pred_elastic = self.scaler_y.inverse_transform([[pred_elastic_s]])[
            0][0]

        # ── LSTM prediction (only once buffer is full) ───────────────────────
        self.feature_buffer.append(scaled[0])
        pred_lstm = None

        if len(self.feature_buffer) == self.timesteps:
            seq = np.array(self.feature_buffer).reshape(
                1, self.timesteps, len(FEATURES))
            pred_lstm_s = self.lstm.predict(seq, verbose=0)[0][0]
            pred_lstm = float(
                self.scaler_y.inverse_transform([[pred_lstm_s]])[0][0])

        # Use LSTM if available, else fall back to Lasso
        primary_pred = pred_lasso
        self.prediction_history.append(primary_pred)

        # ── Status & early warning ───────────────────────────────────────────
        status, warning_msg, forecast_temp, minutes_to_breach = self._assess(
            primary_pred)

        # ── Fault diagnosis ──────────────────────────────────────────────────
        fault = self._diagnose(raw_row, status)

        return {
            "step":              self.step,
            "pred_lasso":        round(pred_lasso, 3),
            "pred_elastic":      round(pred_elastic, 3),
            "pred_lstm":         round(pred_lstm, 3) if pred_lstm else None,
            "primary_pred":      round(primary_pred, 3),
            "forecast_temp":     round(forecast_temp, 3) if forecast_temp else None,
            "minutes_to_breach": round(minutes_to_breach) if minutes_to_breach else None,
            "status":            status,       # "NORMAL" | "WARNING" | "ALERT"
            "warning_msg":       warning_msg,
            "fault":             fault,
            "threshold":         THRESHOLD,
        }

    def _assess(self, current_pred):
        """
        Core predictive maintenance logic.
        Fits a trend line on recent predictions and projects forward.
        """
        forecast_temp = None
        minutes_to_breach = None

        # ── Already breached threshold ───────────────────────────────────────
        if current_pred >= THRESHOLD:
            return "ALERT", f"🔴 ALERT: Outlet temp {current_pred:.1f}°C has crossed {THRESHOLD}°C!", None, None

        # ── Not enough history yet ───────────────────────────────────────────
        if len(self.prediction_history) < MIN_WINDOW:
            return "NORMAL", "🟢 Collecting baseline readings...", None, None

        # ── Fit trend line on recent history ────────────────────────────────
        y_vals = list(self.prediction_history)
        x_vals = list(range(len(y_vals)))
        slope, intercept = np.polyfit(x_vals, y_vals, 1)

        # Project FORECAST_STEPS into the future
        future_x = len(y_vals) - 1 + FORECAST_STEPS
        forecast_temp = intercept + slope * future_x

        # ── Trending toward threshold ────────────────────────────────────────
        if forecast_temp >= THRESHOLD and slope > 0:
            # Estimate exactly when threshold will be crossed
            # threshold = intercept + slope * x  →  x = (threshold - intercept) / slope
            x_breach = (THRESHOLD - intercept) / slope
            steps_remaining = x_breach - (len(y_vals) - 1)
            minutes_to_breach = max(
                0, steps_remaining * 15)  # each step = 15 min

            msg = (
                f"🟡 WARNING: Temperature trending upward. "
                f"Forecast: {forecast_temp:.1f}°C in ~{FORECAST_STEPS * 15} min. "
                f"Threshold breach estimated in ~{minutes_to_breach:.0f} min."
            )
            return "WARNING", msg, forecast_temp, minutes_to_breach

        # ── All clear ────────────────────────────────────────────────────────
        trend_dir = f"({'↑' if slope > 0 else '↓'} {abs(slope):.3f}°C/step)"
        msg = f"🟢 NORMAL — Current: {current_pred:.1f}°C | Trend: {trend_dir} | Forecast: {forecast_temp:.1f}°C"
        return "NORMAL", msg, forecast_temp, minutes_to_breach

    def _diagnose(self, raw_row: dict, status: str) -> str | None:
        """
        If status is WARNING or ALERT, check which parameter is out of normal range.
        Uses physical intuition backed by Lasso coefficient magnitudes.
        """
        if status == "NORMAL":
            return None

        causes = []

        # Fan underperforming → less air → worse cooling → outlet temp rises
        av = raw_row.get("Air Velocity (m/s)", 0)
        lo, hi = NORMAL_RANGES["Air Velocity (m/s)"]
        if av < lo:
            causes.append(
                f"Low air velocity ({av:.2f} m/s < normal {lo}) — fan may be fouled or underperforming")

        # High inlet temp → hardest to cool → outlet rises (coefficient 2.93)
        wi = raw_row.get("Water Inlet Temp (°C)", 0)
        lo, hi = NORMAL_RANGES["Water Inlet Temp (°C)"]
        if wi > hi:
            causes.append(
                f"High water inlet temp ({wi:.1f}°C > normal {hi}°C) — upstream heat load increased")

        # Low efficiency → system degraded
        eff = raw_row.get("Cooling Tower Efficiency (%)", 100)
        lo, hi = NORMAL_RANGES["Cooling Tower Efficiency (%)"]
        if eff < lo:
            causes.append(
                f"Low efficiency ({eff:.1f}% < normal {lo}%) — possible fill media fouling or scaling")

        # High ambient temp → less temperature gradient → cooling worsens
        ot = raw_row.get("Outdoor Temp (°C)", 0)
        lo, hi = NORMAL_RANGES["Outdoor Temp (°C)"]
        if ot > hi:
            causes.append(
                f"High ambient temp ({ot:.1f}°C) — reduced cooling potential")

        if causes:
            return " | ".join(causes)
        return "Multiple parameters in range — monitor closely, possible sensor drift or fill degradation"
