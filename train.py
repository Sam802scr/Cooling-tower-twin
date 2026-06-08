# train.py
# Run this ONCE to train all models and save them.
# After this, the dashboard uses the saved models directly.

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ── 1. LOAD DATA ────────────────────────────────────────────────────────────
df = pd.read_csv("data/cooling_tower.csv")
df = df.sort_values("Timestamp").reset_index(drop=True)
# These are the same 10 features from your thesis
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
TARGET = "Water Outlet Temp (°C)"

X = df[FEATURES].values
y = df[TARGET].values

# ── 2. SPLIT ────────────────────────────────────────────────────────────────
# shuffle=False keeps time order — important for a time-series system
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ── 3. SCALE ────────────────────────────────────────────────────────────────
scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ── 4. LASSO ────────────────────────────────────────────────────────────────
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_s, y_train_s)
y_pred_lasso = scaler_y.inverse_transform(
    lasso.predict(X_test_s).reshape(-1, 1)).ravel()

mse_l = mean_squared_error(y_test, y_pred_lasso)
r2_l = r2_score(y_test, y_pred_lasso)
print(f"Lasso   → RMSE: {mse_l**0.5:.4f}  R²: {r2_l:.4f}")

# ── 5. ELASTIC NET ───────────────────────────────────────────────────────────
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic.fit(X_train_s, y_train_s)
y_pred_elastic = scaler_y.inverse_transform(
    elastic.predict(X_test_s).reshape(-1, 1)).ravel()

mse_e = mean_squared_error(y_test, y_pred_elastic)
r2_e = r2_score(y_test, y_pred_elastic)
print(f"Elastic → RMSE: {mse_e**0.5:.4f}  R²: {r2_e:.4f}")

# ── 6. LSTM ──────────────────────────────────────────────────────────────────
# KEY UPGRADE FROM YOUR THESIS: timesteps=10 instead of 1
# The model now looks at the last 10 readings (2.5 hours) before predicting
TIMESTEPS = 10


def make_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i - timesteps:i])   # last `timesteps` rows
        ys.append(y[i])                  # the value we want to predict
    return np.array(Xs), np.array(ys)


X_seq, y_seq = make_sequences(X_train_s, y_train_s, TIMESTEPS)
X_seq_test, y_seq_test = make_sequences(X_test_s, y_test_s, TIMESTEPS)

model_lstm = Sequential([
    LSTM(32, input_shape=(TIMESTEPS, len(FEATURES))),
    Dense(16, activation="relu"),
    Dense(1),
])
model_lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])
model_lstm.summary()

history = model_lstm.fit(
    X_seq, y_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
)

y_pred_lstm_s = model_lstm.predict(X_seq_test).ravel()
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_s.reshape(-1, 1)).ravel()
y_test_lstm = scaler_y.inverse_transform(y_seq_test.reshape(-1, 1)).ravel()

mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
r2_lstm = r2_score(y_test_lstm, y_pred_lstm)
print(f"LSTM    → RMSE: {mse_lstm**0.5:.4f}  R²: {r2_lstm:.4f}")

# ── 7. SAVE EVERYTHING ───────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

joblib.dump(lasso,     "model/lasso.pkl")
joblib.dump(elastic,   "model/elastic.pkl")
joblib.dump(scaler_X,  "model/scaler_X.pkl")
joblib.dump(scaler_y,  "model/scaler_y.pkl")
model_lstm.save("model/lstm.keras")

# Save feature names and coefficients for the dashboard
coef_df = pd.DataFrame({
    "Feature": FEATURES,
    "Lasso Coef":   lasso.coef_,
    "Elastic Coef": elastic.coef_,
})
coef_df.to_csv("model/coefficients.csv", index=False)

# Save test predictions for the dashboard's initial load
# Align LSTM predictions with test set (LSTM loses first `timesteps` rows)
lstm_full = np.full(len(y_test), np.nan)
lstm_full[TIMESTEPS:] = y_pred_lstm

results_df = pd.DataFrame({
    "Actual":   y_test,
    "Lasso":    y_pred_lasso,
    "Elastic":  y_pred_elastic,
    "LSTM":     lstm_full,
})
results_df.to_csv("model/test_results.csv", index=False)

print("\n✅ All models saved to model/")
print(f"   Features used : {FEATURES}")
print(
    f"   LSTM timesteps: {TIMESTEPS} (each step = 15 min → 2.5 hrs of history)")
