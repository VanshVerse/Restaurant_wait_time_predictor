"""
Restaurant Wait Time Predictor
Dataset: Restaurant Operations Dataset (realistic simulation)
Model: Linear Regression (sklearn)
Target: Predict wait time in minutes
"""

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load ──────────────────────────────────────────────────────
df = pd.read_csv('restaurant_wait.csv')

print("=" * 60)
print("   RESTAURANT WAIT TIME PREDICTOR — LINEAR REGRESSION")
print("=" * 60)
print(f"\nDataset shape : {df.shape}")
print(f"Wait time  →  min: {df['wait_time_min'].min():.1f} min  "
      f"max: {df['wait_time_min'].max():.1f} min  "
      f"mean: {df['wait_time_min'].mean():.1f} min")

# ── 2. Features & Target ─────────────────────────────────────────
FEATURES = [
    'day_of_week', 'hour_of_day', 'party_size',
    'tables_occupied', 'staff_on_duty', 'is_weekend',
    'is_holiday', 'reservations_ahead',
    'avg_service_time_min', 'weather_score', 'occupancy_pct'
]
TARGET = 'wait_time_min'

X = df[FEATURES]
y = df[TARGET]

# ── 3. Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ── 4. Scale ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 5. Train ─────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_s, y_train)

# ── 6. Evaluate ──────────────────────────────────────────────────
y_pred = model.predict(X_test_s)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
cv   = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='r2').mean()

print(f"\n{'─'*44}")
print(f"  MAE   : {mae:.3f} minutes")
print(f"  RMSE  : {rmse:.3f} minutes")
print(f"  R²    : {r2:.4f}  ({r2*100:.1f}%)")
print(f"  CV R² : {cv:.4f}  ({cv*100:.1f}%)")
print(f"{'─'*44}")

# ── 7. Coefficients ──────────────────────────────────────────────
coefs = dict(zip(FEATURES, model.coef_))
print("\nFeature Coefficients (impact on wait time):")
for feat, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
    bar  = "█" * int(abs(coef) / max(abs(v) for v in coefs.values()) * 20)
    sign = "+" if coef > 0 else "-"
    print(f"  {feat:<25} {sign}{abs(coef):.4f}  {bar}")

# ── 8. Sample predictions ────────────────────────────────────────
print("\nSample Predictions vs Actual:")
print(f"  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}")
for a, p in zip(list(y_test[:8]), list(y_pred[:8])):
    print(f"  {a:>8.1f}  {p:>10.1f}  {abs(a-p):>8.1f} min")

# ── 9. Save metadata ─────────────────────────────────────────────
meta = {
    "intercept"   : round(float(model.intercept_), 6),
    "coefficients": {f: round(float(v), 6) for f, v in coefs.items()},
    "scaler_mean" : {f: round(float(v), 6) for f, v in zip(FEATURES, scaler.mean_)},
    "scaler_std"  : {f: round(float(v), 6) for f, v in zip(FEATURES, scaler.scale_)},
    "metrics"     : {"MAE": round(mae,3), "RMSE": round(rmse,3),
                     "R2": round(r2,4), "CV_R2": round(cv,4)},
    "features"    : FEATURES,
    "feature_ranges": {
        f: {"min": round(float(df[f].min()),3),
            "max": round(float(df[f].max()),3),
            "mean": round(float(df[f].mean()),3)}
        for f in FEATURES
    }
}
with open('model_metadata.json', 'w') as fp:
    json.dump(meta, fp, indent=2)

print("\n✅ Saved: model_metadata.json")
