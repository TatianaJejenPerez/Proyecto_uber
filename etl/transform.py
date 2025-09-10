# etl/transform.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------- Limpieza base / Normalización ----------
COLMAP = {
    "Date": "date",
    "Time": "time",
    "Booking ID": "booking_id",
    "Booking Status": "booking_status",
    "Customer ID": "customer_id",
    "Vehicle Type": "vehicle_type",
    "Pickup Location": "pickup_location",
    "Drop Location": "drop_location",
    "Avg VTAT": "avg_vtat_min",  # llegada conductor a pickup (min)
    "Avg CTAT": "avg_ctat_min",  # duración viaje (min)
    "Cancelled Rides by Customer": "cancelled_by_customer",
    "Reason for cancelling by Customer": "customer_cancel_reason",
    "Cancelled Rides by Driver": "cancelled_by_driver",
    "Driver Cancellation Reason": "driver_cancel_reason",
    "Incomplete Rides": "incomplete_ride",
    "Incomplete Rides Reason": "incomplete_reason",
    "Booking Value": "booking_value",
    "Ride Distance": "ride_distance_km",
    "Driver Ratings": "driver_ratings",
    "Customer Rating": "customer_rating",
}

NUMERIC_BOUNDS = {
    "ride_distance_km": (0, 60),
    "booking_value":   (0, 2000),
    "avg_ctat_min":    (0, 240),   # duración
}

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    d.rename(columns=COLMAP, inplace=True)
    
    d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
    return d

def _parse_datetime(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    # parse date
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    # parse time (puede venir HH:MM o HH:MM:SS)
    if "time" in out.columns:
        t = pd.to_datetime(out["time"], format="%H:%M:%S", errors="coerce")
        na = t.isna()
        if na.any():
            t2 = pd.to_datetime(out.loc[na, "time"], format="%H:%M", errors="coerce")
            t.loc[na] = t2
        out["time"] = t.dt.time
    # ts_exact = date + time
    if "date" in out.columns and "time" in out.columns:
        out["ts_exact"] = pd.to_datetime(out["date"].astype(str) + " " + out["time"].astype(str), errors="coerce")
        out["ts_hour"]  = out["ts_exact"].dt.floor("h")
        out["hour"]     = out["ts_exact"].dt.hour
        out["dow"]      = out["ts_exact"].dt.weekday
        out["is_weekend"] = (out["dow"] >= 5).astype(int)
        out["month"]    = out["ts_exact"].dt.month
        out["year"]     = out["ts_exact"].dt.year
    return out

def clean_for_ml(raw: pd.DataFrame) -> pd.DataFrame:
    d = _standardize_columns(raw)
    d = _parse_datetime(d)

    # tipificar numéricos
    for c in ["ride_distance_km","booking_value","avg_ctat_min","avg_vtat_min","driver_ratings","customer_rating","hour","dow"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # limpieza de texto
    for c in ["pickup_location","drop_location","vehicle_type","booking_status"]:
        if c in d.columns:
            d[c] = d[c].astype("string").str.strip()

    # filtrar rows con fecha/hora válidas
    d = d.dropna(subset=["ts_exact","ts_hour","hour","dow"])

    # crear un id de viaje reproducible si no existe
    if "booking_id" not in d.columns:
        d["booking_id"] = d.index.astype(str)

    # métricas core: tarifa y duración
    # booking_value (tarifa), avg_ctat_min (duración)
    # Filtrado razonable para ML
    d = d[(d["ride_distance_km"] > NUMERIC_BOUNDS["ride_distance_km"][0]) &
          (d["ride_distance_km"] <= NUMERIC_BOUNDS["ride_distance_km"][1])]
    d = d[(d["booking_value"]   >= NUMERIC_BOUNDS["booking_value"][0]) &
          (d["booking_value"]   <= NUMERIC_BOUNDS["booking_value"][1])]
    d = d[(d["avg_ctat_min"]    >  NUMERIC_BOUNDS["avg_ctat_min"][0]) &
          (d["avg_ctat_min"]    <= NUMERIC_BOUNDS["avg_ctat_min"][1])]

    # Solo viajes completados para modelar (evita cancelados)
    if "booking_status" in d.columns:
        d = d[d["booking_status"].str.lower() == "completed"]

    # generar trip_id incremental (estable)
    d = d.sort_values("ts_exact").reset_index(drop=True)
    d["trip_id"] = (d.index + 1).astype("int64")

    print(f"🧼 Clean ML → filas: {len(d)}  | cols: {len(d.columns)}")
    return d

# ---------- Opción A: demanda por hora (autoregresivo sencillo) ----------
def _make_hourly_trips(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("ts_hour", as_index=False)
          .agg(trips=("booking_id","count"))
          .sort_values("ts_hour")
          .reset_index(drop=True)
    )
    # features calendario
    g["hour"] = g["ts_hour"].dt.hour
    g["dow"]  = g["ts_hour"].dt.weekday
    g["is_weekend"] = (g["dow"] >= 5).astype(int)
    g["month"] = g["ts_hour"].dt.month
    return g

def _add_lags_rolls(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()
    out["trips_lag24"]  = out["trips"].shift(24)
    out["trips_lag168"] = out["trips"].shift(168)
    out["trips_roll24"] = out["trips"].rolling(24, min_periods=1).mean()
    return out

def train_demand_model(df_clean: pd.DataFrame):
    hist = _make_hourly_trips(df_clean)
    hist = _add_lags_rolls(hist)

    feats = ["hour","dow","is_weekend","month","trips_lag24","trips_lag168","trips_roll24"]
    dtrain = hist.dropna(subset=feats + ["trips"]).copy()
    if dtrain.empty:
        raise ValueError("No hay suficientes datos con lags para entrenar demanda.")

    X = dtrain[feats].values
    y = dtrain["trips"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)

    metrics = {"mae": float(mean_absolute_error(yte, yhat)), "r2": float(r2_score(yte, yhat))}
    print("🔷 Demanda/Hora → MAE: {:.2f} | R²: {:.3f}".format(metrics["mae"], metrics["r2"]))

    # guardamos historia con columnas para forecasting
    return model, metrics, hist

def forecast_next_hours(hist: pd.DataFrame, model, horizon_hours: int = 168) -> pd.DataFrame:
    """Pronóstico iterativo 1-step-ahead usando lags de la serie."""
    preds = []
    cur = hist.copy()

    last_ts = cur["ts_hour"].max()
    for h in range(1, horizon_hours + 1):
        ts_next = last_ts + pd.Timedelta(hours=h)

        # construir fila futura
        row = {
            "ts_hour": ts_next,
            "hour": ts_next.hour,
            "dow": ts_next.weekday(),
            "is_weekend": 1 if ts_next.weekday() >= 5 else 0,
            "month": ts_next.month,
        }
        # lags desde el cur que se va actualizando con predicciones
        # Para obtener lag24/168 y roll24 necesitamos que cur tenga trips hasta ts_next-1
        cur_ext = pd.concat([cur, pd.DataFrame({"ts_hour":[ts_next]})], ignore_index=True).sort_values("ts_hour")
        cur_ext["trips_lag24"]  = cur_ext["trips"].shift(24)
        cur_ext["trips_lag168"] = cur_ext["trips"].shift(168)
        cur_ext["trips_roll24"] = cur_ext["trips"].rolling(24, min_periods=1).mean()

        future_row = cur_ext[cur_ext["ts_hour"] == ts_next].iloc[0]
        row["trips_lag24"]  = future_row.get("trips_lag24", np.nan)
        row["trips_lag168"] = future_row.get("trips_lag168", np.nan)
        row["trips_roll24"] = future_row.get("trips_roll24", np.nan)

        X = pd.DataFrame([row])[["hour","dow","is_weekend","month","trips_lag24","trips_lag168","trips_roll24"]]
        # si faltan lags (inicio de serie), usa fallback: media reciente
        if X.isna().any(axis=None):
            fill = cur["trips"].tail(24).mean() if len(cur) >= 1 else 1.0
            X = X.fillna(fill)

        yhat = float(model.predict(X.values)[0])
        yhat = max(0.0, yhat)
        preds.append({"ts_hour": ts_next, "trips_pred": yhat})

        # actualizar serie cur agregando la predicción como si fuera observación
        cur = pd.concat([cur, pd.DataFrame({"ts_hour":[ts_next], "trips":[yhat]})], ignore_index=True)

    return pd.DataFrame(preds)

# ---------- Opción B: tarifa y duración ----------
def train_and_predict_trip_models(df_clean: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    needed = ["trip_id","booking_id","ride_distance_km","booking_value","avg_ctat_min","hour","dow"]
    miss = [c for c in needed if c not in df_clean.columns]
    if miss:
        raise ValueError(f"Faltan columnas para Opción B: {miss}")

    d = df_clean.dropna(subset=needed).copy()
    # features
    X = d[["ride_distance_km","hour","dow"]].astype(float)
    y_fare = d["booking_value"].astype(float)
    y_dur  = d["avg_ctat_min"].astype(float)

    # dos modelos independientes
    def _fit_eval(y, name):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        return model, {"mae": float(mean_absolute_error(yte, yhat)), "r2": float(r2_score(yte, yhat))}

    fare_model, fare_metrics = _fit_eval(y_fare, "fare")
    dur_model,  dur_metrics  = _fit_eval(y_dur,  "duration")

    print(" TARIFA  → MAE: {:.2f} | R²: {:.3f}".format(fare_metrics["mae"], fare_metrics["r2"]))
    print(" DURACIÓN → MAE: {:.2f} | R²: {:.3f}".format(dur_metrics["mae"],  dur_metrics["r2"]))

    # predicción para todas las filas
    fare_pred = np.clip(fare_model.predict(X), 0, 2000)
    dur_pred  = np.clip(dur_model.predict(X),  0, 240)

    out = d[["trip_id","booking_id","ts_exact","ride_distance_km","booking_value","avg_ctat_min"]].copy()
    out["fare_pred"]         = fare_pred
    out["duration_pred_min"] = dur_pred

    return {"fare": fare_metrics, "duration": dur_metrics}, out
