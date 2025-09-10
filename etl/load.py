# etl/load.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from .extract import extract
from .transform import (
    clean_for_ml,
    train_demand_model,
    forecast_next_hours,
    train_and_predict_trip_models,
)

load_dotenv()

def get_engine():
    uri = os.getenv("PG_URI")
    if not uri:
        raise ValueError("Falta PG_URI en .env")
    return create_engine(uri, pool_pre_ping=True)

DDL_STAGING = """
CREATE TABLE IF NOT EXISTS public.uber_clean_ml (
  trip_id BIGINT,
  booking_id TEXT,
  booking_status TEXT,
  vehicle_type TEXT,
  pickup_location TEXT,
  drop_location TEXT,
  date DATE,
  time TIME,
  ts_exact TIMESTAMP,
  ts_hour TIMESTAMP,
  hour INT,
  dow INT,
  is_weekend INT,
  month INT,
  year INT,
  ride_distance_km NUMERIC,
  booking_value NUMERIC,
  avg_ctat_min NUMERIC,
  avg_vtat_min NUMERIC,
  driver_ratings NUMERIC,
  customer_rating NUMERIC
);
"""

DDL_DEMAND = """
CREATE TABLE IF NOT EXISTS public.hourly_demand_pred (
  ts_hour TIMESTAMP PRIMARY KEY,
  trips_pred NUMERIC
);
"""

DDL_TRIP_PRED = """
CREATE TABLE IF NOT EXISTS public.trip_predictions (
  trip_id BIGINT PRIMARY KEY,
  booking_id TEXT,
  ts_exact TIMESTAMP,
  ride_distance_km NUMERIC,
  booking_value NUMERIC,
  avg_ctat_min NUMERIC,
  fare_pred NUMERIC,
  duration_pred_min NUMERIC
);
"""

UPSERT_DEMAND = """
INSERT INTO public.hourly_demand_pred (ts_hour, trips_pred)
VALUES (:ts_hour, :trips_pred)
ON CONFLICT (ts_hour) DO UPDATE
SET trips_pred = EXCLUDED.trips_pred;
"""

UPSERT_TRIP = """
INSERT INTO public.trip_predictions
(trip_id, booking_id, ts_exact, ride_distance_km, booking_value, avg_ctat_min, fare_pred, duration_pred_min)
VALUES (:trip_id, :booking_id, :ts_exact, :ride_distance_km, :booking_value, :avg_ctat_min, :fare_pred, :duration_pred_min)
ON CONFLICT (trip_id) DO UPDATE
SET booking_id = EXCLUDED.booking_id,
    ts_exact = EXCLUDED.ts_exact,
    ride_distance_km = EXCLUDED.ride_distance_km,
    booking_value = EXCLUDED.booking_value,
    avg_ctat_min = EXCLUDED.avg_ctat_min,
    fare_pred = EXCLUDED.fare_pred,
    duration_pred_min = EXCLUDED.duration_pred_min;
"""

def save_staging(df_clean: pd.DataFrame):
    eng = get_engine()
    with eng.begin() as con:
        con.execute(text(DDL_STAGING))
        df_clean.to_sql("uber_clean_ml", con, if_exists="replace", index=False)
    print(f" Guardado staging public.uber_clean_ml ({len(df_clean)} filas)")

def save_hourly_forecast(df_fcst: pd.DataFrame):
    eng = get_engine()
    with eng.begin() as con:
        con.execute(text(DDL_DEMAND))
        rows = [
            {"ts_hour": r.ts_hour, "trips_pred": float(r.trips_pred)}
            for r in df_fcst.itertuples(index=False)
        ]
        con.execute(text(UPSERT_DEMAND), rows)
    print(f" Guardado forecast en public.hourly_demand_pred ({len(df_fcst)} filas)")

def save_trip_predictions(df_pred: pd.DataFrame):
    eng = get_engine()
    with eng.begin() as con:
        con.execute(text(DDL_TRIP_PRED))
        rows = []
        for r in df_pred.itertuples(index=False):
            rows.append({
                "trip_id": int(r.trip_id),
                "booking_id": r.booking_id,
                "ts_exact": r.ts_exact,
                "ride_distance_km": float(r.ride_distance_km),
                "booking_value": float(r.booking_value),
                "avg_ctat_min": float(r.avg_ctat_min),
                "fare_pred": float(r.fare_pred),
                "duration_pred_min": float(r.duration_pred_min),
            })
        con.execute(text(UPSERT_TRIP), rows)
    print(f"Guardado trip_predictions ({len(df_pred)} filas)")

def main():
    # 1) EXTRACT (Kaggle → CSV → DataFrame)
    raw = extract()

    # 2) TRANSFORM (limpieza / features comunes)
    clean = clean_for_ml(raw)

    # 3) LOAD staging
    save_staging(clean)

    # 4) MODEL A: Demanda por hora (autoregresivo simple con RF)
    model_a, metrics_a, hist = train_demand_model(clean)
    fcst = forecast_next_hours(hist, model_a, horizon_hours=168)  # 7 días
    save_hourly_forecast(fcst)

    # 5) MODEL B: Tarifa y duración por viaje (RF)
    metrics_b, trip_pred = train_and_predict_trip_models(clean)
    save_trip_predictions(trip_pred)

    print(" Pipeline completo.")
    print("   Métricas A (Demanda):", metrics_a)
    print("   Métricas B (Trip):   ", metrics_b)

if __name__ == "__main__":
    main()
