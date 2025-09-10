DROP TABLE IF EXISTS public.dim_datetime_hour;
CREATE TABLE public.dim_datetime_hour (
  datetime_hour_id BIGSERIAL PRIMARY KEY,
  ts_hour          TIMESTAMP NOT NULL UNIQUE,
  "date"           DATE      NOT NULL,
  "year"           INT       NOT NULL,
  "month"          INT       NOT NULL,
  "day"            INT       NOT NULL,
  "hour"           INT       NOT NULL,
  "dow"            INT       NOT NULL,     -- 0=lunes
  is_weekend       SMALLINT  NOT NULL      -- 1 si sab/dom
);

-- Poblar desde staging
INSERT INTO public.dim_datetime_hour (ts_hour, "date", "year", "month", "day", "hour", "dow", is_weekend)
SELECT DISTINCT
  (s."date" ::date + s."time"::time)::timestamp AS ts_hour,
  s."date"::date                               AS "date",
  EXTRACT(YEAR  FROM s."date")::int            AS "year",
  EXTRACT(MONTH FROM s."date")::int            AS "month",
  EXTRACT(DAY   FROM s."date")::int            AS "day",
  EXTRACT(HOUR  FROM s."time"::time)::int      AS "hour",
  EXTRACT(DOW   FROM s."date")::int            AS "dow",
  CASE WHEN EXTRACT(DOW FROM s."date") IN (6,0) THEN 1 ELSE 0 END::smallint AS is_weekend
FROM public.uber_clean_ml s
WHERE s."date" IS NOT NULL
  AND s."time" IS NOT NULL
ON CONFLICT DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_dim_datetime_hour_ts ON public.dim_datetime_hour(ts_hour);




--------------------------------------------------------------------

DROP TABLE IF EXISTS public.dim_location;
CREATE TABLE public.dim_location (
  location_id   BIGSERIAL PRIMARY KEY,
  location_name TEXT NOT NULL,
  location_type TEXT NOT NULL CHECK (location_type IN ('pickup','drop')),
  UNIQUE (location_name, location_type)
);

-- Limpieza: trim/lower y NULLIF a vacíos
INSERT INTO public.dim_location (location_name, location_type)
SELECT DISTINCT
  lower(trim(s."pickup_location")) AS location_name,
  'pickup'                         AS location_type
FROM public.uber_clean_ml s
WHERE NULLIF(trim(s."pickup_location"), '') IS NOT NULL
ON CONFLICT DO NOTHING;

INSERT INTO public.dim_location (location_name, location_type)
SELECT DISTINCT
  lower(trim(s."drop_location")) AS location_name,
  'drop'                         AS location_type
FROM public.uber_clean_ml s
WHERE NULLIF(trim(s."drop_location"), '') IS NOT NULL
ON CONFLICT DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_dim_location_name_type ON public.dim_location(location_name, location_type);



--------------------------------------------------------------

DROP TABLE IF EXISTS public.dim_product;
CREATE TABLE public.dim_product (
  product_id   BIGSERIAL PRIMARY KEY,
  product_name TEXT NOT NULL UNIQUE
);

INSERT INTO public.dim_product (product_name)
SELECT DISTINCT lower(trim(s."vehicle_type"))
FROM public.uber_clean_ml s
WHERE NULLIF(trim(s."vehicle_type"), '') IS NOT NULL
ON CONFLICT DO NOTHING;

-- Asegura al menos 'unknown' si no hay datos
INSERT INTO public.dim_product (product_name)
VALUES ('unknown')
ON CONFLICT DO NOTHING;

----------------------------------------------------------------------


-- FACT: viajes
DROP TABLE IF EXISTS public.fact_trip;
CREATE TABLE public.fact_trip (
  trip_id               BIGSERIAL PRIMARY KEY,
  booking_id            TEXT,
  datetime_hour_id      BIGINT REFERENCES public.dim_datetime_hour(datetime_hour_id),
  pickup_location_id    BIGINT REFERENCES public.dim_location(location_id),
  drop_location_id      BIGINT REFERENCES public.dim_location(location_id),
  product_id            BIGINT REFERENCES public.dim_product(product_id),

  -- métricas
  ride_distance_km      NUMERIC,
  booking_value         NUMERIC,
  duration_min          NUMERIC,  -- Avg CTAT
  arrive_time_min       NUMERIC,  -- Avg VTAT
  driver_ratings        NUMERIC,
  customer_rating       NUMERIC,
  booking_status        TEXT,

  -- banderas textuales
  cancelled_by_customer TEXT,
  cancelled_by_driver   TEXT,
  incomplete_ride       TEXT
);

-- Insertar desde staging con joins a dims (usando Date + Time directamente)
CREATE TEMP TABLE tmp_basee AS
  SELECT
    s.*,
    -- ts_hour desde Date (DATE) + Time (TIME)
    (s."date" + s."time")::timestamp AS ts_exact,
	s."date"                         AS dte,
    EXTRACT(HOUR FROM s."time")::int AS hr,
    -- limpiezas mínimas para join con dims
    lower(trim(s.pickup_location)) AS pickup_loc_clean,
    lower(trim(s.drop_location))   AS drop_loc_clean,
    lower(trim(s.vehicle_type))    AS product_clean
  FROM public.uber_clean_ml  s
  WHERE s."date" IS NOT null
  AND s."time" IS NOT NULL


INSERT INTO public.fact_trip (
  booking_id, datetime_hour_id, pickup_location_id, drop_location_id, product_id,
  ride_distance_km, booking_value, duration_min, arrive_time_min,
  driver_ratings, customer_rating, booking_status,
  cancelled_by_customer, cancelled_by_driver, incomplete_ride,
  ts_exact                                   -- << guardamos timestamp exacto
)
SELECT
  b."booking_id",
  dt.datetime_hour_id,
  dlp.location_id,
  dld.location_id,
  dp.product_id,
  NULLIF(b.ride_distance, 0),
  NULLIF(b.booking_value, 0),
  NULLIF(b.avg_ctat, 0),
  NULLIF(b.avg_vtat, 0),
  NULLIF(b.driver_ratings, 0),
  NULLIF(b.customer_rating, 0),
  b.booking_status,
  b.cancelled_rides_by_customer,
  b.cancelled_rides_by_driver,
  b.incomplete_rides,
  b.ts_exact
FROM tmp_basee b
LEFT JOIN public.dim_datetime_hour dt
  ON dt."ts_hour" = b.ts_exact AND dt."hour" = b.hr   -- FK por hora (robusto)
LEFT JOIN public.dim_location dlp
  ON dlp.location_name = b.pickup_loc_clean AND dlp.location_type = 'pickup'
LEFT JOIN public.dim_location dld
  ON dld.location_name = b.drop_loc_clean   AND dld.location_type = 'drop'
LEFT JOIN public.dim_product dp
  ON dp.product_name = COALESCE(NULLIF(b.product_clean,''), 'unknown');

-- Índices útiles
CREATE INDEX IF NOT EXISTS idx_fact_trip_time   ON public.fact_trip(datetime_hour_id);
CREATE INDEX IF NOT EXISTS idx_fact_trip_prod   ON public.fact_trip(product_id);
CREATE INDEX IF NOT EXISTS idx_fact_trip_pickup ON public.fact_trip(pickup_location_id);
CREATE INDEX IF NOT EXISTS idx_fact_trip_drop   ON public.fact_trip(drop_location_id);


------------------------------------------------------------------


CREATE OR REPLACE VIEW public.vw_hourly_demand AS
WITH base AS (
  SELECT
    date_trunc('hour', f.ts_exact)        AS ts_hour_floor,  -- << 12:00:00
    COUNT(*)::int                         AS trips
  FROM public.fact_trip f
  WHERE f.booking_status ILIKE 'completed'
    AND f.ts_exact IS NOT NULL
  GROUP BY 1
)
SELECT
  d.datetime_hour_id,
  b.ts_hour_floor                         AS ts_hour,
  b.trips
FROM base b
JOIN public.dim_datetime_hour d
  ON d."date" = b.ts_hour_floor::date
 AND d."hour" = EXTRACT(HOUR FROM b.ts_hour_floor);

