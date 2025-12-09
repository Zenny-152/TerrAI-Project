CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS slide_events (
  id SERIAL PRIMARY KEY,
  event_date date,
  severity text,
  source text,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
  id SERIAL PRIMARY KEY,
  prob numeric,
  model_version text,
  meta_info jsonb,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS images (
  id SERIAL PRIMARY KEY,
  user_id text,
  filename text,
  filepath text,
  lat double precision,
  lon double precision,
  exif jsonb,
  model_prob double precision,
  model_version text,
  meta_info jsonb,
  created_at TIMESTAMP DEFAULT now()
);