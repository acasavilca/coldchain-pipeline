# coldchain-pipeline

A synthetic cold-storage data platform: a physics-based simulator generates raw SCADA-style
data for a fruit-packing facility's cold rooms, and a dbt + BigQuery project turns it into
analytics-ready marts.

This repository started as a Data Engineering Zoomcamp project. The submitted version is tagged
[`zoomcamp-submission`](../../tree/zoomcamp-submission) (its Looker Studio dashboard no longer
works). The current code is a ground-up rewrite around the physics-based simulator.

## What is in the repo

| Path | Contents |
|---|---|
| `simulation/` | Numba-JIT Python simulator: room thermal balance, refrigeration cycle (R404A/R134a polynomial fits), evaporator frost and electric defrost, humidity/CO2/O2, door and logistics events, sensor and telemetry-loss layers. Writes raw parquet tables. |
| `bq/` | Shell helpers and table schemas for creating the raw BigQuery tables and loading the simulator's parquet chunks. |
| `dbt_project/` | dbt project (BigQuery): `staging` views, `intermediate` models, `marts` (dimensions and facts), seeds generated from the simulator configs, and tests. |

## Raw tables produced by the simulator

`telemetry`, `batch_sensors` (long format: tag, timestamp, value) and the event/ledger tables
`compressor_events`, `humidifier_events`, `door_events`, `crop_rotation_events`,
`refrigerant_rotation_events`, `downtime_sequence_events`, `lp_trip_events`,
`defrost_events`, `inbound_orders`, `outbound_orders`, `inbound_pallet_history`,
`outbound_pallet_history`. Weather comes from satellite/reanalysis data.

## dbt design notes

- Large intermediate models are incremental with `insert_overwrite`, partitioned by day. Each run
  recomputes whole days touched by new rows (day-aligned bounds), so window functions stay correct.
- Window state that crosses a day boundary (for example the last decisive compliance status) is
  carried in a seed row read from the model's own previous output.
- `ingested_at` is the max over every input of a row, so late-arriving data re-triggers its own day.
- `chain_test.sh`, `hourly_test.sh` and `mart_test.sh` compare per-day fingerprints of an
  incremental run against a snapshot of a full refresh.
- Seeds (`dbt_project/seeds`) are generated from the simulator configs:
  `python simulation/generate_dbt_seeds.py` (run it from `simulation/`).
- Sensor tags are configuration-driven (`telemetry_sensor_tags` / `batch_sensor_tags` in
  `dbt_project.yml` plus `seed_sensor_polling_config`). A singular test,
  `assert_sensor_tags_consistent`, fails the build if a raw tag is missing from either, so a
  rename cannot silently drop rows.

## Running

1. Simulator: `simulation/run.py` is the entry point for one chunk of simulated time. Configuration
   lives in `simulation/configs/`. Dependencies include numpy, numba, pandas, pyarrow,
   timezonefinder, reverse_geocoder and pycountry.
2. Load the raw parquet tables into a BigQuery dataset named `coldchain_raw` with the helpers in
   `bq/`: `bash bq/create_tables.sh` once, then `bash bq/load_chunks.sh <first> <last>` (see `bq/README.md`).
3. `cd dbt_project && dbt deps && dbt seed && dbt build` with a profile named `coldchain_pipeline`.
   The source project and dataset names are in `dbt_project/models/staging/sources.yml`.

## Status

Work in progress. The marts are built and tested for the incremental chain.

The simulator models evaporator frost, electric defrost, low-pressure trips and the airflow
interlock, and exports them as raw tables (`defrost_events`, `lp_trip_events`) and telemetry
(`t_coil_evap_c`, `evap_air_velocity_m_s`). The dbt layer ingests these (staging models and
sensor tags), but the models do not yet use them analytically.

## Roadmap

- **Defrost and trip analytics in dbt**: classify every compressor stop by cause (thermostat,
  defrost, low-pressure trip, airflow interlock); defrost and trip windows in the state timeline;
  defrost recovery and cause attribution in the compliance models; post-defrost handling and
  airflow-ratio metrics in the compressor performance mart; per-defrost and per-trip fact tables.
- **Tests and documentation**: `schema.yml` with column tests, defrost/trip sanity tests.
- **Dashboard**: Evidence.dev on top of the marts.
- **Orchestration**: Kestra flows for loading and running dbt.
