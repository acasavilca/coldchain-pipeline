# coldchain-pipeline

A synthetic cold-storage data platform: a physics-based simulator generates raw SCADA-style
data for a fruit-packing facility's cold rooms, and a dbt + BigQuery project turns it into
analytics-ready marts.

This repository started as a Data Engineering Zoomcamp project. The submitted version is tagged
[`zoomcamp-submission`](../../tree/zoomcamp-submission) (its Looker Studio dashboard no longer
works). The current code is a ground-up rewrite around the physics-based simulator.

## About the data

The data is synthetic, but the simulator is not a closed refrigeration loop. Besides the physics
(room heat balance, refrigerant cycle, evaporator frost and defrost, humidity and gas exchange), it
models a facility that is operated by people and instrumented imperfectly:

- **Stochastic operations**: truck arrivals and dispatches that vary by hour and weekday, doors left
  open, scheduled downtime, and crop and refrigerant changeovers.
- **Human behaviour**: drivers and employees whose fatigue (time of day, weekday, end of month) and
  carelessness change their probability of making an error.
- **Imperfect instrumentation**: sensor noise, per-sensor sampling intervals, background dropout,
  per-channel outages and facility-wide outages on the sensor tables.
- **Real weather**: outdoor conditions are real historical data for the configured site
  ([Open-Meteo](https://open-meteo.com/)), and the indoor climate responds to them through the
  envelope and the doors.

Because the simulator knows what actually happened, its event ledgers can serve as ground truth for
checking the marts, which real plant data does not allow (planned, see the roadmap). The simulator
has not been validated against a real installation; see `simulation/README.md` for assumptions and
limits.

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

1. Simulator: `python simulation/run.py` simulates one chunk of time (`--hours`, `--smoke-test`,
   `--run-dir` and more; see `simulation/README.md`). Configuration lives in `simulation/configs/`.
   Dependencies include numpy, numba, pandas, pyarrow, CoolProp, timezonefinder, reverse_geocoder
   and pycountry. Numba compilation is cached automatically (about a minute on a cold start).
2. Load the raw parquet tables into a BigQuery dataset named `coldchain_raw` with the helpers in
   `bq/`: `bash bq/create_tables.sh` once, then `bash bq/load_chunks.sh <first> <last>` (see `bq/README.md`).
3. `cd dbt_project && dbt deps && dbt seed && dbt build` with a profile named `coldchain_pipeline`.
   The source project and dataset names are in `dbt_project/models/staging/sources.yml`.

## Status

Work in progress. The simulator, the raw layer and the BigQuery loading are working. The dbt marts
for the facility's core operations (room conditions, compressor and humidifier cycles, doors,
downtime, temperature compliance, inventory and orders) are built, and the incremental chain is
checked against full refreshes. Column-level tests (`schema.yml`) are still to be added.

Evaporator frost, electric defrost, low-pressure trips and the airflow interlock were added to the
simulator later, motivated by a separate long-running contract project (DataAnnotation). They are
exported as raw tables (`defrost_events`, and `lp_trip_events` for low-pressure trips) and
telemetry (`t_coil_evap_c`, `evap_air_velocity_m_s`). Every compressor state change in
`compressor_events` carries a `switch_reason` (thermostat, defrost, downtime, low-pressure trip,
airflow interlock), and the reason codes are decoded in the marts through seeds generated from the
simulator config. The dbt layer already ingests all of this (staging models, sensor tags, seeds).
What remains is wiring it into the existing models, which reuses the established patterns more than
it adds new modelling.

## Roadmap

Items are listed in the order planned. Nothing below is marked done until it is merged.

| # | Item | Status |
|---|---|---|
| 1 | **Defrost and low-pressure trip analytics in dbt**: wire the new raw tables into the existing models. Classify every compressor stop by cause using `switch_reason`; defrost and trip windows in the state timeline; defrost recovery and cause attribution in the compliance models; post-defrost handling and airflow-ratio metrics in the compressor performance mart; per-defrost and per-trip fact tables. | In progress |
| 2 | **History generation**: simulate a long history as consecutive daily chunks (one UTC day each) and load it into BigQuery. | Planned |
| 3 | **Tests and documentation**: `schema.yml` with column tests, defrost/trip sanity tests, and ground-truth checks of the marts against the simulator's event ledgers. | Planned |
| 4 | **Dashboard**: Evidence.dev on top of the marts. | Planned |
| 5 | **Containers and CI**: Dockerfile for the simulator, GitHub Actions for tests and `dbt build`. | Planned |
| 6 | **Infrastructure as code**: Terraform for the storage bucket, BigQuery tables and service accounts, replacing the shell table-creation helper. | Planned |
| 7 | **Orchestration**: Kestra flows that run the simulator for the previous day, load it, test it and run dbt. | Planned |
| 8 | **v2, streaming and Spark**: refactor the simulator into a digital twin that streams its output through Kafka into Spark, alongside the BigQuery marts. | Planned |
