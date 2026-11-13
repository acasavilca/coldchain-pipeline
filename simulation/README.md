# simulation/

A physics-based simulator of one refrigerated fruit-storage room and the refrigeration plant that
serves it. It produces the raw data the dbt project models: SCADA-style sensor telemetry, controller
event ledgers, and warehouse order and pallet records. The physics runs on a fixed 0.1 s step
(Numba-compiled), and the output is then pushed through a layer that turns internal state arrays
into what a historian and a WMS would actually hand over, including sensor noise, sampling rates and
transmission loss.

## Quick start

```bash
cd simulation
pip install numpy numba pandas pyarrow scipy requests timezonefinder reverse_geocoder pycountry CoolProp matplotlib
python run.py
```

`run.py` is a command-line script (`python run.py --help` lists everything):

| Option | Meaning |
|---|---|
| `--location ID` | Which site (id in `facility_location.locations`). Cold start: defaults to `default_location_id` and prints a message saying so. Resume: must match the run directory. |
| `--seed N` | RNG seed. Cold start only (an error on a resume). Default: the location's `rng_seed` in the config. |
| `--hours N` / `--days N` | Length of one chunk (default 72 h). `--days` is the daily mode: whole UTC days, and the chunk must start at 00:00 UTC. The two are mutually exclusive. |
| `--align-first` | With `--days` on a cold start that is not at 00:00 UTC: run only up to the next 00:00 UTC. |
| `--start-date ISO` | Facility-local start of operations. Cold start only; ignored when resuming. |
| `--fruit {blueberry,avocado}`, `--refrigerant {r404a,r134a}` | Initial crop and refrigerant. Cold start only. |
| `--run-dir PATH` | Where the state and outputs live (default: `runs/location_<id>/`, git-ignored). |
| `--refresh-weather` | Download weather for this chunk from Open-Meteo and overwrite the location's cache, instead of reading it. |
| `--weather-only` | Download the weather for the period (`--hours`/`--days` from the start) into the location's cache and exit without simulating. Use it to fetch a long period once. |
| `--no-run-outputs` | Skip the multi-GB `run_outputs_chunk_NNNN.npz` diagnostic dump. |
| `--smoke-test [MIN]` | Simulate only a few minutes (default 4) to warm the Numba cache. Writes nothing. |
| `--no-cache`, `--clear-cache` | Ignore / delete the Numba compilation cache (see below). |
| `--fastmath {safe,full,off}` | Numba fast-math level (default `safe`). |

**Numba cache.** Compiled functions are cached in `.numba_cache/<hash>-<level>/` (git-ignored). The
hash covers every `.py` file and config in this folder, so editing any of them, including the shared
constants that Numba's own cache does not track, recompiles automatically. `--no-cache` and
`--clear-cache` are escape hatches and are rarely needed. A cold compile takes about a minute, a
cached start a few seconds. Compilation uses `jit.py`; the `safe` fast-math level keeps NaN/Inf
semantics, and a function can opt out with `@njit(fastmath=False)`.

## Locations and run directories

One run directory simulates one location. The default is `simulation/runs/location_<id>/`, so locations
never share state and can run as separate processes in parallel (about 2 GB of RAM per simulated day).

- **State and identity.** Each call continues from the state saved by the previous call (`state.npz`,
  `rng_state.json`), so consecutive calls produce consecutive chunks. The cold start also writes
  `run_meta.json` (location, seed, start time). A resume reads it: a mismatched `--location` is an
  error, and a `state.npz` without `run_meta.json` (from before multi-location support) is refused,
  because its location and seed are unknown. To start over, delete the run directory's `state.npz`,
  `rng_state.json`, `run_meta.json`, `run_outputs/` and `raw_parquets/`.
- **Seeds.** A cold start uses `--seed` or else the location's `rng_seed` in the config (location 1:
  42, the others differ). If a sibling run directory already uses the same seed, a warning is printed:
  two runs with the same seed draw the same random sequence.
- **Weather.** Cached per location in `weather/location_<id>/` (`weather_data.parquet` for the
  simulator, `raw_weather_data.parquet` for BigQuery). `run.py` refuses a cache whose `location_id` is
  a different location or that does not cover the chunk. The committed cache is location 1 only
  (2024-12-30 to 2026-01-02); for another location, or a longer period, fetch it once with
  `python run.py --location 2 --start-date 2025-04-30T19:00:00 --days 400 --weather-only`.
- **Daily mode.** `--days N` makes every chunk whole UTC days, so a chunk maps onto BigQuery's day
  partitions. A cold start that is not at 00:00 UTC is refused unless `--align-first` is given, which
  runs only up to the next midnight. A resume must begin at 00:00 UTC; if it does not, the error says
  how many `--hours` reach the next midnight.
- **Run manifest.** Every chunk also writes `raw_parquets/chunk_NNNN/run_manifest.json`: location, seed,
  start and end (UTC), row count and SHA-256 per table, the code fingerprint (`jit.source_fingerprint`),
  the facility-config hash and the fast-math level. `bq/load_chunks.sh` checks it before loading.

Each chunk writes:

- `raw_parquets/chunk_NNNN/<table>.parquet` and `run_manifest.json`, the raw tables that get loaded into BigQuery.
- `run_outputs/run_outputs_chunk_NNNN.npz`, the full-resolution internal arrays, used only for plotting and
  debugging. At 0.1 s resolution a chunk is several GB, so this folder is git-ignored.

## What a run does

1. **Configs to arrays** (`array_builders.py`): the JSON configs become flat NumPy vectors and matrices, which is
   the form Numba needs. Equipment is picked from the catalog by ID (`active_equipment_id_selection`).
2. **Setup** (`run_setup.py`): weather retrieval and resampling into forcing arrays, downtime and
   crop/refrigerant rotation schedules, and first-run state initialization.
3. **Step loop** (`engine.py`, `run_simulator`): every 0.1 s it updates ambient forcing, room and
   fruit thermal balance, humidity and gas balance, the refrigeration cycle, evaporator frost and
   defrost, sensors and control loops, and the logistics (arrivals, dispatches, doors, workers).
   `output_writer.py` logs the step into pre-allocated output arrays.
4. **Raw data layer** (`scadafier.py`, `nullifier.py`): converts the arrays into long tag/timestamp/value
   telemetry (each sensor is only written at its own sampling interval) and tidy event and
   business tables, then applies simulated transmission loss to the sensor tables only.
5. **Persistence** (`state_persistence.py`): saves room, pallet, door and box state plus RNG state so the
   next call resumes exactly where this one stopped.

## What is modelled

- **Room**: envelope, floor and ground heat flows, infiltration through two doorways, fruit bulk
  temperature, an adjacent-plant thermal model, and a 20 × 15 × 6 m room geometry.
- **Fruit**: crop-specific respiration heat and CO2/O2 balance (blueberry and avocado), moisture loss
  and mass change, quality grades, spoilage.
- **Refrigeration cycle**: a compressor, an air-cooled condenser and a fan-driven evaporator with a
  superheat PID loop, solved each step on polynomial fits of refrigerant properties (R404A, R134a).
  Running mass flow, cooling duty and power follow from the load: the evaporator duty comes from the
  coil heat transfer, mass flow from duty over the enthalpy rise, and power from the isentropic work
  and a literature isentropic-efficiency fit. Datasheet data enters only through calibration at
  start-up: the compressor's volumetric-efficiency fit (fitted to the datasheet mass-flow table; it sets
  the maximum mass flow and an idle power floor, and never limited the flow in a 24 h test run), the
  coil UA values (from rated capacities) and the fan curve (airflow against static pressure, which
  sets how much airflow a frosted coil loses).
- **Frost and defrost**: frost growth on the coil, a coil-temperature electric defrost controller
  (wait for stop, settle, heat, drip, fan delay), an evaporator airflow-proof switch and a
  low-pressure trip. Timings and thresholds are in `electric_defrost_control` and
  `evaporator_airflow_proof` in the facility config.
- **Humidity**: humidifier discs and clusters under PID control, evaporator moisture exchange.
- **Logistics and people**: stochastic arrivals and dispatches by hour and weekday, drivers and
  employees with fatigue and carelessness, supervisors, doors left open, scheduled downtime and
  rotation events.

## Raw tables

| Table | Content |
|---|---|
| `telemetry` | Long format (tag, value) from fixed room and plant instruments, 14 tags: room, return and supply air temperature, relative humidity, CO2, O2, suction and liquid line temperature and pressure, compressor power, evaporator coil temperature, evaporator air velocity. |
| `batch_sensors` | Pallet-mounted pulp temperature (`t_pulp_c`) and load cell (`mass_kg`), polled every 10 minutes. |
| `compressor_events`, `humidifier_events`, `door_events` | Controller and door event ledgers. Compressor events carry `switch_reason` (thermostat, defrost, downtime, low-pressure trip, airflow interlock); codes are decoded by the generated seed `seed_compressor_switch_reason_reference`. |
| `lp_trip_events`, `defrost_events` | Low-pressure switch state changes, and defrost controller phase changes (one row each time a defrost phase is entered, with the termination reason on the heat-to-drip row). Phase codes are decoded by the generated seed `seed_defrost_phase_reference`. |
| `crop_rotation_events`, `refrigerant_rotation_events`, `downtime_sequence_events` | Planned changeovers and downtime windows. |
| `inbound_orders`, `outbound_orders`, `inbound_pallet_history`, `outbound_pallet_history` | Warehouse records. |

Schemas and dtypes are in `schemas/table_schemas.json`. Telemetry and batch sensors get simulated
transmission loss (background dropout, per-channel outages, facility-wide outages); the event and business
ledgers do not, because they are systems of record.

## Configuration

| File | Content |
|---|---|
| `configs/facility_runtime_config.json` | Sites (`facility_location.locations`, each with its own `rng_seed`, and `default_location_id`), room geometry, hardware control settings, sensor profiles, crops, workforce, logistics, runtime settings. |
| `configs/catalog_equipment_config.json` | Compressors, evaporators, condensers and fans with datasheet data. Which ones are active is set by `active_equipment_id_selection` in the facility config. |
| `configs/fruit_respiration_curves.json` | Respiration curves per crop. |
| `configs/downtime_schedule.json` | Scheduled downtime windows. |
| `polynomial_surfaces/*.json` | Fitted refrigerant and fan surfaces, generated by `tools/`. |

Changing a config means regenerating the dbt seeds, which are derived from the same files:

```bash
python generate_dbt_seeds.py     # writes ../dbt_project/seeds/*.csv
```

## Checking a run

- `python inspect_outputs.py [--out-dir run_outputs] [--plots-dir plots]` plots room and
  refrigeration behavior, frost and defrost diagnostics, solver health, inventory and
  events, and sensor traces from the saved `run_outputs/` chunks.
- `python tools/compare_telemetry.py baseline.parquet new.parquet` compares two telemetry files row
  by row. Two runs from the same state, seed and config must be identical (exit code 0). It is the
  regression check for code changes that should not alter behavior.

## Tools (offline calibration)

| Script | Purpose |
|---|---|
| `tools/fit_refrigerant_surfaces.py` | Fits polynomial and Antoine surfaces of refrigerant properties from CoolProp (needs `CoolProp`). |
| `tools/fit_compressor_performance_surfaces.py` | Fits compressor performance surfaces from the datasheet arrays. |
| `tools/fit_fan_curves.py` | Fits the evaporator fan curve (airflow versus static pressure) from datasheet points. |

## Assumptions and limits

- The simulator has not been validated against a real installation. Equipment parameters come from
  public datasheets and calibrate the component models; no model output is fitted to measured plant
  data. Derived quantities were checked for plausibility against typical ranges for this
  kind of plant, which is a sanity check, not a validation. For example, a 24 h cold-start run
  (R404A, 0 C setpoint, evaporating near -5 C, condensing near 32 C) gave a running COP with median
  3.2 (5th to 95th percentile 2.7 to 4.0), about 10 kW compressor input and about 32 kW evaporator
  duty while running, against datasheet-fit rating-point values of about 31 kW cooling and 11 kW input. Control settings (defrost interval,
  termination temperature, airflow-proof threshold) are declared trial values, not verified
  manufacturer settings.
- Sensor noise levels, filter windows and dropout and outage rates are plausible placeholders. The
  facility-wide outage rate in particular is not literature-backed.
- The room, crop mix and logistics patterns are synthetic. The weather is real historical data for the
  configured site.

## Sources

- Weather: [Open-Meteo](https://open-meteo.com/), licensed CC BY 4.0.
- Equipment datasheets: Danfoss MTZ compressors, HC Heat-Exchangers HMR-630 unit coolers,
  Greenhalgh C1-series condensers, ebm-papst S4D630 fans (exact models in the catalog).
- Refrigerant properties: [CoolProp](http://www.coolprop.org/).
