# bq/

Helpers for loading the simulator's raw parquet output into BigQuery (dataset `coldchain_raw`),
which is what `dbt_project` reads. They need the `bq` CLI, authenticated for the target project.
Project and dataset default to `fruit-packing-plant-simulator` / `coldchain_raw`; override with the
`PROJECT` and `DATASET` environment variables.

```bash
bash bq/create_tables.sh          # once: dataset + 15 partitioned/clustered tables (idempotent)
bash bq/load_chunks.sh 0 3        # append chunks 0..3 from simulation/raw_parquets/
bash bq/load_chunks.sh --weather  # (re)load the weather table from simulation/raw_weather_data.parquet
```

`schemas/` holds one BigQuery schema per table (orders and pallet history are shared by their
inbound/outbound tables). They must match `simulation/schemas/table_schemas.json` plus the
`event_timestamp` and `loaded_at` columns the pipeline adds, so update both together when a raw
table changes.

`create_tables.sh` never alters or recreates existing tables (`bq mk --force` skips them). After a
schema change (for example the `switch_reason` column on `compressor_events`), either add the column in
place, `ALTER TABLE <dataset>.compressor_events ADD COLUMN switch_reason INT64`, or drop the table and
rerun `bash bq/create_tables.sh`, then reload your chunks. To keep an existing dataset untouched, point
`DATASET` at a new one: create it first with `bq mk --dataset --location=<same as the other datasets>
<project>:<dataset>`, then run the two scripts with `DATASET=<dataset>`.
