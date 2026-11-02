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
