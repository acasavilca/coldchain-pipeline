# bq/

Helpers for loading the simulator's raw parquet output into BigQuery (dataset `coldchain_raw`),
which is what `dbt_project` reads. They need the `bq` CLI, authenticated for the target project.
Project and dataset default to `fruit-packing-plant-simulator` / `coldchain_raw`; override with the
`PROJECT` and `DATASET` environment variables.

```bash
bash bq/create_tables.sh          # once: dataset + 15 partitioned/clustered tables (idempotent)
bash bq/load_chunks.sh simulation/runs/location_1 0 3   # append chunks 0..3 of location 1
bash bq/load_chunks.sh --weather 1                     # replace location 1's rows in weather_data
```

Each simulator run directory holds one location (`simulation/runs/location_<id>/`), so the loader
takes the run directory. Before loading anything it checks every requested chunk against its
`run_manifest.json` (row counts and file checksums) and that the manifest's location matches the run
directory; if one chunk fails, nothing is loaded. `--weather ID` deletes that location's weather rows
and appends the new file, so other locations' weather is untouched.

`schemas/` holds one BigQuery schema per table (orders and pallet history are shared by their
inbound/outbound tables). They must match `simulation/schemas/table_schemas.json` plus the
`event_timestamp` and `loaded_at` columns the pipeline adds, so update both together when a raw
table changes.

After a schema change (for example the `switch_reason` column on `compressor_events`), rerun
`bash bq/create_tables.sh` before loading. It recreates the tables, so reload your chunks afterwards.
