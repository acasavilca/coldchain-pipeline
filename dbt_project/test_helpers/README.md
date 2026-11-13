# test_helpers

Bash helpers for checking incremental dbt models against BigQuery. Run them from anywhere; they find `dbt_project/` themselves.
Defaults: project `fruit-packing-plant-simulator`, dataset `coldchain_dev`, location `us`.
Override with env vars, e.g. `BQ_DATASET=coldchain_prod ./diff_vs_snapshot.sh fct_pallet_movements`.

| Script | What it does |
|---|---|
| `_common.sh` | Shared settings and helpers (sourced by the others). Not run directly. |
| `snapshot_table.sh <model>...` | Copies each built table to `snap_<model>`. |
| `diff_vs_snapshot.sh <model> [partition_col] [--except cols] [--round cols] [--digits N]` | Diffs a table against `snap_<model>` both ways plus a per-day count/hash check. Every number must be 0. `--except dbt_updated_at` skips volatile columns; `--round` rounds float columns before comparing. |
| `ingest_spread.sh <table>...` | min / p20 / p50 / p80 / max of `ingested_at`, to pick a cutoff inside the load. |
| `incremental_slice_test.sh <model> <cutoff> [diff options]` | Whole check in one go: snapshot, `dbt run` with `cutoff_override`, diff, partitions rewritten, bytes. Cutoff `1970-01-01 00:00:00+00` reprocesses everything. |
| `job_bytes.sh "<UTC start>" [model]` | MB processed/billed per dbt statement since a time (needs `bigquery.jobs.listAll`; the jobs view lags ~30 s). |
| `watermark_check.sh <model> [hours]` | New vs old watermark, latest partitions, dry-run bytes old vs pruned. Add `{{ log('WM new: ' ~ watermark, info=True) }}` after the model's `set watermark` line first. |

## Typical flows

Check one incremental model after changing it (table already built):
```
./ingest_spread.sh int_room_temperature_compliance_readings      # pick a cutoff around p80
./incremental_slice_test.sh int_door_recovery_events "2026-11-05 04:59:00+00"
./incremental_slice_test.sh int_door_recovery_events "1970-01-01 00:00:00+00"   # full-range pass
```
Check a model after a rebuild against an earlier version of itself:
```
./snapshot_table.sh fct_pallet_movements        # before
dbt run --full-refresh -s fct_pallet_movements
./diff_vs_snapshot.sh fct_pallet_movements --except dbt_updated_at
```

## Legacy scripts (in `legacy_prod/`)
`chain_test.sh`, `correctness_test.sh`, `hourly_test.sh`, `mart_test.sh`, `snapshot_test.sh` are the earlier `coldchain_prod` checks, kept as they were.
`watermark_test.sh` was a one-off and is superseded by `watermark_check.sh`.
