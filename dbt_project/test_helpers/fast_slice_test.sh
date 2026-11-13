#!/usr/bin/env bash
# End-to-end incremental check for ONE model whose table is already built:
#   snapshot -> dbt run with a forced cutoff -> diff vs snapshot -> partitions rewritten -> bytes.
# Pass/fail: the three diff numbers must be 0. Use ./ingest_spread.sh first to pick a cutoff inside the load.
# Usage: ./incremental_slice_test.sh <model> <cutoff UTC, e.g. "2026-11-05 04:59:00+00"> [diff_vs_snapshot options]
#   cutoff "1970-01-01 00:00:00+00" = reprocess everything (full-range incremental run)
source "$(dirname "$0")/_common.sh"
[ $# -ge 2 ] || { sed -n 2,8p "$0"; exit 1; }
M="$1"; C="$2"; shift 2
START="$(date -u +'%Y-%m-%d %H:%M:%S')"
echo "start (UTC): $START"
"$(dirname "$0")/snapshot_table.sh" "$M"
echo "=== dbt run (cutoff_override=$C)"
( cd "$DBT_PROJECT_DIR" && dbt run -s "$M" --vars "{cutoff_override: \"$C\"}" 2>&1 | grep -E "WM new|Succeeded|Error|error" )
echo "=== diff vs snapshot"
"$(dirname "$0")/diff_vs_snapshot.sh" "$M" "$@"
echo "=== partitions rewritten since start"
bqq "select countif(last_modified_time >= timestamp('${START}')) as rewritten_partitions, count(*) as total_partitions,
            sum(if(last_modified_time >= timestamp('${START}'), total_rows, 0)) as rewritten_rows, sum(total_rows) as total_rows
     from ${FQ}.INFORMATION_SCHEMA.PARTITIONS
     where table_name = '${M}' and partition_id not in ('__NULL__','__UNPARTITIONED__')"
