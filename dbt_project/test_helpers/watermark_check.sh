#!/usr/bin/env bash
# Check the ingest_watermark macro on one incremental model.
#   1. new watermark (printed by the model's own log line, if any) vs the old 'max(ingested_at) - N h'
#   2. latest partitions of the built table
#   3. dry-run bytes: old full-column read vs the partition-pruned read the macro does
# Usage: ./watermark_check.sh <model> [hours=6]
# The model should contain  {{ log('WM new: ' ~ watermark, info=True) }}  right after the set line (add temporarily).
source "$(dirname "$0")/_common.sh"
[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 1; }
M="$1"; H="${2:-6}"
PCOL="$(partition_col_of "$M")"
echo "partition column: $PCOL"
echo "--- new watermark (from dbt compile)"
( cd "$DBT_PROJECT_DIR" && dbt compile -s "$M" 2>&1 | grep "WM new" )
echo "--- old watermark"
bqq "select cast(timestamp_sub(max(ingested_at), interval ${H} hour) as string) from $(TBL ${M})"
echo "--- latest partitions"
bqp "select partition_id, total_rows from ${FQ}.INFORMATION_SCHEMA.PARTITIONS
     where table_name = '${M}' and partition_id not in ('__NULL__','__UNPARTITIONED__')
     order by partition_id desc limit 3"
LD="$(bqq "select parse_date('%Y%m%d', max(partition_id)) from ${FQ}.INFORMATION_SCHEMA.PARTITIONS
           where table_name = '${M}' and partition_id not in ('__NULL__','__UNPARTITIONED__') and total_rows > 0")"
echo "--- dry-run bytes (old / new)"
DT="$(bqq "select data_type from ${FQ}.INFORMATION_SCHEMA.COLUMNS where table_name = '${M}' and column_name = '${PCOL}'")"
if [ "$DT" = "DATE" ]; then FLOOR="date_sub(date('${LD}'), interval 1 day)"; else FLOOR="timestamp_sub(timestamp('${LD}'), interval 1 day)"; fi
bq query --use_legacy_sql=false --dry_run "select max(ingested_at) from $(TBL ${M})" 2>/dev/null
bq query --use_legacy_sql=false --dry_run "select max(ingested_at) from $(TBL ${M}) where ${PCOL} >= ${FLOOR}" 2>/dev/null
