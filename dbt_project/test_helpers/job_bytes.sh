#!/usr/bin/env bash
# Bytes processed/billed of the dbt statements (service account) since a UTC time, optionally for one model.
# Needs bigquery.jobs.listAll. The jobs view lags by ~30 s, so run it a bit after the dbt run.
# Usage: ./job_bytes.sh "<UTC start, e.g. 2026-11-11 05:23:24>" [model_name]
source "$(dirname "$0")/_common.sh"
[ $# -ge 1 ] || { sed -n 2,6p "$0"; exit 1; }
START="$1"; MODEL="${2:-}"
FILTER=""; [ -n "$MODEL" ] && FILTER="and query like '%${MODEL}%'"
bqp "select format_timestamp('%H:%M:%S', creation_time) t_utc, statement_type,
            round(total_bytes_processed/1e6,1) mb_processed, round(total_bytes_billed/1e6,1) mb_billed,
            round(timestamp_diff(end_time, start_time, millisecond)/1000,1) sec,
            left(regexp_replace(query, r'\\s+', ' '), 70) q
     from \`region-${BQ_LOCATION}\`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
     where creation_time >= timestamp('${START}')
       and user_email like '%gserviceaccount.com'
       and statement_type in ('SCRIPT','CREATE_TABLE_AS_SELECT','MERGE','INSERT','SELECT')
       ${FILTER}
     order by creation_time"
