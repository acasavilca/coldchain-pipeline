#!/usr/bin/env bash
# Copy one or more built tables to snap_<model> so a later run can be diffed against them.
# Usage: ./snapshot_table.sh <model> [<model> ...]
source "$(dirname "$0")/_common.sh"
[ $# -ge 1 ] || { echo "usage: $0 <model> [<model> ...]"; exit 1; }
for m in "$@"; do
  bq query --use_legacy_sql=false --quiet \
    "create or replace table $(TBL snap_${m}) as select * from $(TBL ${m})" 2>/dev/null \
    && echo "snapshot: snap_${m}"
done
