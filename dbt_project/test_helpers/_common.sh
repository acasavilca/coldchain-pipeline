#!/usr/bin/env bash
# Shared settings/helpers, sourced by the other scripts. Override with env vars.
#   BQ_PROJECT   (default fruit-packing-plant-simulator)
#   BQ_DATASET   (default coldchain_dev)   e.g. BQ_DATASET=coldchain_prod ./diff_vs_snapshot.sh ...
#   BQ_LOCATION  (default us)              used for INFORMATION_SCHEMA.JOBS_* region qualifiers
BQ_PROJECT="${BQ_PROJECT:-fruit-packing-plant-simulator}"
BQ_DATASET="${BQ_DATASET:-coldchain_dev}"
BQ_LOCATION="${BQ_LOCATION:-us}"
DBT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FQ="\`${BQ_PROJECT}.${BQ_DATASET}\`"   # backticked project.dataset, for use inside SQL strings

# TBL <name> -> backticked project.dataset.name
TBL() { echo "\`${BQ_PROJECT}.${BQ_DATASET}.$1\`"; }

# bqq "<sql>"  -> CSV rows without the header line, bq warnings suppressed
bqq() { bq query --use_legacy_sql=false --format=csv --quiet "$1" 2>/dev/null | tail -n +2; }
# bqp "<sql>"  -> pretty table
bqp() { bq query --use_legacy_sql=false --format=pretty --quiet "$1" 2>/dev/null; }

# partition_col_of <model> -> name of the table's partitioning column
partition_col_of() {
  bqq "select column_name from ${FQ}.INFORMATION_SCHEMA.COLUMNS
       where table_name = '$1' and is_partitioning_column = 'YES' limit 1"
}
