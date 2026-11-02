#!/usr/bin/env bash
# Appends the raw parquet tables of simulator chunks to BigQuery.
#
# Usage:  bash bq/load_chunks.sh FIRST [LAST]     e.g.  bash bq/load_chunks.sh 0 3
#         bash bq/load_chunks.sh --weather        replaces the weather table from simulation/raw_weather_data.parquet
# Run from the repository root. Needs `bq` authenticated for the target project.
#
# Chunks are loaded with --noreplace (append), so loading the same chunk twice duplicates its rows.
# The dbt staging models keep only the latest row per key, so duplicates are harmless downstream,
# but they do re-trigger incremental recomputation for those days.
set -euo pipefail

PROJECT="${PROJECT:-fruit-packing-plant-simulator}"
DATASET="${DATASET:-coldchain_raw}"
RAW_DIR="${RAW_DIR:-simulation/raw_parquets}"

TABLES=(
  telemetry batch_sensors
  compressor_events humidifier_events door_events
  crop_rotation_events refrigerant_rotation_events downtime_sequence_events
  lp_trip_events defrost_events
  inbound_orders outbound_orders inbound_pallet_history outbound_pallet_history
)

if [[ "${1:-}" == "--weather" ]]; then
  bq load --project_id="$PROJECT" --replace --source_format=PARQUET \
    "$DATASET.weather_data" simulation/raw_weather_data.parquet
  exit 0
fi

FIRST="${1:?usage: load_chunks.sh FIRST [LAST] | --weather}"
LAST="${2:-$FIRST}"

for ((i = FIRST; i <= LAST; i++)); do
  chunk=$(printf "chunk_%04d" "$i")
  for table in "${TABLES[@]}"; do
    file="$RAW_DIR/$chunk/$table.parquet"
    if [[ ! -f "$file" ]]; then
      echo "missing $file" >&2
      exit 1
    fi
    echo "loading $chunk/$table"
    bq load --project_id="$PROJECT" --noreplace --source_format=PARQUET "$DATASET.$table" "$file"
  done
done
