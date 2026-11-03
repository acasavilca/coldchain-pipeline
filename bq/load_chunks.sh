#!/usr/bin/env bash
# Loads one simulator run directory (one location) into BigQuery.
#
# Usage:  bash bq/load_chunks.sh RUN_DIR FIRST [LAST]   e.g.  bash bq/load_chunks.sh simulation/runs/location_2 0 3
#         bash bq/load_chunks.sh --weather LOCATION_ID   replaces that location's weather rows
#                                                         (from simulation/weather/location_<id>/raw_weather_data.parquet)
# Run from the repository root. Needs `bq` authenticated for the target project.
#
# Chunks are loaded with --noreplace (append), so loading the same chunk twice duplicates its rows.
# The dbt staging models keep only the latest row per key, so duplicates are harmless downstream,
# but they do re-trigger incremental recomputation for those days.
# Every chunk's run_manifest.json is checked first (row counts and file checksums), and a chunk
# whose manifest location does not match the run directory is refused.
set -euo pipefail

PROJECT="${PROJECT:-fruit-packing-plant-simulator}"
DATASET="${DATASET:-coldchain_raw}"

TABLES=(
  telemetry batch_sensors
  compressor_events humidifier_events door_events
  crop_rotation_events refrigerant_rotation_events downtime_sequence_events
  lp_trip_events defrost_events
  inbound_orders outbound_orders inbound_pallet_history outbound_pallet_history
)

if [[ "${1:-}" == "--weather" ]]; then
  LOC="${2:?usage: load_chunks.sh --weather LOCATION_ID}"
  [[ "$LOC" =~ ^[0-9]+$ ]] || { echo "LOCATION_ID must be a number" >&2; exit 1; }
  file="simulation/weather/location_$LOC/raw_weather_data.parquet"
  [[ -f "$file" ]] || { echo "missing $file" >&2; exit 1; }
  # delete this location's rows, then append: other locations' weather is untouched
  bq query --project_id="$PROJECT" --use_legacy_sql=false \
    "DELETE FROM \`$PROJECT.$DATASET.weather_data\` WHERE location_id = $LOC"
  bq load --project_id="$PROJECT" --noreplace --source_format=PARQUET "$DATASET.weather_data" "$file"
  exit 0
fi

RUN_DIR="${1:?usage: load_chunks.sh RUN_DIR FIRST [LAST] | --weather LOCATION_ID}"
FIRST="${2:?usage: load_chunks.sh RUN_DIR FIRST [LAST]}"
LAST="${3:-$FIRST}"
RAW_DIR="$RUN_DIR/raw_parquets"
[[ -f "$RUN_DIR/run_meta.json" ]] || { echo "$RUN_DIR has no run_meta.json (not a run directory)" >&2; exit 1; }

# Verify first, load second: nothing is loaded if any requested chunk fails its manifest check.
python3 - "$RUN_DIR" "$FIRST" "$LAST" <<'PY'
import hashlib, json, sys
from pathlib import Path
import pyarrow.parquet as pq
run_dir, first, last = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
loc = json.loads((run_dir / "run_meta.json").read_text())["location_id"]
for i in range(first, last + 1):
    d = run_dir / "raw_parquets" / f"chunk_{i:04d}"
    m = json.loads((d / "run_manifest.json").read_text())
    if m["location_id"] != loc:
        sys.exit(f"{d}: manifest location {m['location_id']} != run directory location {loc}")
    for name, rows in m["row_counts"].items():
        f = d / f"{name}.parquet"
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != m["sha256"][name]:
            sys.exit(f"{f}: checksum differs from the manifest")
        if pq.ParquetFile(f).metadata.num_rows != rows:
            sys.exit(f"{f}: row count differs from the manifest")
print(f"manifests ok for chunks {first}..{last} of location {loc}")
PY

for ((i = FIRST; i <= LAST; i++)); do
  chunk=$(printf "chunk_%04d" "$i")
  for table in "${TABLES[@]}"; do
    file="$RAW_DIR/$chunk/$table.parquet"
    if [[ ! -f "$file" ]]; then
      echo "missing $file" >&2
      exit 1
    fi
    echo "loading $RUN_DIR $chunk/$table"
    bq load --project_id="$PROJECT" --noreplace --source_format=PARQUET "$DATASET.$table" "$file"
  done
done
