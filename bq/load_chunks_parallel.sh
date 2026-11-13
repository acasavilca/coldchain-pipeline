#!/usr/bin/env bash
# Loads one simulator run directory (one location) into BigQuery, several tables/chunks at a time,
# and skips anything that was already loaded. Same checks as load_chunks.sh, which stays as is.
#
# Usage:  bash bq/load_chunks_parallel.sh RUN_DIR FIRST [LAST]
#         JOBS=8 bash bq/load_chunks_parallel.sh simulation/runs/location_1 0 29
#         DRY_RUN=1 bash bq/load_chunks_parallel.sh simulation/runs/location_1 0 29   # list the jobs only
# Run from the repository root. Needs `bq` authenticated for the target project.
# Weather is not handled here: use `bash bq/load_chunks.sh --weather LOCATION_ID`.
#
# Env: PROJECT (default fruit-packing-plant-simulator), DATASET (default coldchain_raw, the dataset
#      create_tables.sh and dbt use; load_chunks.sh defaults to coldchain_scratch), JOBS (default 6).
#
# How "already loaded" works: every (chunk, table) load gets a deterministic BigQuery job id built from
# dataset, location, chunk, table and the first 12 chars of the parquet's sha256 from the manifest.
# Before loading, the script asks BigQuery for that job: DONE -> skip; running -> skip; not found -> load;
# failed -> try the same id with _r1, _r2, ... (a failed job id cannot be reused). BigQuery keeps job
# history for about 180 days, and it only knows loads made with this script (earlier loads were
# made without job ids). If you drop and recreate a table, its old job ids still say DONE: use a new
# DATASET, or set FORCE=1 to ignore the history and load everything again (appends, so duplicates the rows
# of tables that still hold them).
set -euo pipefail
export PYTHONWARNINGS=ignore

PROJECT="${PROJECT:-fruit-packing-plant-simulator}"
DATASET="${DATASET:-coldchain_raw}"
JOBS="${JOBS:-6}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

TABLES=(
  telemetry batch_sensors
  compressor_events humidifier_events door_events
  crop_rotation_events refrigerant_rotation_events downtime_sequence_events
  lp_trip_events defrost_events
  inbound_orders outbound_orders inbound_pallet_history outbound_pallet_history
)

RUN_DIR="${1:?usage: load_chunks_parallel.sh RUN_DIR FIRST [LAST]}"
FIRST="${2:?usage: load_chunks_parallel.sh RUN_DIR FIRST [LAST]}"
LAST="${3:-$FIRST}"
[[ -f "$RUN_DIR/run_meta.json" ]] || { echo "$RUN_DIR has no run_meta.json (not a run directory)" >&2; exit 1; }

SCHEMAS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/schemas"
TASKS="$(mktemp)"
RESULTS="$(mktemp)"
trap 'rm -f "$TASKS" "$RESULTS"' EXIT

# Verify first, load second: nothing is loaded if any requested chunk fails its manifest check.
# Writes one tab-separated line per (chunk, table): file, table, job id, label.
python3 - "$RUN_DIR" "$FIRST" "$LAST" "$SCHEMAS_DIR" "$DATASET" "$TASKS" "${TABLES[*]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
import pyarrow.parquet as pq
run_dir, first, last = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
schemas_dir, dataset, tasks_file = Path(sys.argv[4]), sys.argv[5], Path(sys.argv[6])
tables = sys.argv[7].split()
SCHEMA_OF = {"inbound_orders": "orders", "outbound_orders": "orders",
             "inbound_pallet_history": "pallet_history", "outbound_pallet_history": "pallet_history"}
loc = json.loads((run_dir / "run_meta.json").read_text())["location_id"]
lines = []
for i in range(first, last + 1):
    d = run_dir / "raw_parquets" / f"chunk_{i:04d}"
    m = json.loads((d / "run_manifest.json").read_text())
    if m["location_id"] != loc:
        sys.exit(f"{d}: manifest location {m['location_id']} != run directory location {loc}")
    missing = [t for t in tables if t not in m["row_counts"]]
    if missing:
        sys.exit(f"{d}: tables missing from the manifest: {missing}")
    for name, rows in m["row_counts"].items():
        f = d / f"{name}.parquet"
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != m["sha256"][name]:
            sys.exit(f"{f}: checksum differs from the manifest")
        pf = pq.ParquetFile(f)
        if pf.metadata.num_rows != rows:
            sys.exit(f"{f}: row count differs from the manifest")
        table_cols = {c["name"] for c in json.loads((schemas_dir / f"{SCHEMA_OF.get(name, name)}.json").read_text())}
        extra = sorted(set(pf.schema_arrow.names) - table_cols)
        if extra:
            sys.exit(f"{f}: columns not in the BigQuery table schema: {extra}")
    for name in tables:
        f = d / f"{name}.parquet"
        job = f"ld_{dataset}_loc{loc}_c{i:04d}_{name}_{m['sha256'][name][:12]}"
        lines.append(f"{f}\t{name}\t{job}\tloc{loc}/chunk_{i:04d}/{name}")
tasks_file.write_text("\n".join(lines) + "\n")
print(f"manifests ok for chunks {first}..{last} of location {loc}: {len(lines)} load jobs", file=sys.stderr)
PY

echo "target: $PROJECT:$DATASET, up to $JOBS parallel loads"
if [[ "$DRY_RUN" == "1" ]]; then
  cut -f4 "$TASKS"
  exit 0
fi

# the dataset's location, needed to look up job ids
LOCATION="$(bq --project_id="$PROJECT" show --format=prettyjson "$PROJECT:$DATASET" |
  python3 -c 'import sys, json; print(json.load(sys.stdin)["location"])')"
[[ -n "$LOCATION" ]] || { echo "could not read the location of $PROJECT:$DATASET" >&2; exit 1; }
export PROJECT DATASET LOCATION RESULTS FORCE

# state of a job id: NONE | RUNNING | DONE | FAILED
job_state() {
  bq --project_id="$PROJECT" --location="$LOCATION" show --format=json -j "$1" 2>/dev/null |
    python3 -c '
import sys, json
try:
    s = json.load(sys.stdin)["status"]
    print("FAILED" if "errorResult" in s else ("DONE" if s["state"] == "DONE" else "RUNNING"))
except Exception:
    print("NONE")' || true
}

load_one() {
  local file="$1" table="$2" base="$3" label="$4" job state k
  job="$base"
  if [[ "$FORCE" == "1" ]]; then
    job="${base}_f$(date +%s)"
  else
    for k in $(seq 0 20); do
      [[ $k -gt 0 ]] && job="${base}_r$k"
      state="$(job_state "$job")"
      case "$state" in
        DONE)    echo "SKIP $label (already loaded: $job)" | tee -a "$RESULTS"; return 0 ;;
        RUNNING) echo "SKIP $label (job still running: $job)" | tee -a "$RESULTS"; return 0 ;;
        NONE)    break ;;
        FAILED)  continue ;;
      esac
    done
  fi
  if bq --project_id="$PROJECT" --location="$LOCATION" load --job_id="$job" --noreplace \
        --source_format=PARQUET "$DATASET.$table" "$file" >/dev/null 2>"$RESULTS.$$.err"; then
    echo "LOAD $label" | tee -a "$RESULTS"
  else
    echo "FAIL $label: $(tail -n 2 "$RESULTS.$$.err" | tr '\n' ' ')" | tee -a "$RESULTS" >&2
    rm -f "$RESULTS.$$.err"
    return 1
  fi
  rm -f "$RESULTS.$$.err"
}

fail=0
while IFS=$'\t' read -r file table job label; do
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || fail=1; done
  load_one "$file" "$table" "$job" "$label" &
done < "$TASKS"
for p in $(jobs -rp); do wait "$p" || fail=1; done

echo "loaded: $(grep -c '^LOAD ' "$RESULTS" || true), skipped: $(grep -c '^SKIP ' "$RESULTS" || true), failed: $(grep -c '^FAIL ' "$RESULTS" || true)"
if [[ "$fail" -ne 0 ]]; then
  echo "some loads failed: fix the cause and rerun the same command (loaded ones are skipped)" >&2
  exit 1
fi
