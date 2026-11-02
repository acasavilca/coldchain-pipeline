#!/usr/bin/env bash
# Creates the raw BigQuery dataset and tables the simulator's parquet output is loaded into.
# Safe to re-run: tables that already exist are left alone (--force).
#
# Usage (from anywhere):  bash bq/create_tables.sh
# Override the target:    PROJECT=my-project DATASET=my_raw bash bq/create_tables.sh
set -uo pipefail

PROJECT="${PROJECT:-fruit-packing-plant-simulator}"
DATASET="${DATASET:-coldchain_raw}"
SCHEMAS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/schemas"

bq mk --dataset --force --project_id="$PROJECT" "$PROJECT:$DATASET"

# make_table <table> <schema file> <clustering fields> [partition field]
make_table() {
  local table="$1" schema="$2" clustering="$3" partition="${4:-event_timestamp}"
  bq mk --table --force --project_id="$PROJECT" \
    --time_partitioning_field="$partition" --time_partitioning_type=DAY \
    --clustering_fields="$clustering" \
    "$DATASET.$table" "$SCHEMAS_DIR/$schema.json"
}

make_table telemetry                    telemetry                    loaded_at,tag_name
make_table batch_sensors                batch_sensors                loaded_at,batch_id,tag_name
make_table compressor_events            compressor_events            loaded_at
make_table humidifier_events            humidifier_events            loaded_at
make_table door_events                  door_events                  loaded_at,door_code
make_table crop_rotation_events         crop_rotation_events         loaded_at
make_table refrigerant_rotation_events  refrigerant_rotation_events  loaded_at
make_table downtime_sequence_events     downtime_sequence_events     loaded_at
make_table lp_trip_events               lp_trip_events               loaded_at
make_table defrost_events               defrost_events               loaded_at
make_table inbound_orders               orders                       loaded_at,fruit_type_code
make_table outbound_orders              orders                       loaded_at,fruit_type_code
make_table inbound_pallet_history       pallet_history               loaded_at,quality_grade
make_table outbound_pallet_history      pallet_history               loaded_at,quality_grade
make_table weather_data                 weather_data                 loaded_at timestamp
