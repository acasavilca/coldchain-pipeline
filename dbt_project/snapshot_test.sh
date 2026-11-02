#!/usr/bin/env bash

for m in int_telemetry_sensor_gap_filled int_telemetry_sparse_pivot int_batch_sensors_sparse_pivot int_facility_events_joined int_batch_sensors_with_room_conditions int_order_fulfillment; do
  bq query --use_legacy_sql=false "create or replace table coldchain_prod.snap_$m as select * from coldchain_prod.$m"
done
