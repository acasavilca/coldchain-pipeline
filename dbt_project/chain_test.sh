#!/usr/bin/env bash
P=coldchain_prod
M="int_temperature_telemetry_labeled int_room_temperature_compliance_readings"

if [ "$1" = "snap" ]; then
  for m in $M; do
    bq query --use_legacy_sql=false --quiet "create or replace table $P.snap_$m as select * from $P.$m"
  done
  exit 0
fi

for m in $M; do
  echo -n "$m mismatched days: "
  bq query --use_legacy_sql=false --format=csv --quiet "
    with a as (select date(event_timestamp) d, count(*) n, bit_xor(farm_fingerprint(to_json_string(t))) h from $P.$m t group by 1),
         b as (select date(event_timestamp) d, count(*) n, bit_xor(farm_fingerprint(to_json_string(t))) h from $P.snap_$m t group by 1)
    select count(*) from a full outer join b on a.d = b.d where a.d is null or b.d is null or a.n != b.n or a.h != b.h" | tail -1
done
