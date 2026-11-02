#!/usr/bin/env bash
P=coldchain_prod
for m in int_batch_sensors_sparse_pivot int_facility_events_joined int_batch_sensors_with_room_conditions; do
  for pair in "$m snap_$m" "snap_$m $m"; do
    set -- $pair
    echo -n "$1 minus $2: "
    bq query --use_legacy_sql=false --format=csv --quiet \
      "select count(*) from (select * from $P.$1 except distinct select * from $P.$2)" | tail -1
  done
done

R="select * except (total_net_mass_shipped_kg, fill_rate_mass), round(total_net_mass_shipped_kg, 6) as total_net_mass_shipped_kg, round(fill_rate_mass, 9) as fill_rate_mass"
for pair in "int_order_fulfillment snap_int_order_fulfillment" "snap_int_order_fulfillment int_order_fulfillment"; do
  set -- $pair
  echo -n "$1 minus $2: "
  bq query --use_legacy_sql=false --format=csv --quiet \
    "select count(*) from ($R from $P.$1 except distinct $R from $P.$2)" | tail -1
done
