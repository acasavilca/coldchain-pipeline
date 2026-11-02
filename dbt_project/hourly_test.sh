#!/usr/bin/env bash
P=coldchain_prod
M=fct_room_conditions_hourly
F="avg_t_room_c avg_rh_room_frac stddev_rh_room_frac avg_co2_ppm avg_o2_pct avg_return_supply_air_temp_delta_c pct_time_in_temp_spec pct_time_temp_unknown pct_time_temp_downtime total_temperature_excursion_duration_sec avg_temperature_excursion_duration_sec avg_setpoint_recovery_sec_after_door_close"

if [ "$1" = "snap" ]; then
  bq query --use_legacy_sql=false --quiet "create or replace table $P.snap_$M as select * from $P.$M"
  exit 0
fi

EX=$(echo $F | tr ' ' ',')
RD=$(for c in $F; do printf "round(%s, 6) as %s," $c $c; done | sed 's/,$//')
R="select * except (dbt_updated_at, $EX), $RD"

echo -n "$M mismatched days: "
bq query --use_legacy_sql=false --format=csv --quiet "
  with a as (select reading_date_utc d, count(*) n, bit_xor(farm_fingerprint(to_json_string(t))) h from ($R from $P.$M) t group by 1),
       b as (select reading_date_utc d, count(*) n, bit_xor(farm_fingerprint(to_json_string(t))) h from ($R from $P.snap_$M) t group by 1)
  select count(*) from a full outer join b on a.d = b.d where a.d is null or b.d is null or a.n != b.n or a.h != b.h" | tail -1
