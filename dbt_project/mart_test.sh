#!/usr/bin/env bash
P=coldchain_prod
check() {
  m=$1; col=$2
  echo -n "$m mismatched days: "
  bq query --use_legacy_sql=false --format=csv --quiet "
  with a as (select $col d, count(*) n, bit_xor(farm_fingerprint(to_json_string(t))) h
             from (select * except (dbt_updated_at) from $P.$m) t group by 1),
       b as (select $col d, count(*) n, bit_xor(farm_fingerprint(to_json_string(t))) h
             from (select * except (dbt_updated_at) from $P.snap_$m) t group by 1)
  select count(*) from a full outer join b on a.d = b.d
  where a.d is null or b.d is null or a.n != b.n or a.h != b.h" | tail -1
}
check fct_pallet_movements arrival_date_utc
check fct_outbound_orders_fullfillment order_date_utc
check fct_compressor_performance reading_date_utc
