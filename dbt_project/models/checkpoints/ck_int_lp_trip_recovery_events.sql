{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "checkpoint_day",
      "data_type": "date",
      "granularity": "day"
    },
    cluster_by=["location_id"]
) }}

{% if is_incremental() and execute %}
    {% set lower_bound_sql %}
        select cast(timestamp(date(min(event_timestamp))) as string) as min_ts
        from {{ ref('int_room_temperature_compliance_readings') }}
        where ingested_at >= {% if var('cutoff_override', none) %}
            timestamp('{{ var("cutoff_override") }}')
        {% else %}
            (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
        {% endif %}
    {% endset %}
    {% set b = run_query(lower_bound_sql) %}
    {% set min_ts = b.columns[0].values()[0] %}
    {% if min_ts is none %}
        {% set min_ts = '2100-01-01 00:00:00+00:00' %}
    {% endif %}
    {% if var('min_ts_override', none) %}
        {% set min_ts = var('min_ts_override') %}
    {% endif %}
{% endif %}

with day_partitioned_readings_slice as (
    select
        location_id,
        ingested_at,
        date(event_timestamp) as checkpoint_day,
        max(event_timestamp) over (partition by location_id) as latest_reading_ts
    from {{ ref('int_room_temperature_compliance_readings') }}
    {% if is_incremental() %}
        where event_timestamp >= timestamp('{{ min_ts }}')
    {% endif %}
    qualify row_number() over (partition by location_id, date(event_timestamp) order by event_timestamp desc) = 1
),

recovery_events_slice as (
    select
        location_id,
        lp_trip_switch_off_ts
    from {{ ref('int_lp_trip_recovery_events') }}
    where recovery_ts is null
    -- qualify row_number() over (partition by location_id, date() order by lp_trip_switch_off_ts asc) = 1
),

lp_trip_events_slice as (
    select
        location_id,
        event_timestamp as last_lp_trip_switch_off_ts
    from {{ ref('int_lp_trip_events_enriched') }}
    where not is_low_p_switch_tripped
    qualify row_number() over (partition by location_id order by event_timestamp desc) = 1
)

select
    d.location_id,
    d.ingested_at,
    d.checkpoint_day,
    lp.last_lp_trip_switch_off_ts,
    coalesce(
        (d.latest_reading_ts <= lp.last_lp_trip_switch_off_ts or r.lp_trip_switch_off_ts is not null),
        false
    ) as last_lp_trip_is_open
from day_partitioned_readings_slice as d
left join lp_trip_events_slice as lp
    on d.location_id = lp.location_id
left join recovery_events_slice as r
    on
        lp.location_id = r.location_id
        and lp.last_lp_trip_switch_off_ts = r.lp_trip_switch_off_ts
