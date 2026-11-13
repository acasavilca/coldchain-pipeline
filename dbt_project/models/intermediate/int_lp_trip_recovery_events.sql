{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "lp_trip_switch_off_ts",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id"]
) }}

{% if is_incremental() and execute %}
    {% set checkpoint_table = ensure_checkpoint_table(this) %}
    {% set watermark = ingest_watermark(checkpoint_table, 'checkpoint_day', hours=6, ts_type='date') %}

    {% set raw_lower_bound_sql %}
        select cast(timestamp(date(min(ts))) as string) as raw_min_ts
        from (
            select event_timestamp as ts
            from {{ ref('int_room_temperature_compliance_readings') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select event_timestamp as ts
            from {{ ref('int_lp_trip_events_enriched') }}
            where ingested_at >= timestamp('{{ watermark }}')
        )
    {% endset %}
    {% set b = run_query(raw_lower_bound_sql) %}
    {% set raw_min_ts = b.columns[0].values()[0] %}
    {% if raw_min_ts is none %}
        {% set raw_min_ts = '2100-01-01 00:00:00+00:00' %}
    {% endif %}
    {% if var('min_ts_override', none) %}
        {% set raw_min_ts = var('min_ts_override') %}
    {% endif %}
    {% set true_lower_bound_sql %}
        select cast(min(if(
            last_lp_trip_is_open,
            least(timestamp('{{ raw_min_ts }}'), timestamp(date(last_lp_trip_switch_off_ts))),
            timestamp('{{ raw_min_ts }}')
        )) as string) as min_ts
        from (
            select * from {{ checkpoint_table }}
            qualify row_number() over (partition by location_id order by checkpoint_day desc) = 1
        )
    {% endset %}
    {% set a = run_query(true_lower_bound_sql) %}
    {% set min_ts = a.columns[0].values()[0] %}
    {% if min_ts is none %}{% set min_ts = raw_min_ts %}{% endif %}
{% endif %}

with temperature_compliance_readings_slice as (
    select
        location_id,
        event_timestamp,
        temperature_compliance_status,
        ingested_at
    from {{ ref('int_room_temperature_compliance_readings') }}
    {% if is_incremental() %}
        where event_timestamp >= timestamp('{{ min_ts }}')
    {% endif %}
),

lp_trip_off_events_slice as (
    select
        location_id,
        low_p_trip_episode_id,
        event_timestamp,
        max(ingested_at) as ingested_at
    from {{ ref('int_lp_trip_events_enriched') }}
    where
        not is_low_p_switch_tripped
        {% if is_incremental() %}
            and event_timestamp >= timestamp('{{ min_ts }}')
        {% endif %}
    group by 1, 2, 3
),

unioned as (
    select
        location_id,
        null as low_p_trip_episode_id,
        event_timestamp as event_ts,
        null as lp_trip_switch_off_ts,
        temperature_compliance_status,
        'reading' as row_type,
        ingested_at
    from temperature_compliance_readings_slice

    union all

    select
        location_id,
        low_p_trip_episode_id,
        event_timestamp as event_ts,
        event_timestamp as lp_trip_switch_off_ts,
        null as temperature_compliance_status,
        'lp_trip_event' as row_type,
        ingested_at
    from lp_trip_off_events_slice
),

forward_filled as (
    select
        location_id,
        event_ts,
        temperature_compliance_status,
        row_type,
        ingested_at,
        last_value(low_p_trip_episode_id ignore nulls) over w as low_p_trip_episode_id,
        last_value(lp_trip_switch_off_ts ignore nulls) over w as lp_trip_switch_off_ts
    from unioned
    window
        w as (
            partition by location_id
            order by
                event_ts,
                case when row_type = 'lp_trip_event' then 0 else 1 end
            rows between unbounded preceding and current row
        )
),

lp_trip_off_disrupted as (
    select distinct
        location_id,
        lp_trip_switch_off_ts
    from forward_filled
    where
        row_type = 'reading'
        and event_ts >= lp_trip_switch_off_ts
    qualify
        row_number() over (partition by location_id, lp_trip_switch_off_ts order by event_ts asc) = 1
        and temperature_compliance_status = 'out_of_spec'
)

select
    ff.location_id,
    ff.low_p_trip_episode_id,
    ff.lp_trip_switch_off_ts,
    min(case when ff.temperature_compliance_status = 'in_spec' then ff.event_ts end) as recovery_ts,
    max(ff.ingested_at) as ingested_at
from forward_filled as ff
inner join lp_trip_off_disrupted as lpd
    on
        ff.location_id = lpd.location_id
        and ff.lp_trip_switch_off_ts = lpd.lp_trip_switch_off_ts
where
    ff.row_type = 'reading'
    and ff.event_ts > ff.lp_trip_switch_off_ts
group by 1, 2, 3
