{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "door_close_ts",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id"]
) }}

{% if is_incremental() and execute %}
    {% set raw_lower_bound_sql %}
        select cast(timestamp(date(min(ts))) as string) as raw_min_ts
        from (
            select event_timestamp as ts
            from {{ ref('int_room_temperature_compliance_readings') }}
            where ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
            {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
            {% endif %}

            union all

            select event_timestamp as ts
            from {{ ref('int_door_events_enriched') }}
            where ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
            {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
            {% endif %}
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
        select cast(timestamp(date(min(ts))) as string) as min_ts
        from (
            select max(event_timestamp) as ts
            from {{ ref('int_door_events_enriched') }}
            where not is_open and event_timestamp < timestamp('{{ raw_min_ts }}')
            group by location_id
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

door_close_events_slice as (
    select
        location_id,
        event_timestamp,
        max(ingested_at) as ingested_at
    from {{ ref('int_door_events_enriched') }}
    where
        is_open = false
        {% if is_incremental() %}
            and event_timestamp >= timestamp('{{ min_ts }}')
        {% endif %}
    group by 1, 2
),

unioned as (
    select
        location_id,
        event_timestamp as event_ts,
        null as door_close_ts,
        temperature_compliance_status,
        'reading' as row_type,
        ingested_at
    from temperature_compliance_readings_slice

    union all

    select
        location_id,
        event_timestamp as event_ts,
        event_timestamp as door_close_ts,
        null as temperature_compliance_status,
        'door_event' as row_type,
        ingested_at
    from door_close_events_slice
),

forward_filled as (
    select
        location_id,
        event_ts,
        temperature_compliance_status,
        row_type,
        ingested_at,
        last_value(door_close_ts ignore nulls) over (
            partition by location_id
            order by
                event_ts,
                case when row_type = 'door_event' then 0 else 1 end
            rows between unbounded preceding and current row
        ) as door_close_ts
    from unioned
),

door_close_disrupted as (
    select distinct
        location_id,
        door_close_ts
    from forward_filled
    where
        row_type = 'reading'
        and event_ts >= door_close_ts
        and temperature_compliance_status = 'out_of_spec'
)

select
    ff.location_id,
    ff.door_close_ts,
    min(case when ff.temperature_compliance_status = 'in_spec' then ff.event_ts end) as recovery_ts,
    -- max(ff.event_ts) as max_event_ts,
    max(ff.ingested_at) as ingested_at
from forward_filled as ff
inner join door_close_disrupted as dcd
    on
        ff.location_id = dcd.location_id
        and ff.door_close_ts = dcd.door_close_ts
where
    ff.row_type = 'reading'
    and ff.event_ts > ff.door_close_ts
group by 1, 2
