{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "event_timestamp",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id"]
) }}

{% if is_incremental() and execute %}
    {% set lower_bound_sql %}
        select cast(timestamp(date(min(ts))) as string) as min_ts
        from (
            select event_timestamp as ts
            from {{ ref('int_telemetry_sensor_gap_filled') }}
            where ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
            {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
            {% endif %}

            union all

            select event_timestamp
            from {{ ref('int_crop_rotation_events_enriched') }}
            where ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
            {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
            {% endif %}

            union all

            select phase_start_ts
            from {{ ref('int_downtime_enriched') }}
            where ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
            {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
            {% endif %}
        )
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

with crop_rotation_state as (
    select
        location_id,
        event_timestamp,
        fruit_type_code
    from {{ ref('int_crop_rotation_events_enriched') }}
),

downtime_sequence_state as (
    select
        location_id,
        phase_start_ts as event_timestamp,
        phase_name as downtime_phase_name
    from {{ ref('int_downtime_enriched') }}
    where
        phase_start_ts is not null
        and (phase_start_ts != phase_end_ts or phase_end_ts is null)
),

telemetry_source_slice as (
    select
        location_id,
        ingested_at,
        event_timestamp,
        value_filled as t_room_c,
        fill_status
    from {{ ref('int_telemetry_sensor_gap_filled') }}
    where
        tag_name = 't_room_c'
        {% if is_incremental() %}
            and event_timestamp >= timestamp('{{ min_ts }}')
        {% endif %}
),

-- telemetry_gap_filled as (
--     select
--         location_id,
--         ingested_at,
--         event_timestamp,
--         value_filled as t_room_c,
--         fill_status
--     from {{ ref('int_telemetry_sensor_gap_filled') }}
--     where tag_name = 't_room_c'
-- ),

unioned_timeline as (
    select
        location_id,
        null as ingested_at,
        event_timestamp,
        fruit_type_code,
        null as downtime_phase_name,
        null as t_room_c,
        null as fill_status,
        'rotation' as row_type
    from crop_rotation_state

    union all

    select
        location_id,
        null as ingested_at,
        event_timestamp,
        null as fruit_type_code,
        downtime_phase_name,
        null as t_room_c,
        null as fill_status,
        'downtime_seq' as row_type
    from downtime_sequence_state

    union all

    select
        location_id,
        ingested_at,
        event_timestamp,
        null as fruit_type_code,
        null as downtime_phase_name,
        t_room_c,
        fill_status,
        'reading' as row_type
    from telemetry_source_slice
)

select
    * except (fruit_type_code, downtime_phase_name),
    last_value(fruit_type_code ignore nulls) over w as active_fruit_type_code,
    last_value(downtime_phase_name ignore nulls) over w as downtime_phase_name
from unioned_timeline
qualify
    row_type = 'reading'
    {% if is_incremental() %}
        and event_timestamp >= timestamp('{{ min_ts }}')
    {% endif %}
window
    w as (
        partition by location_id
        order by
            event_timestamp,
            case when row_type = 'reading' then 1 else 0 end
        rows between unbounded preceding and current row
    )
