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
    {% set watermark = ingest_watermark(this, 'event_timestamp', hours=6) %}
    {% set lower_bound_sql %}
        select cast(timestamp(date(min(ts))) as string) as min_ts
        from (
            select event_timestamp as ts
            from {{ ref('int_telemetry_sensor_gap_filled') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select event_timestamp
            from {{ ref('int_crop_rotation_events_enriched') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select phase_start_ts
            from {{ ref('int_downtime_enriched') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select phase_start_ts
            from {{ ref('int_defrost_events_enriched') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select event_timestamp
            from {{ ref('int_door_events_enriched') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select event_timestamp
            from {{ ref('int_lp_trip_events_enriched') }}
            where ingested_at >= timestamp('{{ watermark }}')
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

defrost_sequence_state as (
    select
        location_id,
        phase_start_ts as event_timestamp,
        defrost_cycle_id,
        defrost_phase_code,
        termination_reason_code as defrost_termination_reason_code,
        has_missing_rows as defrost_cycle_has_missing_rows
    from {{ ref('int_defrost_events_enriched') }}
    -- {% if is_incremental() %}
    --     where phase_start_ts >= timestamp('{{ min_ts }}')
    -- {% endif %}
),

door_events_state as (
    select
        location_id,
        event_timestamp,
        any_value(open_door_count) as open_door_count,
        max(case when not is_open then event_timestamp end) as last_door_close_ts
    from {{ ref('int_door_events_enriched') }}
    -- {% if is_incremental() %}
    --     where event_timestamp >= timestamp('{{ min_ts }}')
    -- {% endif %}
    group by 1, 2
),

lp_trip_events_state as (
    select
        location_id,
        event_timestamp,
        low_p_trip_count,
        is_low_p_switch_tripped,
        low_p_trip_episode_anomalous
    from {{ ref('int_lp_trip_events_enriched') }}
    -- {% if is_incremental() %}
    --     where event_timestamp >= timestamp('{{ min_ts }}')
    -- {% endif %}
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

unioned_timeline as (
    select
        location_id,
        null as ingested_at,
        event_timestamp,
        fruit_type_code,
        null as downtime_phase_name,
        null as t_room_c,
        null as fill_status,
        null as defrost_cycle_id,
        null as defrost_phase_code,
        null as defrost_termination_reason_code,
        null as defrost_cycle_has_missing_rows,
        null as open_door_count,
        null as last_door_close_ts,
        null as low_p_trip_count,
        null as is_low_p_switch_tripped,
        null as low_p_trip_episode_anomalous,
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
        null as defrost_cycle_id,
        null as defrost_phase_code,
        null as defrost_termination_reason_code,
        null as defrost_cycle_has_missing_rows,
        null as open_door_count,
        null as last_door_close_ts,
        null as low_p_trip_count,
        null as is_low_p_switch_tripped,
        null as low_p_trip_episode_anomalous,
        'downtime_seq' as row_type
    from downtime_sequence_state

    union all

    select
        location_id,
        null as ingested_at,
        event_timestamp,
        null as fruit_type_code,
        null as downtime_phase_name,
        null as t_room_c,
        null as fill_status,
        defrost_cycle_id,
        defrost_phase_code,
        defrost_termination_reason_code,
        defrost_cycle_has_missing_rows,
        null as open_door_count,
        null as last_door_close_ts,
        null as low_p_trip_count,
        null as is_low_p_switch_tripped,
        null as low_p_trip_episode_anomalous,
        'defrost_seq' as row_type
    from defrost_sequence_state

    union all

    select
        location_id,
        null as ingested_at,
        event_timestamp,
        null as fruit_type_code,
        null as downtime_phase_name,
        null as t_room_c,
        null as fill_status,
        null as defrost_cycle_id,
        null as defrost_phase_code,
        null as defrost_termination_reason_code,
        null as defrost_cycle_has_missing_rows,
        open_door_count,
        last_door_close_ts,
        null as low_p_trip_count,
        null as is_low_p_switch_tripped,
        null as low_p_trip_episode_anomalous,
        'door_events' as row_type
    from door_events_state

    union all

    select
        location_id,
        null as ingested_at,
        event_timestamp,
        null as fruit_type_code,
        null as downtime_phase_name,
        null as t_room_c,
        null as fill_status,
        null as defrost_cycle_id,
        null as defrost_phase_code,
        null as defrost_termination_reason_code,
        null as defrost_cycle_has_missing_rows,
        null as open_door_count,
        null as last_door_close_ts,
        low_p_trip_count,
        is_low_p_switch_tripped,
        low_p_trip_episode_anomalous,
        'lp_trip_events' as row_type
    from lp_trip_events_state

    union all

    select
        location_id,
        ingested_at,
        event_timestamp,
        null as fruit_type_code,
        null as downtime_phase_name,
        t_room_c,
        fill_status,
        null as defrost_cycle_id,
        null as defrost_phase_code,
        null as defrost_termination_reason_code,
        null as defrost_cycle_has_missing_rows,
        null as open_door_count,
        null as last_door_close_ts,
        null as low_p_trip_count,
        null as is_low_p_switch_tripped,
        null as low_p_trip_episode_anomalous,
        'reading' as row_type
    from telemetry_source_slice
)

select
    location_id,
    ingested_at,
    event_timestamp,
    t_room_c,
    fill_status,
    last_value(fruit_type_code ignore nulls) over w as active_fruit_type_code,
    last_value(downtime_phase_name ignore nulls) over w as downtime_phase_name,
    last_value(defrost_cycle_id ignore nulls) over w as defrost_cycle_id,
    last_value(defrost_phase_code ignore nulls) over w as defrost_phase_code,
    last_value(defrost_termination_reason_code ignore nulls) over w as defrost_termination_reason_code,
    last_value(defrost_cycle_has_missing_rows ignore nulls) over w as defrost_cycle_has_missing_rows,
    last_value(open_door_count ignore nulls) over w as open_door_count,
    last_value(last_door_close_ts ignore nulls) over w as last_door_close_ts,
    last_value(low_p_trip_count ignore nulls) over w as low_p_trip_count,
    last_value(is_low_p_switch_tripped ignore nulls) over w as is_low_p_switch_tripped,
    last_value(low_p_trip_episode_anomalous ignore nulls) over w as low_p_trip_episode_anomalous
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
