{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "event_timestamp",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "active_fruit_type_code", "downtime_phase_name"]
) }}

{% if is_incremental() and execute %}
    {% set watermark = ingest_watermark(this, 'event_timestamp', hours=6) %}
    {% set checkpoint_table = ensure_checkpoint_table(this) %}

    {% set lower_bound_sql %}
        select cast(timestamp(date(min(event_timestamp))) as string) as min_ts
        from {{ ref('int_temperature_telemetry_labeled') }}
        where ingested_at >= timestamp('{{ watermark }}')
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

with source_slice as (
    select
        location_id,
        ingested_at,
        event_timestamp,
        active_fruit_type_code,
        t_room_c,
        downtime_phase_name,
        defrost_cycle_id,
        defrost_phase_code,
        defrost_termination_reason_code,
        defrost_cycle_has_missing_rows,
        open_door_count,
        last_door_close_ts,
        low_p_trip_count,
        is_low_p_switch_tripped,
        low_p_trip_episode_anomalous
    from {{ ref('int_temperature_telemetry_labeled') }}
    {% if is_incremental() %}
        where event_timestamp >= timestamp('{{ min_ts }}')
    {% endif %}
),

seed_rows as (
    {% if is_incremental() %}
        select
            location_id,
            ingested_at,
            event_timestamp,
            active_fruit_type_code,
            t_room_c,
            downtime_phase_name,
            defrost_cycle_id,
            defrost_phase_code,
            defrost_termination_reason_code,
            defrost_cycle_has_missing_rows,
            open_door_count,
            last_door_close_ts,
            low_p_trip_count,
            is_low_p_switch_tripped,
            low_p_trip_episode_anomalous,
            carried_decisive_status as seed_decisive_status
        from {{ checkpoint_table }}
        where checkpoint_day < date('{{ min_ts }}')
        qualify row_number() over (partition by location_id order by checkpoint_day desc) = 1
    {% else %}
        select
            location_id,
            ingested_at,
            event_timestamp,
            active_fruit_type_code,
            t_room_c,
            downtime_phase_name,
            defrost_cycle_id,
            defrost_phase_code,
            defrost_termination_reason_code,
            defrost_cycle_has_missing_rows,
            open_door_count,
            last_door_close_ts,
            low_p_trip_count,
            is_low_p_switch_tripped,
            low_p_trip_episode_anomalous,
            cast(null as string) as seed_decisive_status
        from source_slice
        where false
    {% endif %}
),

telemetry_unioned as (
    select
        location_id,
        ingested_at,
        event_timestamp,
        active_fruit_type_code,
        t_room_c,
        downtime_phase_name,
        defrost_cycle_id,
        defrost_phase_code,
        defrost_termination_reason_code,
        defrost_cycle_has_missing_rows,
        open_door_count,
        last_door_close_ts,
        low_p_trip_count,
        is_low_p_switch_tripped,
        low_p_trip_episode_anomalous,
        seed_decisive_status
    from seed_rows

    union all

    select
        location_id,
        ingested_at,
        event_timestamp,
        active_fruit_type_code,
        t_room_c,
        downtime_phase_name,
        defrost_cycle_id,
        defrost_phase_code,
        defrost_termination_reason_code,
        defrost_cycle_has_missing_rows,
        open_door_count,
        last_door_close_ts,
        low_p_trip_count,
        is_low_p_switch_tripped,
        low_p_trip_episode_anomalous,
        null as seed_decisive_status
    from source_slice
),

crop_setpoints as (
    select
        fruit_type_code,
        refrigeration_target_t_setpoint_c
    from {{ ref('dim_crop_variety') }}
),

equipment_design_params as (
    select
        location_id,
        compressor_deadband_differential_c,
        temperature_hysteresis_enter_margin_c as temp_enter_margin_c,
        temperature_hysteresis_exit_margin_c as temp_exit_margin_c
    from {{ ref('dim_equipment_design') }}
),

joined as (
    select
        t.*,
        cs.refrigeration_target_t_setpoint_c as sp,
        eq.compressor_deadband_differential_c as deadband,
        eq.temp_enter_margin_c as enter_margin,
        eq.temp_exit_margin_c as exit_margin,
        t.downtime_phase_name in ('shutdown', 'ramp_up') as is_downtime
    from telemetry_unioned as t
    left join crop_setpoints as cs on t.active_fruit_type_code = cs.fruit_type_code
    left join equipment_design_params as eq on t.location_id = eq.location_id
),

decisive as (
    select
        *,
        case
            when is_downtime or t_room_c is null then null -- handled as 'unknown'
            when
                t_room_c > sp + deadband + enter_margin
                or t_room_c < sp - enter_margin then 'out_of_spec'
            when t_room_c between sp + exit_margin and sp + deadband - exit_margin then 'in_spec'
            -- else null: dead zone, keep previous
        end as decisive_status
    from joined
)

select
    location_id,
    ingested_at,
    event_timestamp,
    t_room_c,
    active_fruit_type_code,
    downtime_phase_name,
    defrost_cycle_id,
    defrost_phase_code,
    defrost_termination_reason_code,
    defrost_cycle_has_missing_rows,
    open_door_count,
    last_door_close_ts,
    low_p_trip_count,
    is_low_p_switch_tripped,
    low_p_trip_episode_anomalous,
    sp as refrigeration_target_t_setpoint_c,
    case
        when is_downtime then 'downtime'
        when t_room_c is null then 'unknown'
        else last_value(coalesce(decisive_status, seed_decisive_status) ignore nulls) over w
    end as temperature_compliance_status,
    last_value(coalesce(decisive_status, seed_decisive_status) ignore nulls) over w as carried_decisive_status
from decisive
{% if is_incremental() %}
    qualify event_timestamp >= timestamp('{{ min_ts }}')
{% endif %}
window
    w as (
        partition by location_id order by event_timestamp
        rows between unbounded preceding and current row
    )
