{{ config(
    materialized='table',
    partition_by={
      "field": "event_timestamp",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "low_p_trip_episode_id"]
) }}

with trip_events as (
    select *
    from {{ ref('stg_lp_trip_events') }}
),

detected_states as (
    select
        location_id,
        ingested_at,
        event_timestamp,
        is_lp_switch_tripped,
        -- Detect state change (Returns true for first row or when status flips)
        coalesce(
            is_lp_switch_tripped != lag(is_lp_switch_tripped, 1) over w,
            true
        ) as state_changed_flag,
        coalesce(
            is_lp_switch_tripped != lag(is_lp_switch_tripped, 1) over w,
            is_lp_switch_tripped
        ) as count_trip_flag
    from trip_events
    window
        w as (partition by location_id order by event_timestamp)
),

trip_blocks as (
    select
        location_id,
        ingested_at,
        event_timestamp,
        is_lp_switch_tripped,
        state_changed_flag,
        -- Label cycles (1 cycle = 1 on + 1 off period)
        sum(case when is_lp_switch_tripped or not state_changed_flag then 1 else 0 end) over w as low_p_trip_episode_id,
        -- Count trips
        sum(
            case
                when count_trip_flag and is_lp_switch_tripped then 1
                when count_trip_flag then -1
                else 0
            end
        ) over (partition by location_id order by event_timestamp) as lp_trip_count,
        -- Calculate duration of each row
        lead(event_timestamp) over w as next_event_timestamp,
        lead(not state_changed_flag) over (partition by location_id order by event_timestamp) as next_row_is_duplicate,
        timestamp_diff(
            lead(event_timestamp) over w,
            event_timestamp,
            millisecond
        ) as row_duration_msec
    from detected_states
    window
        w as (
            partition by location_id
            order by event_timestamp
        )
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'low_p_trip_episode_id']) }} as low_p_trip_episode_hk,
    location_id,
    low_p_trip_episode_id,
    ingested_at,
    event_timestamp,
    next_event_timestamp,
    is_lp_switch_tripped as is_low_p_switch_tripped,
    row_duration_msec,
    lp_trip_count as low_p_trip_count,
    -- Mark every row in the cycle if the cycle has an anomaly
    logical_or(not state_changed_flag or coalesce(next_row_is_duplicate, false)) over w as low_p_trip_episode_anomalous,
    (
        low_p_trip_episode_id > 0
        and count(*) over w < 2
        and (next_event_timestamp is not null or not is_lp_switch_tripped)
    ) as orphan_episode,
    min(event_timestamp) over w as low_p_trip_episode_start_ts
from trip_blocks
window
    w as (partition by location_id, low_p_trip_episode_id)
