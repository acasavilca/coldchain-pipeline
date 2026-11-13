{{ config(
    materialized='table',
    partition_by={
      "field": "event_timestamp",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "door_id", "opening_id"]
) }}

with door_events as (
    select *
    from {{ ref('stg_door_events') }}
),

detected_states as (
    select
        location_id,
        door_code,
        ingested_at,
        event_timestamp,
        is_open,
        -- Detect state change (Returns true for first row or when status flips)
        coalesce(
            is_open != lag(is_open, 1) over w,
            true
        ) as state_changed_flag,
        coalesce(
            is_open != lag(is_open, 1) over w,
            is_open
        ) as count_opening_flag
    from door_events
    window
        w as (partition by location_id, door_code order by event_timestamp)
),

opening_blocks as (
    select
        location_id,
        door_code,
        ingested_at,
        event_timestamp,
        is_open,
        state_changed_flag,
        -- Label cycles (1 cycle = 1 on + 1 off period)
        sum(case when is_open or not state_changed_flag then 1 else 0 end) over w as opening_id,
        -- Count openings
        sum(
            case
                when count_opening_flag and is_open then 1
                when count_opening_flag then -1
                else 0
            end
        ) over (partition by location_id order by event_timestamp) as open_door_count,
        -- Calculate duration of each row
        lead(event_timestamp) over w as next_event_timestamp,
        lead(not state_changed_flag) over w as next_row_is_duplicate,
        timestamp_diff(
            lead(event_timestamp) over w,
            event_timestamp,
            millisecond
        ) as row_duration_msec
    from detected_states
    window
        w as (
            partition by location_id, door_code
            order by event_timestamp
        )
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'door_code', 'opening_id']) }} as door_opening_hk,
    location_id,
    door_code as door_id,
    opening_id,
    ingested_at,
    event_timestamp,
    next_event_timestamp,
    is_open,
    row_duration_msec,
    open_door_count,
    -- Mark every row in the cycle if the cycle has an anomaly
    logical_or(not state_changed_flag or coalesce(next_row_is_duplicate, false)) over w as opening_has_anomaly,
    (
        opening_id > 0
        and count(*) over w < 2
        and (next_event_timestamp is not null or not is_open)
    ) as orphan_opening
from opening_blocks
window
    w as (partition by location_id, door_code, opening_id)
