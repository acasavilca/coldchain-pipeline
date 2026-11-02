{{ config(materialized='table') }}

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
            is_open != lag(is_open, 1) over (partition by location_id, door_code order by event_timestamp),
            true
        ) as state_changed_flag
    from door_events
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
        sum(case when state_changed_flag = true and is_open = true then 1 else 0 end) over (
            partition by location_id, door_code
            order by event_timestamp
        ) as opening_id,
        -- Calculate duration of each row
        lead(event_timestamp) over (partition by location_id, door_code order by event_timestamp) as next_event_timestamp,
        timestamp_diff(
            lead(event_timestamp) over (partition by location_id, door_code order by event_timestamp),
            event_timestamp,
            millisecond
        ) as row_duration_msec
    from detected_states
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
    -- Mark every row in the cycle if the cycle has an anomaly
    logical_or(state_changed_flag = false) over (
        partition by location_id, door_code, opening_id
    ) as opening_has_anomaly
from opening_blocks
