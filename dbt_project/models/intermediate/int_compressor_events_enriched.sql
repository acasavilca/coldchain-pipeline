{{ config(materialized='table') }}

with compressor_events as (
    select *
    from {{ ref('stg_compressor_events') }}
),

detected_states as (
    select
        location_id,
        event_timestamp,
        is_active,
        ingested_at,
        -- Detect state change (Returns true for first row or when status flips)
        coalesce(
            is_active != lag(is_active, 1) over (partition by location_id order by event_timestamp),
            true
        ) as state_changed_flag
    from compressor_events
),

cycle_blocks as (
    select
        location_id,
        event_timestamp,
        is_active,
        state_changed_flag,
        ingested_at,
        -- Label cycles (1 cycle = 1 on + 1 off period)
        sum(case when state_changed_flag = true and is_active = true then 1 else 0 end) over (
            partition by location_id
            order by event_timestamp
        ) as cycle_id,
        -- Calculate duration of each row
        lead(event_timestamp) over (partition by location_id order by event_timestamp) as next_event_timestamp,
        timestamp_diff(
            lead(event_timestamp) over (partition by location_id order by event_timestamp),
            event_timestamp,
            millisecond
        ) as row_duration_msec
    from detected_states
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'cycle_id']) }} as compressor_cycle_hk,
    location_id,
    cycle_id,
    ingested_at,
    event_timestamp,
    next_event_timestamp,
    is_active,
    row_duration_msec,
    -- Mark every row in the cycle if the cycle has an anomaly
    logical_or(state_changed_flag = false) over (
        partition by location_id, cycle_id
    ) as cycle_has_anomaly
from cycle_blocks
