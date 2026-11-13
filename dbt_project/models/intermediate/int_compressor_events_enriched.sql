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
        switch_reason_code,
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
        switch_reason_code,
        state_changed_flag,
        ingested_at,
        -- Label cycles (1 cycle = 1 on + 1 off period)
        sum(case when is_active or not state_changed_flag then 1 else 0 end) over w as cycle_id,
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
        w as (partition by location_id order by event_timestamp)
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'cycle_id']) }} as compressor_cycle_hk,
    location_id,
    cycle_id,
    ingested_at,
    event_timestamp,
    next_event_timestamp,
    is_active,
    switch_reason_code,
    row_duration_msec,
    -- Mark every row in the cycle if the cycle has an anomaly
    logical_or(not state_changed_flag or coalesce(next_row_is_duplicate, false)) over w as cycle_has_anomaly,
    (
        cycle_id > 0
        and count(*) over w < 2
        and (next_event_timestamp is not null or not is_active)
    ) as orphan_cycle,
    min(event_timestamp) over w as compressor_cycle_start_ts
from cycle_blocks
window
    w as (partition by location_id, cycle_id)
