{{ config(materialized='table') }}

with telemetry_enriched as (
    select
        * except (carried_decisive_status),
        temperature_compliance_status in ('in_spec', 'out_of_spec') as include_flag
    from {{ ref('int_room_temperature_compliance_readings') }}
),

detected_states as (
    select
        location_id,
        ingested_at,
        event_timestamp,
        temperature_compliance_status,
        include_flag,
        -- Detect state change (Returns true for first row or when status flips)
        last_value(case when include_flag then temperature_compliance_status end ignore nulls) over (
            partition by location_id order by event_timestamp
            rows between unbounded preceding and 1 preceding
        ) as prev_known_state
    from telemetry_enriched
),

flagged_states as (
    select
        *,
        -- A valid state change only occurs if current row is not null and it differs from the last known state
        include_flag and coalesce(temperature_compliance_status != prev_known_state, true) as excursion_state_changed
    from detected_states
),

excursion_blocks as (
    select
        *,
        sum(case when excursion_state_changed and temperature_compliance_status = 'out_of_spec' then 1 else 0 end) over (
            partition by location_id order by event_timestamp
        ) as excursion_id,
        lead(event_timestamp) over (partition by location_id order by event_timestamp) as next_event_timestamp,
        timestamp_diff(
            lead(event_timestamp) over (partition by location_id order by event_timestamp),
            event_timestamp,
            millisecond
        ) as row_duration_msec
    from flagged_states
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'excursion_id', 'temperature_compliance_status']) }} as temperature_excursion_hk,
    location_id,
    excursion_id,
    temperature_compliance_status,
    max(ingested_at) as ingested_at,
    -- Capture the exact boundaries of this specific status phase within the excursion block
    min(event_timestamp) as phase_start_ts,
    max(next_event_timestamp) as phase_end_ts,
    -- Total duration of the actual out-of-spec violation within this block
    sum(row_duration_msec) total_phase_duration_msec,
    -- Flag if this specific block contains any unknown data gaps
    logical_or(temperature_compliance_status = 'unknown') as excursion_block_has_gaps
from excursion_blocks
group by location_id, excursion_id, temperature_compliance_status
