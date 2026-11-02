{{ config(materialized='table') }}

with enriched_events as (
    select *
    from {{ ref('int_compressor_events_enriched') }}
)

select
    location_id,
    cycle_id,
    -- Sum durations for each on or off period
    sum(case when is_active = true then row_duration_msec else 0 end) as on_duration_msec,
    sum(case when is_active = false then row_duration_msec else 0 end) as off_duration_msec,
    min(event_timestamp) as cycle_start_ts,
    max(event_timestamp) as cycle_end_ts,
    -- Mark anomalies for cycles with any row with analomaly flag
    logical_or(cycle_has_anomaly) as cycle_has_anomaly
from enriched_events
group by 1, 2
