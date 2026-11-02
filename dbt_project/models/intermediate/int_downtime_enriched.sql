{{ config(materialized='table') }}

with sequence as (
    select
        location_id,
        downtime_seq_id,
        ingested_at,
        scheduled_start_ts,
        shutdown_ts,
        ramp_up_start_ts,
        ramp_up_end_ts
    from {{ ref('int_downtime_windows') }}
),

expanded as (
    select
        s.location_id,
        s.downtime_seq_id,
        s.ingested_at,
        d.phase_name,
        d.phase_start_ts
    from sequence as s
    cross join unnest([
        struct('pre_shutdown' as phase_name, scheduled_start_ts as phase_start_ts),
        struct('shutdown' as phase_name, shutdown_ts as phase_start_ts),
        struct('ramp_up' as phase_name, ramp_up_start_ts as phase_start_ts),
        struct('normal' as phase_name, ramp_up_end_ts as phase_start_ts)
    ]) as d
)

select
    *,
    lead(phase_start_ts) over (
        partition by location_id order by phase_start_ts
    ) as phase_end_ts
from expanded
