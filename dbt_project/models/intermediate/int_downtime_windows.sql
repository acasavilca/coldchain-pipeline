{{ config(materialized='table') }}

with downtime_sequence_events as (
    select
        *,
        sum(case when scheduled_start_ts is not null then 1 else 0 end) over (
            partition by location_id order by event_timestamp
        ) as downtime_seq_id
    from {{ ref('stg_downtime_sequence_events') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'downtime_seq_id']) }} as downtime_seq_hk,
    location_id,
    downtime_seq_id,
    max(ingested_at) as ingested_at,
    min(scheduled_start_ts) as scheduled_start_ts,
    min(shutdown_ts) as shutdown_ts,
    min(ramp_up_start_ts) as ramp_up_start_ts,
    max(ramp_up_end_ts) as ramp_up_end_ts
from downtime_sequence_events
group by location_id, downtime_seq_id
