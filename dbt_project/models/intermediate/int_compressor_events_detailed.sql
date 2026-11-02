{{ config(materialized='table') }}

with compressor_events as (
    select
        location_id,
        cycle_id,
        ingested_at,
        event_timestamp,
        next_event_timestamp,
        is_active,
        cycle_has_anomaly,
        row_duration_msec,
        sum(case when is_active then row_duration_msec else 0 end) over (
            partition by location_id, cycle_id
            order by event_timestamp
        ) as on_duration_msec
    from {{ ref('int_compressor_events_enriched') }}
),

expanded as (
    select
        location_id,
        cycle_id,
        ingested_at,
        event_timestamp,
        next_event_timestamp,
        on_duration_msec,
        row_duration_msec,
        is_active,
        cycle_has_anomaly,
        phase
    from compressor_events
    cross join unnest(
        case
            when not is_active then ['off']
            when on_duration_msec > 120000 then ['stabilizing', 'steady_state']
            else ['stabilizing']
        end
    ) as phase
)

select
    location_id,
    cycle_id,
    ingested_at,
    phase as compressor_status,
    cycle_has_anomaly,
    is_active,
    case
        when phase = 'off' or phase = 'stabilizing' then event_timestamp
        else timestamp_add(event_timestamp, interval 2 minute)
    end as phase_start_ts,
    case
        when phase = 'stabilizing' and on_duration_msec > 120000 then timestamp_add(event_timestamp, interval 2 minute)
        else next_event_timestamp
    end as phase_end_ts
from expanded
