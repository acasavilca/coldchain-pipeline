{{ config(materialized='table') }}

{% set stabilization_cutoff_sec = var('compressor_stabilization_sec') %}

with compressor_events as (
    select
        location_id,
        cycle_id,
        ingested_at,
        event_timestamp,
        next_event_timestamp,
        is_active,
        switch_reason_code,
        cycle_has_anomaly,
        orphan_cycle,
        row_duration_msec,
        compressor_cycle_start_ts,
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
        compressor_cycle_start_ts,
        is_active,
        switch_reason_code,
        cycle_has_anomaly,
        orphan_cycle,
        phase
    from compressor_events
    cross join unnest(
        case
            when not is_active then ['off']
            when (is_active and row_duration_msec is null) or (on_duration_msec > ({{ stabilization_cutoff_sec }} * 1000.0)) then ['stabilizing', 'steady_state']
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
    orphan_cycle,
    is_active,
    switch_reason_code,
    compressor_cycle_start_ts,
    case
        when phase = 'off' or phase = 'stabilizing' then event_timestamp
        else timestamp_add(event_timestamp, interval {{ stabilization_cutoff_sec }} second)
    end as phase_start_ts,
    case
        when
            phase = 'stabilizing'
            and (row_duration_msec is null or on_duration_msec > ({{ stabilization_cutoff_sec }} * 1000.0))
            then timestamp_add(event_timestamp, interval {{ stabilization_cutoff_sec }} second)
        else next_event_timestamp
    end as phase_end_ts
from expanded
