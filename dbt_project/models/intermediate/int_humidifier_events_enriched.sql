{{ config(materialized='table') }}

with equipment_constants as (
    select
        location_id,
        humidifier_discs_per_cluster_count,
        humidifier_max_total_discs_ceiling
    from {{ ref('dim_equipment_design') }}
),

humidifier_events as (
    select
        hum.*,
        (hum.num_discs_on > 0) as is_humidifying,
        (hum.num_discs_on / eq.humidifier_max_total_discs_ceiling) * 100.0 as max_humid_capacity_utilized_pct,
        (hum.num_discs_on / eq.humidifier_discs_per_cluster_count) as num_clusters_on
    from {{ ref('stg_humidifier_events') }} as hum
    inner join equipment_constants as eq
        on hum.location_id = eq.location_id
),

detected_states as (
    select
        location_id,
        event_timestamp,
        num_discs_on,
        num_clusters_on,
        max_humid_capacity_utilized_pct,
        is_humidifying,
        ingested_at,
        -- Detect state change (Returns true for first row or when status flips)
        coalesce(
            is_humidifying != lag(is_humidifying, 1) over (partition by location_id order by event_timestamp),
            true
        ) as state_changed_flag
    from humidifier_events
),

cycle_blocks as (
    select
        location_id,
        event_timestamp,
        num_discs_on,
        num_clusters_on,
        max_humid_capacity_utilized_pct,
        is_humidifying,
        state_changed_flag,
        ingested_at,
        sum(
            case
                when is_humidifying and state_changed_flag then 1 -- off -> on: new cycle
                when not is_humidifying and not state_changed_flag then 1 -- off -> off: duplicate off, new id
                else 0
            end
        ) over w as cycle_id,
        lead(event_timestamp) over w as next_event_timestamp,
        (not is_humidifying and not state_changed_flag) as is_duplicate_off,
        lead(not is_humidifying and not state_changed_flag) over w as next_row_is_duplicate,
        timestamp_diff(
            lead(event_timestamp) over w,
            event_timestamp, millisecond
        ) as row_duration_msec
    from detected_states
    window
        w as (partition by location_id order by event_timestamp)
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'cycle_id']) }} as humidifier_cycle_hk,
    location_id,
    cycle_id,
    ingested_at,
    event_timestamp,
    next_event_timestamp,
    num_discs_on,
    num_clusters_on,
    max_humid_capacity_utilized_pct,
    is_humidifying,
    row_duration_msec,
    logical_or(is_duplicate_off or coalesce(next_row_is_duplicate, false)) over w as cycle_has_anomaly,
    (
        cycle_id > 0
        and (logical_and(is_humidifying) over w = logical_or(is_humidifying) over w)
        and (
            logical_and(next_event_timestamp is not null) over w
            or not logical_or(is_humidifying) over w
        )
    ) as orphan_cycle
from cycle_blocks
window
    w as (partition by location_id, cycle_id)
