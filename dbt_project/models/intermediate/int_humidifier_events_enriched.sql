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
        sum(case when state_changed_flag = true and is_humidifying = true then 1 else 0 end) over (
            partition by location_id order by event_timestamp
        ) as cycle_id,
        lead(event_timestamp) over (partition by location_id order by event_timestamp) as next_event_timestamp,
        timestamp_diff(
            lead(event_timestamp) over (partition by location_id order by event_timestamp),
            event_timestamp, millisecond
        ) as row_duration_msec
    from detected_states
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
    logical_or(state_changed_flag = false) over (partition by location_id, cycle_id) as cycle_has_anomaly
from cycle_blocks
