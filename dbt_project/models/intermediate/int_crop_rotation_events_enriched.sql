{{ config(materialized='table') }}

with crop_rotation_events as (
    select * from {{ ref('stg_crop_rotation_events') }}
),

enriched as (
    select
        location_id,
        fruit_type_code,
        ingested_at,
        event_timestamp,
        lead(event_timestamp) over (partition by location_id order by event_timestamp) as next_event_timestamp,
        fruit_type_code = lag(fruit_type_code) over (partition by location_id order by event_timestamp) as rotation_is_anomalous
    from crop_rotation_events
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'event_timestamp']) }} as crop_rotation_hk,
    location_id,
    ingested_at,
    event_timestamp,
    next_event_timestamp,
    fruit_type_code,
    coalesce(rotation_is_anomalous, false) as rotation_is_anomalous
from enriched
