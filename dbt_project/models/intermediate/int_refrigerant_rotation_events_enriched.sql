{{ config(materialized='table') }}

with refrigerant_rotation_events as (
    select * from {{ ref('stg_refrigerant_rotation_events') }}
),

enriched as (
    select
        location_id,
        ingested_at,
        event_timestamp,
        refrigerant_code,
        lead(event_timestamp) over (partition by location_id order by event_timestamp) as next_event_timestamp,
        refrigerant_code = lag(refrigerant_code) over (partition by location_id order by event_timestamp) as rotation_is_anomalous
    from refrigerant_rotation_events
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'event_timestamp']) }} as refrigerant_rotation_hk,
    location_id,
    ingested_at,
    event_timestamp,
    next_event_timestamp,
    refrigerant_code,
    coalesce(rotation_is_anomalous, false) as rotation_is_anomalous
from enriched
