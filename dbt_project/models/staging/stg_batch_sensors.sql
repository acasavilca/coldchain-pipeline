{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'batch_sensors') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,
        cast(batch_id as int64) as batch_id,
        cast(pallet_slot as int64) as pallet_slot,
        cast(tag_name as string) as tag_name,
        cast(unit as string) as unit,

        -- timestamps
        cast(event_timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at,

        -- telemetry data
        cast(value as float64) as sensor_reading

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, batch_id, pallet_slot, tag_name, event_timestamp
    order by ingested_at desc
) = 1
