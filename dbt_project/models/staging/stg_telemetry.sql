{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'telemetry') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,
        cast(tag_name as string) as tag_name,
        cast(unit as string) as unit,

        -- timestamps
        cast(event_timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at,

        -- unix epoch timestamp
        -- cast(timestamp_ms as int64) as unix_epoch_ms,

        -- telemetry data
        cast(value as float64) as sensor_reading

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, tag_name, event_timestamp
    order by ingested_at desc
) = 1
