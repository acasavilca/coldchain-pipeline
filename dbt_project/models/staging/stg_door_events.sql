{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'door_events') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,
        cast(door_code as int64) as door_code,

        -- timestamps
        cast(event_timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at,

        -- door_state
        cast(is_open as boolean) as is_open

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, door_code, event_timestamp
    order by ingested_at desc
) = 1
