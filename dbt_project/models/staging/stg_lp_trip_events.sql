{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'lp_trip_events') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,

        -- timestamps
        cast(event_timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at,

        -- low-pressure switch state
        cast(switch_on as boolean) as is_lp_switch_tripped

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, event_timestamp
    order by ingested_at desc
) = 1
