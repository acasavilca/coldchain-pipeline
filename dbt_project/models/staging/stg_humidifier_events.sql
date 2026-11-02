{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'humidifier_events') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,

        -- timestamps
        cast(event_timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at,

        -- humidifier state
        cast(num_discs_on as int64) as num_discs_on

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, event_timestamp
    order by ingested_at desc
) = 1
