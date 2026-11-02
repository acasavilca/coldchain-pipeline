{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'refrigerant_rotation_events') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,
        cast(refrigerant_code as int64) as refrigerant_code,

        -- timestamps
        cast(event_timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, refrigerant_code, event_timestamp
    order by ingested_at desc
) = 1
