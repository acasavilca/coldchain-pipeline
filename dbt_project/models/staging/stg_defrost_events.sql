{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'defrost_events') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,

        -- timestamps
        cast(event_timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at,

        cast(defrost_phase as int64) as defrost_phase_code,
        cast(termination_reason as int64) as termination_reason_code

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, event_timestamp
    order by ingested_at desc
) = 1
