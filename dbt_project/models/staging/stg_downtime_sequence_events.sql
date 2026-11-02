{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'downtime_sequence_events') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,

        -- timestamps
        cast(event_timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at,
        timestamp_millis(scheduled_start_timestamp_ms) as scheduled_start_ts,
        timestamp_millis(shutdown_timestamp_ms) as shutdown_ts,
        timestamp_millis(ramp_up_start_timestamp_ms) as ramp_up_start_ts,
        timestamp_millis(ramp_up_end_timestamp_ms) as ramp_up_end_ts
    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, event_timestamp
    order by ingested_at desc
) = 1
