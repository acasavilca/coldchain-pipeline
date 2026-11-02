{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'defrost_events') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,

        -- timestamps (event_timestamp is the defrost END; start is carried separately)
        cast(event_timestamp as timestamp) as event_timestamp,
        timestamp_millis(cast(start_timestamp_ms as int64)) as defrost_start_ts,
        cast(loaded_at as timestamp) as ingested_at,

        -- termination: 1 = coil probe reached termination temperature,
        -- 2 = max heat time reached (see simulation/constants.py DF_TERMINATION_*)
        cast(termination_reason as int64) as termination_reason_code,
        case cast(termination_reason as int64)
            when 1 then 'probe'
            when 2 then 'timeout'
        end as termination_reason,

        -- ice removed during the cycle (grams)
        cast(ice_melted_g as int64) as ice_melted_g,
        cast(coil_ice_melted_g as int64) as coil_ice_melted_g,
        cast(pan_ice_melted_g as int64) as pan_ice_melted_g,
        cast(face_ice_shed_g as int64) as face_ice_shed_g

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, event_timestamp
    order by ingested_at desc
) = 1
