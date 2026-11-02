{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'outbound_orders') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,
        cast(order_id as int64) as order_id,
        cast(fruit_type_code as int64) as fruit_type_code,

        -- timestamps
        cast(event_timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at,

        -- entries
        cast(target_fruit_mass_kg as float64) as requested_net_fruit_mass_kg,
        cast(total_box_count as int64) as target_box_count,
        cast(total_pallet_count as int64) as target_pallet_count,
        cast(minimum_grade as int64) as minimum_grade_requested,
        cast(assigned_door_code as int64) as assigned_door_code,
        cast(assigned_driver_code as int64) as assigned_driver_code,
        cast(dispatch_type as int64) as dispatch_type_code

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, order_id, event_timestamp
    order by ingested_at desc
) = 1
