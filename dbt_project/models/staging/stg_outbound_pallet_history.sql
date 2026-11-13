{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'outbound_pallet_history') }}

),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,
        cast(batch_id as int64) as batch_id,
        cast(parent_order_id as int64) as parent_order_id,

        -- timestamps
        cast(event_timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at,

        -- entries
        cast(gross_start_kg as float64) as recorded_gross_mass_at_dispatch_kg,
        cast(gross_moved_kg as float64) as gross_fruit_mass_moved_kg,
        cast(box_start_count as int64) as box_count_at_dispatch,
        cast(box_moved_count as int64) as box_count_moved,
        cast(pulp_temp_c as float64) as recorded_pulp_temp_at_dispatch_c,
        cast(quality_grade as int64) as quality_grade,
        timestamp_millis(cast(supervisor_intervention_ms as int64)) as intervention_timestamp

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, batch_id, parent_order_id, event_timestamp
    order by ingested_at desc
) = 1
