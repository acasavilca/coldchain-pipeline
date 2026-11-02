{{ config(
    materialized='table',
    partition_by={
        'field': 'event_timestamp',
        'data_type': 'timestamp',
        'granularity': 'day'
    },
    cluster_by=['location_id', 'fruit_type_code', 'event_type']
) }}

with location_details as (
    select
        location_id,
        iana_timezone,
        region,
        country_name
    from {{ ref('dim_geographic_details') }}
),

crop_variety as (
    select
        fruit_type_code,
        fruit_name
    from {{ ref('dim_crop_variety') }}
),

driver_reference as (
    select
        driver_code,
        driver_name
    from {{ ref('dim_driver') }}
),

door_reference as (
    select
        door_code,
        door_name
    from {{ ref('dim_door') }}
),

dispatch_type_reference as (
    select
        dispatch_type_code,
        dispatch_type_name
    from {{ ref('dim_dispatch_type') }}
),

inbound_orders as (
    select
        location_id,
        order_id,
        fruit_type_code,
        event_timestamp,
        box_count,
        pallet_count,
        assigned_door_code,
        assigned_driver_code,
        null as dispatch_type_code,
        'inbound' as event_type
    from {{ ref('stg_inbound_orders') }}
),

outbound_orders as (
    select
        location_id,
        order_id,
        fruit_type_code,
        event_timestamp,
        target_box_count as box_count,
        target_pallet_count as pallet_count,
        assigned_door_code, -- always happen through extenal door, add later to python dataframe
        assigned_driver_code,
        dispatch_type_code,
        'outbound' as event_type
    from {{ ref('stg_outbound_orders') }}
),

unioned as (
    select * from inbound_orders
    union all
    select * from outbound_orders
)

select
    {{ dbt_utils.generate_surrogate_key(['u.location_id', 'u.order_id', 'u.event_type']) }} as order_event_hk,
    u.location_id,
    loc.region,
    loc.country_name,
    u.order_id,
    u.fruit_type_code,
    crop.fruit_name,
    u.event_type,
    u.event_timestamp,
    datetime(u.event_timestamp, loc.iana_timezone) as event_local_dt,
    u.box_count,
    u.pallet_count,
    u.assigned_door_code,
    door.door_name as assigned_door_name,
    u.assigned_driver_code,
    drv.driver_name as assigned_driver_name,
    u.dispatch_type_code,
    bt.dispatch_type_name,
    timestamp_diff(
        u.event_timestamp,
        lag(u.event_timestamp) over (
            partition by u.location_id, u.fruit_type_code, u.event_type
            order by u.event_timestamp
        ),
        millisecond
    ) / 60000.0 as inter_event_period_min,
    current_timestamp() as dbt_updated_at
from unioned as u
left join location_details as loc on u.location_id = loc.location_id
left join crop_variety as crop on u.fruit_type_code = crop.fruit_type_code
left join driver_reference as drv on u.assigned_driver_code = drv.driver_code
left join door_reference as door on u.assigned_door_code = door.door_code
left join dispatch_type_reference as bt on u.dispatch_type_code = bt.dispatch_type_code
