select
    bs.batch_id,
    bs.event_timestamp,
    ord.fruit_type_code as recorded_fruit_type,
    crop.fruit_type_code as active_fruit_at_reading_time
from {{ ref('stg_batch_sensors') }} as bs
inner join {{ ref('stg_inbound_pallet_history') }} as ph
    on
        bs.location_id = ph.location_id
        and bs.batch_id = ph.batch_id
inner join {{ ref('stg_inbound_orders') }} as ord
    on
        ph.location_id = ord.location_id
        and ph.parent_order_id = ord.order_id
left join {{ ref('int_crop_rotation_events_enriched') }} as crop
    on
        bs.location_id = crop.location_id
        and bs.event_timestamp >= crop.event_timestamp
        and bs.event_timestamp < coalesce(crop.next_event_timestamp, timestamp '2099-01-01')
where crop.fruit_type_code is distinct from ord.fruit_type_code
