-- every outbound pallet row must point at an order of the same location
select
    ph.location_id,
    ph.parent_order_id
from {{ source('coldchain_raw', 'outbound_pallet_history') }} as ph
left join {{ source('coldchain_raw', 'outbound_orders') }} as ord
    on
        ph.location_id = ord.location_id
        and ph.parent_order_id = ord.order_id
where ph.parent_order_id is not null and ord.order_id is null
