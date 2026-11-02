{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      'field': 'order_ts',
      'data_type': 'timestamp',
      'granularity': 'day'
    },
    cluster_by=['location_id', 'fruit_type_code']
) }}

with pallet_history_aggregate as (
    select
        location_id,
        outbound_order_id,
        max(ingested_at) as ingested_at,
        sum(dispatched_net_fruit_mass_est_kg) as total_net_mass_shipped_kg,
        sum(box_count_moved) as total_boxes_shipped,
        max(quality_grade_at_dispatch) as worst_grade_shipped, -- scale: 0 (best) to n (worst)
        count(distinct batch_id) as distinct_batches_in_order
    from {{ ref('int_pallet_history_enriched') }}
    group by 1, 2
),

new_rows as (
    select ord.event_timestamp
    from {{ ref('stg_outbound_orders') }} as ord
    left join pallet_history_aggregate as agg
        on
            ord.location_id = agg.location_id
            and ord.order_id = agg.outbound_order_id
    {% if is_incremental() %}
        where
            greatest(ord.ingested_at, coalesce(agg.ingested_at, ord.ingested_at)) >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
        {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
        {% endif %}
    {% endif %}
),

bounds as (
    select
        timestamp(date(min(event_timestamp))) as min_ts,
        timestamp(date_add(date(max(event_timestamp)), interval 1 day)) as max_ts
    from new_rows
),

outbound_orders as (
    select
        location_id,
        order_id,
        fruit_type_code,
        event_timestamp,
        ingested_at,
        requested_net_fruit_mass_kg,
        target_box_count,
        target_pallet_count,
        minimum_grade_requested, -- remember spoilage-type dispatches are null
        dispatch_type_code,
        assigned_driver_code
    from {{ ref('stg_outbound_orders') }}
    {% if is_incremental() %}
        where
            event_timestamp >= (select min_ts from bounds)
            and event_timestamp < (select max_ts from bounds)
    {% endif %}
)

select
    ord.* except (order_id, event_timestamp, ingested_at),
    ord.event_timestamp as order_ts,
    ord.order_id as outbound_order_id,
    agg.total_net_mass_shipped_kg,
    agg.total_boxes_shipped,
    agg.worst_grade_shipped,
    agg.distinct_batches_in_order,
    -- ord.location_id,
    -- ord.order_id as outbound_order_id,
    -- ord.fruit_type_code,
    -- ord.event_timestamp,
    -- ord.ingested_at,
    (agg.total_net_mass_shipped_kg / ord.requested_net_fruit_mass_kg) as fill_rate_mass,
    (agg.total_boxes_shipped / ord.target_box_count) as fill_rate_boxes,
    (agg.total_boxes_shipped / ord.target_box_count) >= 1.0 as is_order_fully_filled,
    (agg.worst_grade_shipped > ord.minimum_grade_requested) as grade_shortfall_flag,
    greatest(ord.ingested_at, coalesce(agg.ingested_at, ord.ingested_at)) as ingested_at
from outbound_orders as ord
left join pallet_history_aggregate as agg
    on
        ord.location_id = agg.location_id
        and ord.order_id = agg.outbound_order_id
