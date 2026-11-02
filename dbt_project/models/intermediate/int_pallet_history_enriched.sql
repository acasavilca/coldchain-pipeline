{{ config(
    materialized='table',
    partition_by={
      "field": "arrival_ts",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "batch_id"]
) }}

with crop_variety_params as (
    select
        fruit_type_code,
        box_packaging_tare_mass_kg,
        pallet_structural_base_tare_mass_kg,
        biological_shelf_life_days * 86400000 as biological_shelf_life_ms,
        target_retention_limit_days * 86400000 as target_retention_limit_ms
    from {{ ref('dim_crop_variety') }}
),

inbound_orders as (
    select
        location_id,
        order_id,
        fruit_type_code
    from {{ ref('stg_inbound_orders') }}
),

inbound_pallet_history_enriched as (
    select
        in_pall.location_id,
        in_pall.batch_id,
        in_pall.parent_order_id,
        in_ord.fruit_type_code,
        in_pall.event_timestamp,
        in_pall.intervention_timestamp,
        in_pall.arrival_gross_fruit_mass_kg,
        in_pall.incoming_pulp_temperature_c,
        in_pall.arrival_box_count,
        in_pall.quality_grade,
        in_pall.ingested_at,
        {{ estimate_batch_net_mass('in_pall.arrival_gross_fruit_mass_kg', 'c.box_packaging_tare_mass_kg', 'in_pall.arrival_box_count') }} as arrival_net_fruit_mass_est_kg
    from {{ ref('stg_inbound_pallet_history') }} as in_pall
    left join inbound_orders as in_ord
        on
            in_pall.location_id = in_ord.location_id
            and in_pall.parent_order_id = in_ord.order_id
    inner join crop_variety_params as c
        on in_ord.fruit_type_code = c.fruit_type_code
),

outbound_pallet_history as (
    select
        location_id,
        batch_id,
        parent_order_id,
        event_timestamp,
        intervention_timestamp,
        gross_fruit_mass_moved_kg,
        box_count_at_dispatch,
        box_count_moved,
        recorded_gross_mass_at_dispatch_kg,
        recorded_pulp_temp_at_dispatch_c,
        quality_grade,
        ingested_at,
        (box_count_at_dispatch = box_count_moved) as is_pallet_fully_dispatched
    from {{ ref('stg_outbound_pallet_history') }}
),

pallet_history_joined as (
    select
        i.location_id,
        i.batch_id,
        i.parent_order_id as inbound_order_id,
        i.fruit_type_code,
        greatest(i.ingested_at, coalesce(o.ingested_at, i.ingested_at)) as ingested_at,
        i.event_timestamp as arrival_ts,
        i.intervention_timestamp as arrival_supervisor_intervention_ts,
        i.arrival_gross_fruit_mass_kg,
        i.arrival_box_count,
        i.arrival_net_fruit_mass_est_kg,
        i.incoming_pulp_temperature_c,
        i.quality_grade as quality_grade_at_arrival,
        o.parent_order_id as outbound_order_id,
        o.event_timestamp as dispatch_ts,
        o.intervention_timestamp as dispatch_supervisor_intervention_ts,
        o.gross_fruit_mass_moved_kg,
        o.recorded_gross_mass_at_dispatch_kg,
        o.box_count_at_dispatch,
        o.box_count_moved,
        o.recorded_pulp_temp_at_dispatch_c,
        o.quality_grade as quality_grade_at_dispatch,
        o.is_pallet_fully_dispatched,
        c.biological_shelf_life_ms,
        c.target_retention_limit_ms,
        timestamp_diff(o.event_timestamp, i.event_timestamp, millisecond) as pallet_dwell_time_ms,
        (o.quality_grade - i.quality_grade) as quality_grade_decrease,
        {{ estimate_batch_net_mass('o.gross_fruit_mass_moved_kg', 'c.box_packaging_tare_mass_kg', 'o.box_count_moved') }} as dispatched_net_fruit_mass_est_kg,
        {{ estimate_batch_net_mass('o.recorded_gross_mass_at_dispatch_kg', 'c.box_packaging_tare_mass_kg', 'o.box_count_at_dispatch') }} as net_fruit_mass_at_dispatch_est_kg,
        row_number() over (partition by i.location_id, i.batch_id order by o.event_timestamp) as dispatch_sequence_num,
        sum(o.box_count_moved) over (partition by i.location_id, i.batch_id order by o.event_timestamp) as cumulative_dispatched_box_count,
        count(*) over (partition by i.location_id, i.batch_id) as total_dispatch_count_for_batch
    from inbound_pallet_history_enriched as i
    left join outbound_pallet_history as o
        on
            i.location_id = o.location_id
            and i.batch_id = o.batch_id
    inner join crop_variety_params as c
        on i.fruit_type_code = c.fruit_type_code
),

enriched as (
    select
        * except (biological_shelf_life_ms, target_retention_limit_ms),
        (cumulative_dispatched_box_count = arrival_box_count) as is_batch_fully_closed_out_now,
        sum(dispatched_net_fruit_mass_est_kg) over (partition by location_id, batch_id) as cumulative_dispatched_net_mass_est_kg,
        greatest(pallet_dwell_time_ms - biological_shelf_life_ms, 0.0) / 3600000 as hours_past_shelf_life,
        greatest(pallet_dwell_time_ms - target_retention_limit_ms, 0.0) / 3600000 as hours_past_target_retention_limit
    from pallet_history_joined
)

select
    *,
    arrival_net_fruit_mass_est_kg - cumulative_dispatched_net_mass_est_kg as batch_total_net_mass_loss_est_kg
from enriched
