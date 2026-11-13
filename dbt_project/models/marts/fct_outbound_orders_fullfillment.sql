{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
        'field': 'order_date_utc',
        'data_type': 'date',
        'granularity': 'day'
    },
    cluster_by=['location_id', 'fruit_type_code']
) }}

{% if is_incremental() %}
    {% set watermark = ingest_watermark(this, 'order_date_utc', hours=3, ts_type='date') %}
{% endif %}

with location_details as (
    select
        location_id,
        region,
        country_name,
        iana_timezone
    from {{ ref('dim_geographic_details') }}
),

crop_variety as (
    select
        fruit_type_code,
        fruit_name
    from {{ ref('dim_crop_variety') }}
),

new_rows as (
    select order_ts
    from {{ ref('int_order_fulfillment') }}
    {% if is_incremental() %}
        where
            ingested_at >= timestamp('{{ watermark }}')
    {% endif %}
),

bounds as (
    select
        timestamp(date(min(order_ts))) as min_ts,
        timestamp(date_add(date(max(order_ts)), interval 1 day)) as max_ts
    from new_rows
),

staged_orders as (
    select *
    from {{ ref('int_order_fulfillment') }}
    {% if is_incremental() %}
        where
            order_ts >= (select min_ts from bounds)
            and order_ts < (select max_ts from bounds)
    {% endif %}
),

filtered_source as (
    select
        ord.*,
        loc.region,
        loc.country_name,
        loc.iana_timezone,
        crop.fruit_name,
        date(ord.order_ts) as order_date_utc
    from staged_orders as ord
    left join location_details as loc
        on ord.location_id = loc.location_id
    left join crop_variety as crop
        on ord.fruit_type_code = crop.fruit_type_code
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'outbound_order_id']) }} as order_fulfillment_hk,
    location_id,
    outbound_order_id,
    fruit_type_code,
    fruit_name,

    -- Timestamps (UTC)
    order_ts,
    -- Localized Datetimes for Operations Reporting
    datetime(order_ts, iana_timezone) as order_local_dt,

    -- Metrics
    requested_net_fruit_mass_kg,
    total_net_mass_shipped_kg,
    fill_rate_mass,
    target_box_count,
    total_boxes_shipped,
    fill_rate_boxes,
    is_order_fully_filled,

    minimum_grade_requested,
    worst_grade_shipped,
    grade_shortfall_flag,

    distinct_batches_in_order,
    dispatch_type_code,
    assigned_driver_code,

    -- Metadata & Partitions
    region,
    country_name,
    iana_timezone,
    order_date_utc,
    ingested_at,
    current_timestamp() as dbt_updated_at
from filtered_source
