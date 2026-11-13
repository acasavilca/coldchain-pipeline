{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      'field': 'quantized_event_ts',
      'data_type': 'timestamp',
      'granularity': 'day'
    },
    cluster_by=['location_id', 'batch_id']
) }}

{% if is_incremental() %}
    {% set watermark = ingest_watermark(this, 'quantized_event_ts', hours=6) %}
{% endif %}

{% set sensor_tags = var('batch_sensor_tags') %}
{% set min_interval_msec = get_min_polling_interval_ms(sensor_tags) %}

with new_rows as (
    select event_timestamp
    from {{ ref('int_batch_sensor_gap_filled') }}
    {% if is_incremental() %}
        where
            ingested_at >= timestamp('{{ watermark }}')
    {% endif %}
),

bounds as (
    select
        timestamp(date(min(event_timestamp))) as min_ts,
        timestamp(date_add(date(max(event_timestamp)), interval 1 day)) as max_ts
    from new_rows
),

filled as (
    select *
    from {{ ref('int_batch_sensor_gap_filled') }}
    {% if is_incremental() %}
        where
            event_timestamp >= (select min_ts from bounds)
            and event_timestamp < (select max_ts from bounds)
    {% endif %}
)

select
    location_id,
    batch_id,
    pallet_slot,
    {{ quantize_timestamp_ms('event_timestamp', min_interval_msec) }} as quantized_event_ts,
    max(ingested_at) as ingested_at,
    {{ get_sensor_fill_columns(sensor_tags) }}
from filled
group by 1, 2, 3, 4
