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

{% set sensor_tags = var('telemetry_sensor_tags') %}

{% set telem_min_interval_msec = get_min_polling_interval_ms(sensor_tags) %}

{% set selected_sensor_tags = [
    't_room_c',
    'rh_room_frac',
    'co2_ppm',
    'o2_pct',
    't_room_air_return_c',
    't_room_air_supply_c'
] %}

{% set fill_status_cols = [] %}
{% for tag in selected_sensor_tags %}
    {% do fill_status_cols.append(tag ~ '_fill_status') %}
{% endfor %}

{% set cadence_fill_flags = [] %}
{% for tag in selected_sensor_tags %}
    {% do cadence_fill_flags.append('is_' ~ tag ~ '_cadence_filled') %}
{% endfor %}

with new_rows_batches as (
    select quantized_event_ts
    from {{ ref('int_batch_sensors_sparse_pivot') }}
    {% if is_incremental() %}
        where
            ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
        {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
        {% endif %}
    {% endif %}
),

new_rows_facility as (
    select quantized_event_ts
    from {{ ref('int_facility_events_forward_filled') }}
    {% if is_incremental() %}
        where
            ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
        {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
        {% endif %}
    {% endif %}
),

bounds_batches as (
    select
        timestamp(date(min(quantized_event_ts))) as min_ts,
        timestamp(date_add(date(max(quantized_event_ts)), interval 1 day)) as max_ts
    from new_rows_batches
),

bounds_facility as (
    select
        timestamp(date(min(quantized_event_ts))) as min_ts,
        timestamp(date_add(date(max(quantized_event_ts)), interval 1 day)) as max_ts
    from new_rows_facility
),

bounds_unioned as (
    select
        min_ts,
        max_ts
    from bounds_batches

    union all

    select
        min_ts,
        max_ts
    from bounds_facility
),

bounds as (
    select
        min(min_ts) as min_ts,
        max(max_ts) as max_ts
    from bounds_unioned
),

batch_sensors_sparse_pivot_filtered as (
    select
        * except (quantized_event_ts),
        {{ quantize_timestamp_ms('quantized_event_ts', telem_min_interval_msec) }} as quantized_event_ts
    from {{ ref('int_batch_sensors_sparse_pivot') }}
    {% if is_incremental() %}
        where
            quantized_event_ts >= (select min_ts from bounds)
            and quantized_event_ts < (select max_ts from bounds)
    {% endif %}
),

facility_events_forward_filled_filtered as (
    select
        location_id,
        ingested_at,
        quantized_event_ts,
        is_compressor_active,
        is_humidifier_active,
        {{ selected_sensor_tags | join(', ') }},
        {{ fill_status_cols | join(', ') }},
        {{ cadence_fill_flags | join(', ') }}
    from {{ ref('int_facility_events_forward_filled') }}
    {% if is_incremental() %}
        where
            quantized_event_ts >= (select min_ts from bounds)
            and quantized_event_ts < (select max_ts from bounds)
    {% endif %}
)

select
    b.* except (ingested_at),
    t.* except (location_id, quantized_event_ts, ingested_at),
    greatest(b.ingested_at, coalesce(t.ingested_at, b.ingested_at)) as ingested_at
from batch_sensors_sparse_pivot_filtered as b
left join facility_events_forward_filled_filtered as t
    on
        b.location_id = t.location_id
        and b.quantized_event_ts = t.quantized_event_ts
