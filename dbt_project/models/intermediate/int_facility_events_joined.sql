{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      'field': 'quantized_event_ts',
      'data_type': 'timestamp',
      'granularity': 'day'
    },
    cluster_by=["location_id"]
) }}

{% if is_incremental() %}
    {% set watermark = ingest_watermark(this, 'quantized_event_ts', hours=6) %}
{% endif %}

with new_rows_telemetry as (
    select quantized_event_ts
    from {{ ref('int_telemetry_sparse_pivot') }}
    {% if is_incremental() %}
        where
            ingested_at >= timestamp('{{ watermark }}')
    {% endif %}
),

new_rows_states as (
    select quantized_event_ts
    from {{ ref('int_state_transitions_timeline') }}
    {% if is_incremental() %}
        where
            ingested_at >= timestamp('{{ watermark }}')
    {% endif %}
),

bounds_telemetry as (
    select
        timestamp(date(min(quantized_event_ts))) as min_ts,
        timestamp(date_add(date(max(quantized_event_ts)), interval 1 day)) as max_ts
    from new_rows_telemetry
),

bounds_states as (
    select
        timestamp(date(min(quantized_event_ts))) as min_ts,
        timestamp(date_add(date(max(quantized_event_ts)), interval 1 day)) as max_ts
    from new_rows_states
),

bounds_unioned as (
    select
        min_ts,
        max_ts
    from bounds_telemetry

    union all

    select
        min_ts,
        max_ts
    from bounds_states
),

bounds as (
    select
        min(min_ts) as min_ts,
        max(max_ts) as max_ts
    from bounds_unioned
),

telemetry_filtered as (
    select *
    from {{ ref('int_telemetry_sparse_pivot') }}
    {% if is_incremental() %}
        where
            quantized_event_ts >= (select min_ts from bounds)
            and quantized_event_ts < (select max_ts from bounds)
    {% endif %}
),

states_filtered as (
    select *
    from {{ ref('int_state_transitions_timeline') }}
    {% if is_incremental() %}
        where
            quantized_event_ts >= (select min_ts from bounds)
            and quantized_event_ts < (select max_ts from bounds)
    {% endif %}
)

select
    t.* except (ingested_at),
    s.* except (location_id, quantized_event_ts, ingested_at),
    greatest(t.ingested_at, coalesce(s.ingested_at, t.ingested_at)) as ingested_at
from telemetry_filtered as t
left join states_filtered as s
    on
        t.location_id = s.location_id
        and t.quantized_event_ts = s.quantized_event_ts
