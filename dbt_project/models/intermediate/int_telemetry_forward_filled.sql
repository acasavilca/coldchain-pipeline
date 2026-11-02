{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "quantized_event_ts",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id"]
) }}

{% set sensor_tags = var('telemetry_sensor_tags') %}
{% set state_columns = var('state_columns') %}

{% set fill_status_cols = [] %}
{% for tag in sensor_tags %}
    {% do fill_status_cols.append(tag ~ '_fill_status') %}
{% endfor %}

{% if is_incremental() and execute %}
    {% set lower_bound_sql %}
        select cast(timestamp(date(min(quantized_event_ts))) as string) as min_ts
        from {{ ref('int_facility_events_joined') }}
        where ingested_at >= {% if var('cutoff_override', none) %}
            timestamp('{{ var("cutoff_override") }}')
        {% else %}
            (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
        {% endif %}
    {% endset %}
    {% set b = run_query(lower_bound_sql) %}
    {% set min_ts = b.columns[0].values()[0] %}
    {% if min_ts is none %}
        {% set min_ts = '2100-01-01 00:00:00+00:00' %}
    {% endif %}
{% endif %}

with source_slice as (
    select *
    from {{ ref('int_facility_events_joined') }}
    {% if is_incremental() %}
        where quantized_event_ts >= timestamp('{{ min_ts }}')
    {% endif %}
),

seed_rows as (
    {% if is_incremental() %}
        select * from {{ this }}
        where
            quantized_event_ts >= timestamp_sub(timestamp('{{ min_ts }}'), interval 1 day)
            and quantized_event_ts < timestamp('{{ min_ts }}')
        qualify row_number() over (partition by location_id order by quantized_event_ts desc) = 1
    {% else %}
        select * from source_slice where false
    {% endif %}
),

unioned as (
    select
        location_id,
        quantized_event_ts,
        {{ sensor_tags | join(', ') }},
        {{ fill_status_cols | join(', ') }},
        {{ state_columns | join(', ') }},
        ingested_at
    from seed_rows

    union all

    select
        location_id,
        quantized_event_ts,
        {{ sensor_tags | join(', ') }},
        {{ fill_status_cols | join(', ') }},
        {{ state_columns | join(', ') }},
        ingested_at
    from source_slice
)

select
    * except (
        {{ sensor_tags | join(', ') }},
        {{ fill_status_cols | join(', ') }}
    ),
    {{ fill_sensor_cadence_gaps_window(sensor_tags, 'unknown', window_name='w') }}
from unioned
{% if is_incremental() %}
    qualify quantized_event_ts >= timestamp('{{ min_ts }}')
{% endif %}
window
    w as (
        partition by location_id order by quantized_event_ts
        rows between unbounded preceding and current row
    )
