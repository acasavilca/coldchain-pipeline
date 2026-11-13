{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "checkpoint_day",
      "data_type": "date",
      "granularity": "day"
    },
    cluster_by=["location_id"]
) }}

{% if is_incremental() and execute %}
    {% set lower_bound_sql %}
        select cast(timestamp(date(min(quantized_event_ts))) as string) as min_ts
        from {{ ref('int_telemetry_forward_filled') }}
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
    {% if var('min_ts_override', none) %}
        {% set min_ts = var('min_ts_override') %}
    {% endif %}
{% endif %}

select
    *,
    date(quantized_event_ts) as checkpoint_day
from {{ ref('int_telemetry_forward_filled') }}
{% if is_incremental() %}
    where quantized_event_ts >= timestamp('{{ min_ts }}')
{% endif %}
qualify row_number() over (partition by location_id, date(quantized_event_ts) order by quantized_event_ts desc) = 1
