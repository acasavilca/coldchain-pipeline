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
        select cast(timestamp(date(min(event_timestamp))) as string) as min_ts
        from {{ ref('int_room_temperature_compliance_readings') }}
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
    date(event_timestamp) as checkpoint_day
from {{ ref('int_room_temperature_compliance_readings') }}
{% if is_incremental() %}
    where event_timestamp >= timestamp('{{ min_ts }}')
{% endif %}
qualify row_number() over (partition by location_id, date(event_timestamp) order by event_timestamp desc) = 1
