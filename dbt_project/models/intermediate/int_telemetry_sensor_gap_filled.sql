{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "event_timestamp",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "tag_name"]
) }}

{% if is_incremental() and execute %}
    {% set bounds_sql %}
        select
          cast(timestamp(date(min(cast(event_timestamp as timestamp)))) as string) as min_ts,
          cast(timestamp(date_add(date(max(cast(event_timestamp as timestamp)) + interval 1 minute), interval 1 day)) as string) as max_ts
        from {{ source('coldchain_raw', 'telemetry') }}
        where cast(loaded_at as timestamp) >= {% if var('cutoff_override', none) %}
        timestamp('{{ var("cutoff_override") }}')
      {% else %}
          (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
      {% endif %}
    {% endset %}
    {% set b = run_query(bounds_sql) %}
    {% set min_ts = b.columns[0].values()[0] %}
    {% set max_ts = b.columns[1].values()[0] %}
    {% if min_ts is none %}
        {% set min_ts = '2100-01-01 00:00:00+00:00' %}
        {% set max_ts = '2100-01-02 00:00:00+00:00' %}
    {% endif %}
{% endif %}

with sensor_polling_config as (
    select
        tag_name,
        expected_polling_interval_sec
    from {{ ref('seed_sensor_polling_config') }}
),

source_slice as (
    select *
    from {{ ref('stg_telemetry') }}
    {% if is_incremental() %}
        where
            event_timestamp >= timestamp_sub(timestamp('{{ min_ts }}'), interval 1 minute)
            and event_timestamp < timestamp('{{ max_ts }}')
    {% endif %}
),

gap_flagged as (
    select
        t.*,
        p.expected_polling_interval_sec,
        last_value(
            case when t.sensor_reading is not null then t.event_timestamp end ignore nulls
        ) over w as last_real_reading_ts,
        last_value(
            t.sensor_reading ignore nulls
        ) over w as imputed_reading
    from source_slice as t
    inner join sensor_polling_config as p
        on t.tag_name = p.tag_name
    window
        w as (
            partition by t.location_id, t.tag_name order by t.event_timestamp
            rows between unbounded preceding and current row
        )
),

gap_flagged_enriched as (
    select
        *,
        timestamp_diff(event_timestamp, last_real_reading_ts, millisecond) / 1000.0 as gap_since_last_reading_sec
    from gap_flagged
)

select
    * except (last_real_reading_ts, imputed_reading, gap_since_last_reading_sec),
    case
        when sensor_reading is not null then sensor_reading
        when gap_since_last_reading_sec <= expected_polling_interval_sec * 2.0 then imputed_reading
    end as value_filled,
    case
        when sensor_reading is not null then 'real'
        when gap_since_last_reading_sec <= expected_polling_interval_sec * 2.0 then 'forward_filled'
        else 'unknown'
    end as fill_status
from gap_flagged_enriched
{% if is_incremental() %}
    where event_timestamp >= timestamp('{{ min_ts }}')
{% endif %}
