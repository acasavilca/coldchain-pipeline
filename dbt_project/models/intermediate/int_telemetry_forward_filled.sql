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
    {% set watermark = ingest_watermark(this, 'quantized_event_ts', hours=6) %}
    {% set checkpoint_table = ensure_checkpoint_table(this) %}

    {% set lower_bound_sql %}
        select cast(timestamp(date(min(quantized_event_ts))) as string) as min_ts
        from {{ ref('int_facility_events_joined') }}
        where ingested_at >= timestamp('{{ watermark }}')
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

with source_slice as (
    select *
    from {{ ref('int_facility_events_joined') }}
    {% if is_incremental() %}
        where quantized_event_ts >= timestamp('{{ min_ts }}')
    {% endif %}
),

seed_rows as (
    {% if is_incremental() %}
        select * except (checkpoint_day)
        from {{ checkpoint_table }}
        where checkpoint_day < date('{{ min_ts }}')
        qualify row_number() over (partition by location_id order by checkpoint_day desc) = 1
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
),

-- Forward fill split by day: the last non-null value (and fill status) of each day is carried to the
-- following days through a small day-level window, so no sort ever holds more than one day of one location.
day_last as (
    select
        location_id,
        timestamp_trunc(quantized_event_ts, day) as day_ts,
        {% for tag in sensor_tags %}
            array_agg({{ tag }} ignore nulls order by quantized_event_ts desc limit 1)[safe_offset(0)] as {{ tag }},
            array_agg({{ tag }}_fill_status ignore nulls order by quantized_event_ts desc limit 1)[safe_offset(0)] as {{ tag }}_fill_status{% if not loop.last %},{% endif %}
        {% endfor %}
    from unioned
    group by location_id, day_ts
),

day_carry as (
    select
        location_id,
        day_ts,
        {% for tag in sensor_tags %}
            last_value({{ tag }} ignore nulls) over w as {{ tag }},
            last_value({{ tag }}_fill_status ignore nulls) over w as {{ tag }}_fill_status{% if not loop.last %},{% endif %}
        {% endfor %}
    from day_last
    window
        w as (
            partition by location_id order by day_ts
            rows between unbounded preceding and 1 preceding
        )
)

select
    u.* except (
        {{ sensor_tags | join(', ') }},
        {{ fill_status_cols | join(', ') }}
    ),
    {% for tag in sensor_tags %}
        case
            when coalesce(last_value(u.{{ tag }}_fill_status ignore nulls) over w, c.{{ tag }}_fill_status) = 'unknown' then null
            else coalesce(last_value(u.{{ tag }} ignore nulls) over w, c.{{ tag }})
        end as {{ tag }},
        coalesce(last_value(u.{{ tag }}_fill_status ignore nulls) over w, c.{{ tag }}_fill_status) as {{ tag }}_fill_status,
        u.{{ tag }} is null as is_{{ tag }}_cadence_filled{% if not loop.last %},{% endif %}
    {% endfor %}
from unioned as u
inner join day_carry as c
    on
        u.location_id = c.location_id
        and timestamp_trunc(u.quantized_event_ts, day) = c.day_ts
{% if is_incremental() %}
    where u.quantized_event_ts >= timestamp('{{ min_ts }}')
{% endif %}
window
    w as (
        partition by u.location_id, timestamp_trunc(u.quantized_event_ts, day)
        order by u.quantized_event_ts
        rows between unbounded preceding and current row
    )
