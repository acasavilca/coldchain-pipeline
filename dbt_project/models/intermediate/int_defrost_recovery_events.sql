{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "defrost_end_ts",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "defrost_cycle_id"]
) }}

{% if is_incremental() and execute %}
    {% set watermark = ingest_watermark(this, 'defrost_end_ts', hours=6) %}
    {% set raw_lower_bound_sql %}
        select cast(timestamp(date(min(ts))) as string) as raw_min_ts
        from (
            select event_timestamp as ts
            from {{ ref('int_room_temperature_compliance_readings') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select phase_start_ts as ts
            from {{ ref('int_defrost_events_enriched') }}
            where ingested_at >= timestamp('{{ watermark }}')
        )
    {% endset %}
    {% set b = run_query(raw_lower_bound_sql) %}
    {% set raw_min_ts = b.columns[0].values()[0] %}
    {% if raw_min_ts is none %}
        {% set raw_min_ts = '2100-01-01 00:00:00+00:00' %}
    {% endif %}
    {% if var('min_ts_override', none) %}
        {% set raw_min_ts = var('min_ts_override') %}
    {% endif %}
    {% set true_lower_bound_sql %}
        select cast(timestamp(date(min(ts))) as string) as min_ts
        from (
            select max(phase_start_ts) as ts
            from {{ ref('int_defrost_events_enriched') }}
            where defrost_phase_code = 0 and phase_start_ts < timestamp('{{ raw_min_ts }}')
            group by location_id
        )
    {% endset %}
    {% set a = run_query(true_lower_bound_sql) %}
    {% set min_ts = a.columns[0].values()[0] %}
    {% if min_ts is none %}{% set min_ts = raw_min_ts %}{% endif %}
{% endif %}

with temperature_compliance_readings_slice as (
    select
        location_id,
        event_timestamp,
        temperature_compliance_status,
        ingested_at
    from {{ ref('int_room_temperature_compliance_readings') }}
    {% if is_incremental() %}
        where event_timestamp >= timestamp('{{ min_ts }}')
    {% endif %}
),

defrost_end_events_slice as (
    select
        location_id,
        defrost_cycle_id,
        defrost_phase_code,
        phase_start_ts,
        ingested_at
    from {{ ref('int_defrost_events_enriched') }}
    where
        defrost_phase_code = 0
        and defrost_cycle_id > 0
        {% if is_incremental() %}
            and phase_start_ts >= timestamp('{{ min_ts }}')
        {% endif %}
),

unioned as (
    select
        location_id,
        null as defrost_cycle_id,
        event_timestamp as event_ts,
        null as defrost_end_ts,
        temperature_compliance_status,
        'reading' as row_type,
        ingested_at
    from temperature_compliance_readings_slice

    union all

    select
        location_id,
        defrost_cycle_id,
        phase_start_ts as event_ts,
        phase_start_ts as defrost_end_ts,
        null as temperature_compliance_status,
        'defrost_event' as row_type,
        ingested_at
    from defrost_end_events_slice
),

forward_filled as (
    select
        location_id,
        event_ts,
        temperature_compliance_status,
        row_type,
        ingested_at,
        last_value(defrost_cycle_id ignore nulls) over w as defrost_cycle_id,
        last_value(defrost_end_ts ignore nulls) over w as defrost_end_ts
    from unioned
    window
        w as (
            partition by location_id
            order by
                event_ts,
                case when row_type = 'defrost_event' then 0 else 1 end
            rows between unbounded preceding and current row
        )
),

defrost_end_disrupted as (
    select distinct
        location_id,
        defrost_end_ts
    from forward_filled
    where
        row_type = 'reading'
        and event_ts >= defrost_end_ts
    qualify
        row_number() over (partition by location_id, defrost_end_ts order by event_ts asc) = 1
        and temperature_compliance_status = 'out_of_spec'
)

select
    ff.location_id,
    ff.defrost_cycle_id,
    ff.defrost_end_ts,
    min(case when ff.temperature_compliance_status = 'in_spec' then ff.event_ts end) as recovery_ts,
    max(ff.ingested_at) as ingested_at
from forward_filled as ff
inner join defrost_end_disrupted as ded
    on
        ff.location_id = ded.location_id
        and ff.defrost_end_ts = ded.defrost_end_ts
where
    ff.row_type = 'reading'
    and ff.event_ts > ff.defrost_end_ts
group by 1, 2, 3
