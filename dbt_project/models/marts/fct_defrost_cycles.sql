{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "defrost_cycle_start_ts",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "defrost_cycle_id"]
) }}

{% if is_incremental() and execute %}
    {% set watermark = ingest_watermark(this, 'defrost_cycle_start_ts', hours=6) %}
    {% set raw_lower_bound_sql %}
        select cast(timestamp(date(min(ts))) as string) as raw_min_ts
        from (
            select phase_start_ts as ts
            from {{ ref('int_defrost_events_enriched') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select defrost_end_ts as ts
            from {{ ref('int_defrost_recovery_events') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select quantized_event_ts as ts
            from {{ ref('int_telemetry_derived') }}
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
       select cast(timestamp(date(min(defrost_cycle_start_ts))) as string) as min_ts
       from {{ ref('int_defrost_events_enriched') }}
       where
            defrost_cycle_id > 0
            and phase_start_ts >= timestamp('{{ raw_min_ts }}')
    {% endset %}

    {% set a = run_query(true_lower_bound_sql) %}
    {% set min_ts = a.columns[0].values()[0] %}
    {% if min_ts is none %}
        {% set min_ts = raw_min_ts %}
    {% endif %}
{% endif %}

with equipment_design_constants as (
    select
        location_id,
        defrost_interval_sec
    from {{ ref('dim_equipment_design') }}
),

defrost_termination_reason_code_ref as (
    select
        defrost_termination_reason_code,
        defrost_termination_reason_name,
        defrost_termination_reason_description
    from {{ ref('seed_defrost_termination_reason_reference') }}
),

defrost_events_collapsed as (
    select
        location_id,
        defrost_cycle_id,
        any_value(defrost_cycle_start_ts) as defrost_cycle_start_ts,
        any_value(if(defrost_phase_code = 1, phase_start_ts, null)) as defrost_start_ts,
        any_value(if(defrost_phase_code = 0, phase_start_ts, null)) as defrost_end_ts,
        any_value(if(defrost_phase_code = 3, phase_start_ts, null)) as heat_start_ts,
        any_value(if(defrost_phase_code = 3, phase_end_ts, null)) as heat_end_ts,
        any_value(if(defrost_phase_code = 1, timestamp_diff(phase_end_ts, phase_start_ts, second), null)) as wait_for_stop_duration_sec,
        any_value(if(defrost_phase_code = 2, timestamp_diff(phase_end_ts, phase_start_ts, second), null)) as settle_duration_sec,
        any_value(if(defrost_phase_code = 3, timestamp_diff(phase_end_ts, phase_start_ts, second), null)) as heat_duration_sec,
        any_value(if(defrost_phase_code = 4, timestamp_diff(phase_end_ts, phase_start_ts, second), null)) as drip_duration_sec,
        any_value(if(defrost_phase_code = 5, timestamp_diff(phase_end_ts, phase_start_ts, second), null)) as fan_delay_duration_sec,
        any_value(termination_reason_code) as termination_reason_code,
        logical_or(has_missing_rows) as has_missing_rows,
        max(ingested_at) as ingested_at
    from {{ ref('int_defrost_events_enriched') }}
    {% if is_incremental() %}
        where defrost_cycle_start_ts >= timestamp('{{ min_ts }}')
    {% endif %}
    group by 1, 2
    having logical_or(has_any_defrost_row)
),

recovery_events as (
    select
        location_id,
        defrost_cycle_id,
        recovery_ts,
        ingested_at,
        timestamp_diff(recovery_ts, defrost_end_ts, second) as recovery_duration_sec
    from {{ ref('int_defrost_recovery_events') }}
    {% if is_incremental() %}
        where defrost_end_ts >= timestamp('{{ min_ts }}')
    {% endif %}
),

telemetry_collapsed as (
    select
        location_id,
        defrost_cycle_id,
        array_agg(
            if(defrost_phase_code = 3, struct(t_coil_evap_c, t_coil_evap_c_fill_status), null)
            ignore nulls
            order by quantized_event_ts desc limit 1
        )[safe_offset(0)] as coil_telemetry_at_heat_end,
        array_agg(
            if(
                defrost_phase_code = 5,
                struct(t_coil_evap_c, t_coil_evap_c_fill_status, airflow_ratio, velocity_over_trip_ratio),
                null
            )
            ignore nulls
            order by quantized_event_ts desc limit 1
        )[safe_offset(0)] as coil_telemetry_at_defrost_end,
        array_agg(
            if(
                defrost_phase_code = 1,
                struct(
                    evap_air_velocity_m_s,
                    airflow_ratio,
                    velocity_over_trip_ratio,
                    superheat_k,
                    t_evap_c,
                    minutes_since_last_defrost,
                    minutes_since_last_lp_trip_reset
                ),
                null
            )
            ignore nulls
            order by quantized_event_ts asc limit 1
        )[safe_offset(0)] as telemetry_at_defrost_start,
        max(ingested_at) as ingested_at
    from {{ ref('int_telemetry_derived') }}
    where
        defrost_phase_code in (1, 3, 5)
        {% if is_incremental() %}
            and quantized_event_ts >= timestamp('{{ min_ts }}')
        {% endif %}
    group by 1, 2
),

seed_rows as (
    {% if is_incremental() %}
        select
            location_id,
            defrost_cycle_start_ts
        from {{ this }}
        where defrost_cycle_start_ts < timestamp('{{ min_ts }}')
        qualify row_number() over (partition by location_id order by defrost_cycle_start_ts desc) = 1
    {% else %}
        select
            location_id,
            defrost_cycle_start_ts
        from defrost_events_collapsed
        where false
    {% endif %}
),

unioned as (
    select
        location_id,
        cast(null as int64) as defrost_cycle_id,
        defrost_cycle_start_ts,
        cast(null as timestamp) as defrost_start_ts,
        cast(null as timestamp) as defrost_end_ts,
        cast(null as timestamp) as heat_start_ts,
        cast(null as timestamp) as heat_end_ts,
        cast(null as int64) as wait_for_stop_duration_sec,
        cast(null as int64) as settle_duration_sec,
        cast(null as int64) as heat_duration_sec,
        cast(null as int64) as drip_duration_sec,
        cast(null as int64) as fan_delay_duration_sec,
        cast(null as int64) as termination_reason_code,
        cast(null as bool) as has_missing_rows,
        cast(null as timestamp) as ingested_at
    from seed_rows

    union all

    select
        location_id,
        defrost_cycle_id,
        defrost_cycle_start_ts,
        defrost_start_ts,
        defrost_end_ts,
        heat_start_ts,
        heat_end_ts,
        wait_for_stop_duration_sec,
        settle_duration_sec,
        heat_duration_sec,
        drip_duration_sec,
        fan_delay_duration_sec,
        termination_reason_code,
        has_missing_rows,
        ingested_at
    from defrost_events_collapsed
),

defrost_cycles_with_intervals as (
    select
        *,
        timestamp_diff(
            defrost_cycle_start_ts,
            lag(defrost_cycle_start_ts) over (partition by location_id order by defrost_cycle_start_ts),
            second
        ) as actual_interval_sec
    from unioned
    {% if is_incremental() %}
        qualify defrost_cycle_start_ts >= timestamp('{{ min_ts }}')
    {% endif %}
)

select
    d.* except (ingested_at),
    r.* except (location_id, defrost_cycle_id, ingested_at),
    ref.defrost_termination_reason_name,
    ref.defrost_termination_reason_description,
    t.coil_telemetry_at_heat_end.t_coil_evap_c as t_coil_evap_c_at_heat_end,
    t.coil_telemetry_at_heat_end.t_coil_evap_c_fill_status as t_coil_evap_c_at_heat_end_fill_status,
    t.coil_telemetry_at_defrost_end.t_coil_evap_c as t_coil_evap_c_at_defrost_end,
    t.coil_telemetry_at_defrost_end.t_coil_evap_c_fill_status as t_coil_evap_c_at_defrost_end_fill_status,
    t.coil_telemetry_at_defrost_end.airflow_ratio as airflow_ratio_at_defrost_end,
    t.coil_telemetry_at_defrost_end.velocity_over_trip_ratio as velocity_over_trip_ratio_at_defrost_end,
    t.telemetry_at_defrost_start.evap_air_velocity_m_s as evap_air_velocity_m_s_at_defrost_start,
    t.telemetry_at_defrost_start.airflow_ratio as airflow_ratio_at_defrost_start,
    t.telemetry_at_defrost_start.velocity_over_trip_ratio as velocity_over_trip_ratio_at_defrost_start,
    t.telemetry_at_defrost_start.superheat_k as superheat_k_at_defrost_start,
    t.telemetry_at_defrost_start.t_evap_c as t_evap_c_at_defrost_start,
    t.telemetry_at_defrost_start.minutes_since_last_defrost as minutes_since_last_defrost_at_defrost_start,
    t.telemetry_at_defrost_start.minutes_since_last_lp_trip_reset as minutes_since_last_lp_trip_reset_at_defrost_start,
    timestamp_diff(d.defrost_end_ts, d.defrost_start_ts, second) as total_defrost_duration_sec,
    (d.actual_interval_sec - eq.defrost_interval_sec) as interval_deviation_sec,
    (r.defrost_cycle_id is not null) as caused_temperature_disruption,
    nullif(
        greatest(
            coalesce(d.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(r.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(t.ingested_at, timestamp('1970-01-01 00:00:00'))
        ),
        timestamp('1970-01-01 00:00:00')
    ) as ingested_at
from defrost_cycles_with_intervals as d
left join recovery_events as r
    on
        d.location_id = r.location_id
        and d.defrost_cycle_id = r.defrost_cycle_id
left join telemetry_collapsed as t
    on
        d.location_id = t.location_id
        and d.defrost_cycle_id = t.defrost_cycle_id
left join equipment_design_constants as eq
    on d.location_id = eq.location_id
left join defrost_termination_reason_code_ref as ref
    on d.termination_reason_code = ref.defrost_termination_reason_code
