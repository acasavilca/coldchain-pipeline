{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "compressor_cycle_start_ts",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "cycle_id"]
) }}

{% if is_incremental() and execute %}
    {% set watermark = ingest_watermark(this, 'compressor_cycle_start_ts', hours=6) %}
    {% set raw_lower_bound_sql %}
        select cast(timestamp(date(min(ts))) as string) as raw_min_ts
        from (
            select phase_start_ts as ts
            from {{ ref('int_compressor_events_detailed') }}
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
       select cast(timestamp(date(min(compressor_cycle_start_ts))) as string) as min_ts
       from {{ ref('int_compressor_events_detailed') }}
       where
            cycle_id > 0
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
        compressor_minimum_active_time_sec,
        compressor_minimum_idle_time_sec
    from {{ ref('dim_equipment_design') }}
),

switch_reason_code_ref as (
    select
        compressor_switch_reason_code,
        compressor_switch_reason_name,
        compressor_switch_reason_description
    from {{ ref('seed_compressor_switch_reason_reference') }}
),

compressor_events_collapsed as (
    select
        location_id,
        cycle_id,
        any_value(compressor_cycle_start_ts) as compressor_cycle_start_ts,
        any_value(
            if(
                compressor_status = 'off',
                struct(phase_start_ts as start_ts, phase_end_ts as end_ts, switch_reason_code),
                null
            )
        ) as off_phase,
        any_value(
            if(
                compressor_status = 'stabilizing',
                struct(phase_start_ts as start_ts, phase_end_ts as end_ts, switch_reason_code),
                null
            )
        ) as stabilizing_phase,
        any_value(
            if(
                compressor_status = 'steady_state',
                struct(phase_start_ts as start_ts, phase_end_ts as end_ts),
                null
            )
        ) as steady_state_phase,
        logical_or(cycle_has_anomaly) as cycle_has_anomaly,
        logical_or(orphan_cycle) as orphan_cycle,
        max(ingested_at) as ingested_at
    from {{ ref('int_compressor_events_detailed') }}
    {% if is_incremental() %}
        where compressor_cycle_start_ts >= timestamp('{{ min_ts }}')
    {% endif %}
    group by 1, 2
    having logical_or(is_active)
),

telemetry_collapsed as (
    select
        location_id,
        compressor_cycle_id as cycle_id,
        array_agg(
            if(
                is_compressor_active,
                struct(t_room_c, p_suction_line_bar, t_coil_evap_c, active_refrigerant_code),
                null
            ) ignore nulls order by quantized_event_ts asc limit 1
        )[safe_offset(0)] as telemetry_at_on_start,
        array_agg(
            if(
                is_compressor_active,
                struct(t_room_c, p_suction_line_bar, t_coil_evap_c, active_refrigerant_code),
                null
            ) ignore nulls order by quantized_event_ts desc limit 1
        )[safe_offset(0)] as telemetry_at_on_end,
        max(if(compressor_status = 'stabilizing', compressor_power_w, null)) as peak_power_w_stabilizing,
        avg(if(compressor_status = 'steady_state', compressor_power_w, null)) as avg_power_w_steady_state,
        avg(if(compressor_status = 'steady_state', cop, null)) as avg_cop_steady_state,
        avg(if(compressor_status = 'steady_state', superheat_k, null)) as avg_superheat_k_steady_state,
        avg(if(compressor_status = 'steady_state', t_evap_c, null)) as avg_t_evap_c_steady_state,
        avg(if(compressor_status = 'steady_state', t_cond_c, null)) as avg_t_cond_c_steady_state,
        avg(if(compressor_status = 'steady_state', compressor_pressure_ratio, null)) as avg_pressure_ratio_steady_state,
        avg(if(compressor_status = 'steady_state', q_cooling_estimated_w, null)) as avg_q_cooling_w_steady_state,
        max(ingested_at) as ingested_at
    from {{ ref('int_telemetry_derived') }}
    where
        compressor_cycle_id > 0
        {% if is_incremental() %}
            and quantized_event_ts >= timestamp('{{ min_ts }}')
        {% endif %}
    group by 1, 2
),

seed_rows as (
    {% if is_incremental() %}
        select
            location_id,
            compressor_cycle_start_ts
        from {{ this }}
        where compressor_cycle_start_ts < timestamp('{{ min_ts }}')
        qualify row_number() over (partition by location_id order by compressor_cycle_start_ts desc) = 1
    {% else %}
        select
            location_id,
            compressor_cycle_start_ts
        from compressor_events_collapsed
        where false
    {% endif %}
),

unioned as (
    select
        location_id,
        cast(null as int64) as cycle_id,
        compressor_cycle_start_ts,
        cast(null as struct<start_ts timestamp, end_ts timestamp, switch_reason_code int64>) as off_phase,
        cast(null as struct<start_ts timestamp, end_ts timestamp, switch_reason_code int64>) as stabilizing_phase,
        cast(null as struct<start_ts timestamp, end_ts timestamp>) as steady_state_phase,
        cast(null as bool) as cycle_has_anomaly,
        cast(null as bool) as orphan_cycle,
        cast(null as timestamp) as ingested_at
    from seed_rows

    union all

    select
        location_id,
        cycle_id,
        compressor_cycle_start_ts,
        off_phase,
        stabilizing_phase,
        steady_state_phase,
        cycle_has_anomaly,
        orphan_cycle,
        ingested_at
    from compressor_events_collapsed
),

compressor_cycles_with_intervals as (
    select
        *,
        timestamp_diff(
            compressor_cycle_start_ts,
            lag(compressor_cycle_start_ts) over (partition by location_id order by compressor_cycle_start_ts),
            second
        ) as compressor_cycle_interval_sec
    from unioned
    {% if is_incremental() %}
        qualify compressor_cycle_start_ts >= timestamp('{{ min_ts }}')
    {% endif %}
)

select
    c.* except (ingested_at, off_phase, stabilizing_phase, steady_state_phase),
    -- Phase timestamps (UTC)
    c.stabilizing_phase.start_ts as on_start_ts,
    c.steady_state_phase.start_ts as steady_state_start_ts,
    c.off_phase.start_ts as off_start_ts,
    c.off_phase.end_ts as compressor_cycle_end_ts,
    -- Durations
    timestamp_diff(c.stabilizing_phase.end_ts, c.stabilizing_phase.start_ts, second) as stabilizing_duration_sec,
    timestamp_diff(c.steady_state_phase.end_ts, c.steady_state_phase.start_ts, second) as steady_state_duration_sec,
    timestamp_diff(c.off_phase.start_ts, c.stabilizing_phase.start_ts, second) as on_duration_sec,
    timestamp_diff(c.off_phase.end_ts, c.off_phase.start_ts, second) as off_duration_sec,
    timestamp_diff(c.off_phase.end_ts, c.stabilizing_phase.start_ts, second) as compressor_cycle_duration_sec,
    safe_divide(
        timestamp_diff(c.off_phase.start_ts, c.stabilizing_phase.start_ts, second),
        timestamp_diff(c.off_phase.end_ts, c.stabilizing_phase.start_ts, second)
    ) * 100.0 as per_cycle_duty_cycle_pct,
    -- Design limits
    (timestamp_diff(c.off_phase.start_ts, c.stabilizing_phase.start_ts, second) - eq.compressor_minimum_active_time_sec) as on_duration_vs_minimum_sec,
    (timestamp_diff(c.off_phase.end_ts, c.off_phase.start_ts, second) - eq.compressor_minimum_idle_time_sec) as off_duration_vs_minimum_sec,
    coalesce(
        timestamp_diff(c.off_phase.start_ts, c.stabilizing_phase.start_ts, second) < eq.compressor_minimum_active_time_sec,
        false
    ) as is_below_min_active_time,
    coalesce(
        timestamp_diff(c.off_phase.end_ts, c.off_phase.start_ts, second) < eq.compressor_minimum_idle_time_sec,
        false
    ) as is_below_min_idle_time,
    -- Switch reasons
    c.stabilizing_phase.switch_reason_code as switch_on_reason_code,
    ref_on.compressor_switch_reason_name as switch_on_reason_name,
    c.off_phase.switch_reason_code as switch_off_reason_code,
    ref_off.compressor_switch_reason_name as switch_off_reason_name,
    ref_off.compressor_switch_reason_description as switch_off_reason_description,
    coalesce(c.off_phase.switch_reason_code = 0, false) as is_thermostat_stop,
    -- Open tail: last cycle per location, not yet ended
    (c.off_phase.end_ts is null and not coalesce(c.orphan_cycle, false)) as is_cycle_open,
    -- Telemetry at the start and end of the on period
    t.telemetry_at_on_start.active_refrigerant_code as active_refrigerant_code_at_on_start,
    t.telemetry_at_on_end.active_refrigerant_code as active_refrigerant_code_at_on_end,
    t.telemetry_at_on_start.t_room_c as t_room_c_at_on_start,
    t.telemetry_at_on_end.t_room_c as t_room_c_at_on_end,
    (t.telemetry_at_on_start.t_room_c - t.telemetry_at_on_end.t_room_c) as room_temp_drop_c,
    t.telemetry_at_on_start.p_suction_line_bar as p_suction_line_bar_at_on_start,
    t.telemetry_at_on_end.p_suction_line_bar as p_suction_line_bar_at_on_end,
    t.telemetry_at_on_start.t_coil_evap_c as t_coil_evap_c_at_on_start,
    t.telemetry_at_on_end.t_coil_evap_c as t_coil_evap_c_at_on_end,
    -- Phase aggregates (peak and steady-state averages)
    t.* except (location_id, cycle_id, telemetry_at_on_start, telemetry_at_on_end, ingested_at),
    safe_divide(
        t.avg_power_w_steady_state * timestamp_diff(c.off_phase.start_ts, c.stabilizing_phase.start_ts, second),
        3600000.0
    ) as compressor_energy_kwh_est,
    nullif(
        greatest(
            coalesce(c.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(t.ingested_at, timestamp('1970-01-01 00:00:00'))
        ),
        timestamp('1970-01-01 00:00:00')
    ) as ingested_at
from compressor_cycles_with_intervals as c
left join telemetry_collapsed as t
    on
        c.location_id = t.location_id
        and c.cycle_id = t.cycle_id
left join equipment_design_constants as eq
    on c.location_id = eq.location_id
left join switch_reason_code_ref as ref_on
    on c.stabilizing_phase.switch_reason_code = ref_on.compressor_switch_reason_code
left join switch_reason_code_ref as ref_off
    on c.off_phase.switch_reason_code = ref_off.compressor_switch_reason_code
