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

{% if is_incremental() and execute %}
    {% set watermark = ingest_watermark(this, 'quantized_event_ts', hours=6) %}

    {% set lower_bound_sql %}
        select cast(timestamp(date(min(quantized_event_ts))) as string) as min_ts
        from {{ ref('int_facility_events_forward_filled') }}
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

with operating_constants as (
    select
        location_id,
        refrigerant_code,
        refrigerant_name,
        fruit_type_code,
        fruit_name,
        compressor_model_code,
        inv_gamma,
        c_clearance,
        compressor_displacement_m3_rev,
        compressor_speed_hz
    from {{ ref('dim_operating_constants') }}
),

equipment_reference_constants as (
    select
        location_id,
        evap_rated_face_velocity_m_s,
        evap_airflow_proof_trip_velocity_m_s
    from {{ ref('dim_equipment_design') }}
),

refrigerant_curves as (
    select
        refrigerant_code,
        antoine_evap_coeffs_array as antoine_evap_coeffs,
        antoine_cond_coeffs_array as antoine_cond_coeffs,
        rho_suction_coeffs_array as rho_suction_coeffs,
        rho_suction_bounds_array as rho_suction_bounds,
        h1_suction_coeffs_array as h1_suction_coeffs,
        h1_suction_bounds_array as h1_suction_bounds,
        h4_liquid_coeffs_array as h4_liquid_coeffs,
        h4_liquid_bounds_array as h4_liquid_bounds
    from {{ ref('dim_refrigerant_saturation_curves') }}
),

source_slice as (
    select *
    from {{ ref('int_facility_events_forward_filled') }}
    {% if is_incremental() %}
        where quantized_event_ts >= timestamp('{{ min_ts }}')
    {% endif %}
),

telemetry_enriched as (
    select
        t.*,
        oc.refrigerant_name as active_refrigerant_name,
        oc.fruit_name as active_fruit_name,
        oc.compressor_model_code,
        oc.inv_gamma,
        oc.c_clearance,
        oc.compressor_displacement_m3_rev,
        oc.compressor_speed_hz,
        -- loc.iana_timezone,
        -- loc.region,
        -- loc.country_name,
        rc.rho_suction_coeffs,
        rc.rho_suction_bounds,
        rc.h1_suction_coeffs,
        rc.h1_suction_bounds,
        rc.h4_liquid_coeffs,
        rc.h4_liquid_bounds,
        (t.evap_air_velocity_m_s / ec.evap_rated_face_velocity_m_s) as airflow_ratio,
        (t.evap_air_velocity_m_s / ec.evap_airflow_proof_trip_velocity_m_s) as velocity_over_trip_ratio,
        if(
            rc.antoine_evap_coeffs is not null,
            {{ invert_antoine_fit('t.p_suction_line_bar', 'rc.antoine_evap_coeffs') }},
            null
        ) as t_evap_c,
        if(
            rc.antoine_cond_coeffs is not null,
            {{ invert_antoine_fit('t.p_liquid_line_bar', 'rc.antoine_cond_coeffs') }},
            null
        ) as t_cond_c,
        if(
            compressor_status = 'steady_state',
            {{ compressor_volumetric_metrics('t.p_suction_line_bar', 't.p_liquid_line_bar', 'oc.c_clearance', 'oc.inv_gamma') }},
            null
        ) as compressor_metrics
    from source_slice as t
    left join operating_constants as oc
        on
            t.location_id = oc.location_id
            and t.active_refrigerant_code = oc.refrigerant_code
            and t.active_fruit_type_code = oc.fruit_type_code
    left join equipment_reference_constants as ec
        on t.location_id = ec.location_id
    left join refrigerant_curves as rc
        on t.active_refrigerant_code = rc.refrigerant_code
),

processed as (
    select
        * except (compressor_metrics),
        (t_suction_line_c - t_evap_c) as superheat_k,
        compressor_metrics.eta_volumetric as compressor_eta_volumetric,
        compressor_metrics.pressure_ratio as compressor_pressure_ratio,
        -- Guarded: evaluate_poly raises error() on NULL coefficient arrays (no refrigerant / no curve match).
        if(
            rho_suction_coeffs is not null,
            {{ evaluate_poly(dims=2, x_cols=['t_evap_c', '(t_suction_line_c - t_evap_c)'], coeffs_col='rho_suction_coeffs', bounds_col='rho_suction_bounds') }},
            null
        ) as rho_suction,
        if(
            h1_suction_coeffs is not null,
            {{ evaluate_poly(dims=2, x_cols=['t_evap_c', '(t_suction_line_c - t_evap_c)'], coeffs_col='h1_suction_coeffs', bounds_col='h1_suction_bounds') }},
            null
        ) as h1_suction_j_kg,
        if(
            h4_liquid_coeffs is not null,
            {{ evaluate_poly(dims=1, x_cols=['t_liquid_line_c'], coeffs_col='h4_liquid_coeffs', bounds_col='h4_liquid_bounds') }},
            null
        ) as h4_liquid_j_kg
    from telemetry_enriched
)

select
    * except (
        inv_gamma,
        c_clearance,
        compressor_displacement_m3_rev,
        compressor_speed_hz,
        rho_suction_coeffs,
        rho_suction_bounds,
        h1_suction_coeffs,
        h1_suction_bounds,
        h4_liquid_coeffs,
        h4_liquid_bounds
    ),
    (compressor_displacement_m3_rev * compressor_speed_hz * rho_suction * compressor_eta_volumetric) as refrig_mass_flow_est_kg_s,
    (
        (h1_suction_j_kg - h4_liquid_j_kg)
        * (compressor_displacement_m3_rev * compressor_speed_hz * rho_suction * compressor_eta_volumetric)
    ) as q_cooling_estimated_w,
    safe_divide(
        (
            (h1_suction_j_kg - h4_liquid_j_kg)
            * (compressor_displacement_m3_rev * compressor_speed_hz * rho_suction * compressor_eta_volumetric)
        ),
        compressor_power_w
    ) as cop,
    timestamp_diff(quantized_event_ts, last_defrost_end_ts, second) / 60.0 as minutes_since_last_defrost,
    timestamp_diff(quantized_event_ts, last_defrost_end_ts, second) / 3600.0 as hours_since_last_defrost,
    timestamp_diff(quantized_event_ts, last_lp_trip_switch_off_ts, second) / 60.0 as minutes_since_last_lp_trip_reset,
    timestamp_diff(quantized_event_ts, last_lp_trip_switch_off_ts, second) / 3600.0 as hours_since_last_lp_trip_reset
from processed
