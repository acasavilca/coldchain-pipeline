{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
        'field': 'reading_date_utc',
        'data_type': 'date',
        'granularity': 'day'
    },
    cluster_by=['location_id', 'active_refrigerant_code', 'active_fruit_type_code']
) }}

with location_details as (
    select
        location_id,
        region,
        country_name,
        iana_timezone
    from {{ ref('dim_geographic_details') }}
),

operating_constants as (
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

new_rows as (
    select quantized_event_ts
    from {{ ref('int_facility_events_forward_filled') }}
    {% if is_incremental() %}
        where
            ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
        {% else %}
            (select timestamp_sub(max(t.ingested_at), interval 3 hour) from {{ this }} as t)
        {% endif %}
    {% endif %}
),

bounds as (
    select
        timestamp(date(min(quantized_event_ts))) as min_ts,
        timestamp(date_add(date(max(quantized_event_ts)), interval 1 day)) as max_ts
    from new_rows
),

staged_telemetry as (
    select *
    from {{ ref('int_facility_events_forward_filled') }}
    where
        compressor_status = 'steady_state'
        {% if is_incremental() %}
            and quantized_event_ts >= (select min_ts from bounds)
            and quantized_event_ts < (select max_ts from bounds)
        {% endif %}
),

telemetry_enriched as (
    select
        t.* except (quantized_event_ts),
        t.quantized_event_ts as reading_ts,
        oc.refrigerant_name as active_refrigerant_name,
        oc.fruit_name as active_fruit_name,
        loc.iana_timezone,
        loc.region,
        loc.country_name,
        oc.compressor_model_code,
        oc.inv_gamma,
        oc.c_clearance,
        oc.compressor_displacement_m3_rev,
        oc.compressor_speed_hz,
        rc.rho_suction_coeffs,
        rc.rho_suction_bounds,
        rc.h1_suction_coeffs,
        rc.h1_suction_bounds,
        rc.h4_liquid_coeffs,
        rc.h4_liquid_bounds,
        {{ invert_antoine_fit('t.p_suction_line_bar', 'rc.antoine_evap_coeffs') }} as t_evap_c,
        {{ invert_antoine_fit('t.p_liquid_line_bar', 'rc.antoine_cond_coeffs') }} as t_cond_c,
        {{ calculate_volumetric_efficiency('t.p_suction_line_bar', 't.p_liquid_line_bar', 'oc.c_clearance', 'oc.inv_gamma', prefix='compressor') }}
    from staged_telemetry as t
    left join operating_constants as oc
        on
            t.location_id = oc.location_id
            and t.active_refrigerant_code = oc.refrigerant_code
            and t.active_fruit_type_code = oc.fruit_type_code
    left join refrigerant_curves as rc
        on t.active_refrigerant_code = rc.refrigerant_code
    left join location_details as loc
        on t.location_id = loc.location_id
),

processed_1 as (
    select
        * except (rho_suction_coeffs, rho_suction_bounds, h1_suction_coeffs, h1_suction_bounds, h4_liquid_coeffs, h4_liquid_bounds),
        (t_suction_line_c - t_evap_c) as superheat_k,
        {{ evaluate_poly(dims=2, x_cols=['t_evap_c', '(t_suction_line_c - t_evap_c)'], coeffs_col='rho_suction_coeffs', bounds_col='rho_suction_bounds') }} as rho_suction,
        {{ evaluate_poly(dims=2, x_cols=['t_evap_c', '(t_suction_line_c - t_evap_c)'], coeffs_col='h1_suction_coeffs', bounds_col='h1_suction_bounds') }} as h1_suction_j_kg,
        {{ evaluate_poly(dims=1, x_cols=['t_liquid_line_c'], coeffs_col='h4_liquid_coeffs', bounds_col='h4_liquid_bounds') }} as h4_liquid_j_kg
    from telemetry_enriched
),

processed_2 as (
    select
        *,
        date(reading_ts) as reading_date_utc,
        (compressor_displacement_m3_rev * compressor_speed_hz * rho_suction * compressor_eta_volumetric) as refrig_mass_flow_est_kg_s
    from processed_1
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'reading_ts']) }} as compressor_performance_hk,
    location_id,
    active_refrigerant_code,
    active_refrigerant_name,
    active_fruit_type_code,
    active_fruit_name,
    compressor_model_code,

    -- Timestamps (UTC)
    reading_ts,
    -- Localized Datetimes for Operations Reporting
    datetime(reading_ts, iana_timezone) as reading_local_dt,

    -- Compressor metrics
    compressor_power_w,
    t_suction_line_c,
    t_liquid_line_c,
    t_evap_c,
    t_cond_c,
    superheat_k,
    p_suction_line_bar,
    p_liquid_line_bar,
    compressor_pressure_ratio,
    compressor_eta_volumetric,
    t_room_air_return_c,
    t_room_air_supply_c,
    (t_room_air_return_c - t_room_air_supply_c) as return_supply_air_temp_delta_c,
    rho_suction,
    h1_suction_j_kg / 1000.0 as h1_suction_kj_kg,
    h4_liquid_j_kg / 1000.0 as h4_liquid_kj_kg,
    refrig_mass_flow_est_kg_s,
    ((h1_suction_j_kg - h4_liquid_j_kg) * refrig_mass_flow_est_kg_s) as q_cooling_estimated_w,
    safe_divide((h1_suction_j_kg - h4_liquid_j_kg) * refrig_mass_flow_est_kg_s, compressor_power_w) as cop,

    -- Metadata & Partitions
    region,
    country_name,
    iana_timezone,
    reading_date_utc,
    ingested_at,
    current_timestamp() as dbt_updated_at
from processed_2
