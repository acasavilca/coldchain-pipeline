{{ config(
    materialized='table',
    unique_key='operating_constants_hk'
) }}

with equipment_design_constants as (
    select * from {{ ref('dim_equipment_design') }}
),

compressor_physical_constants as (
    select * from {{ ref('seed_compressor_physical_constants') }}
),

compressor_rating_conditions as (
    select * from {{ ref('seed_compressor_rating_conditions') }}
),

rated_mass_flow_curve as (
    select
        compressor_model_code,
        refrigerant_code,
        refrigerant_name,
        array(
            select cast(val as float64)
            from unnest(json_extract_array(coefficients)) as val
        ) as coeffs_array,
        array(
            select cast(val as float64)
            from unnest(json_extract_array(bounds)) as val
        ) as bounds_array
    from {{ ref('seed_rated_mass_flow_curve') }}
    where curve_name = 'mass_flow_kg_s'
),

refrigerant_physical_constants as (
    select * from {{ ref('seed_refrigerant_physical_constants') }}
),

refrigerant_saturation_curves as (
    select
        refrigerant_code,
        refrigerant_name,
        antoine_evap_coeffs_array,
        antoine_cond_coeffs_array,
        rho_suction_coeffs_array,
        rho_suction_bounds_array
    from {{ ref('dim_refrigerant_saturation_curves') }}
),

crop_variety_reference as (
    select
        fruit_type_code,
        fruit_name,
        refrigeration_target_t_setpoint_c
    from {{ ref('dim_crop_variety') }}
),

join_operating_constants as (
    select
        eq_design.location_id,
        comp_phys.compressor_model_code,
        comp_rating.refrigerant_code,
        comp_rating.refrigerant_name,
        crop_variety.fruit_type_code,
        crop_variety.fruit_name,
        comp_phys.compressor_displacement_m3_rev,
        comp_phys.compressor_speed_hz,
        comp_rating.rated_superheat_k,
        comp_rating.lp_trip_cut_out_bar,
        comp_rating.lp_trip_cut_in_bar,
        refrig_phys_const.gamma_refrigerant,
        mass_flow_curve.coeffs_array as m_dot_coeffs,
        mass_flow_curve.bounds_array as m_dot_bounds,
        refrig_curves.antoine_evap_coeffs_array as antoine_evap_coeffs,
        refrig_curves.antoine_cond_coeffs_array as antoine_cond_coeffs,
        refrig_curves.rho_suction_coeffs_array as rho_suction_coeffs,
        refrig_curves.rho_suction_bounds_array as rho_suction_bounds,
        (crop_variety.refrigeration_target_t_setpoint_c - eq_design.evaporator_design_temperature_difference_k) as t_evap_rated_c,
        (eq_design.ambient_design_temperature_c + eq_design.condenser_design_temperature_difference_k) as t_cond_rated_c
    from compressor_rating_conditions as comp_rating
    inner join compressor_physical_constants as comp_phys
        on comp_rating.compressor_model_code = comp_phys.compressor_model_code
    inner join rated_mass_flow_curve as mass_flow_curve
        on
            comp_rating.compressor_model_code = mass_flow_curve.compressor_model_code
            and comp_rating.refrigerant_code = mass_flow_curve.refrigerant_code
    inner join refrigerant_saturation_curves as refrig_curves
        on comp_rating.refrigerant_code = refrig_curves.refrigerant_code
    inner join refrigerant_physical_constants as refrig_phys_const
        on comp_rating.refrigerant_code = refrig_phys_const.refrigerant_code
    cross join crop_variety_reference as crop_variety
    cross join equipment_design_constants as eq_design
),

raw_operating_constants as (
    select
        location_id,
        compressor_model_code,
        refrigerant_code,
        refrigerant_name,
        fruit_type_code,
        fruit_name,
        t_evap_rated_c,
        t_cond_rated_c,
        gamma_refrigerant,
        compressor_displacement_m3_rev,
        compressor_speed_hz,
        rated_superheat_k,
        lp_trip_cut_out_bar,
        lp_trip_cut_in_bar,
        {{ evaluate_poly(dims=2, x_cols=['t_evap_rated_c', 't_cond_rated_c'], coeffs_col='m_dot_coeffs', bounds_col='m_dot_bounds') }} as m_dot_refrig_kg_s_rated,
        {{ evaluate_poly(dims=2, x_cols=['t_evap_rated_c', 'rated_superheat_k'], coeffs_col='rho_suction_coeffs', bounds_col='rho_suction_bounds') }} as rho_suction_rated,
        {{ calculate_pressure_ratio('t_evap_rated_c', 't_cond_rated_c', 'antoine_evap_coeffs', 'antoine_cond_coeffs') }} as pressure_ratio_rated
    from join_operating_constants
),

operating_constants as (
    select
        location_id,
        compressor_model_code,
        refrigerant_code,
        refrigerant_name,
        fruit_type_code,
        fruit_name,
        t_evap_rated_c,
        t_cond_rated_c,
        m_dot_refrig_kg_s_rated,
        rho_suction_rated,
        pressure_ratio_rated,
        gamma_refrigerant,
        compressor_displacement_m3_rev,
        compressor_speed_hz,
        lp_trip_cut_out_bar,
        lp_trip_cut_in_bar,
        (m_dot_refrig_kg_s_rated / (compressor_displacement_m3_rev * compressor_speed_hz * rho_suction_rated)) as eta_vol_rated
    from raw_operating_constants
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'compressor_model_code', 'refrigerant_code', 'fruit_type_code']) }} AS operating_constants_hk,
    location_id,
    compressor_model_code,
    refrigerant_code,
    refrigerant_name,
    fruit_type_code,
    fruit_name,
    t_evap_rated_c,
    t_cond_rated_c,
    m_dot_refrig_kg_s_rated,
    rho_suction_rated,
    pressure_ratio_rated,
    eta_vol_rated,
    gamma_refrigerant,
    (1.0 / gamma_refrigerant) as inv_gamma,
    compressor_displacement_m3_rev,
    compressor_speed_hz,
    lp_trip_cut_out_bar,
    lp_trip_cut_in_bar,
    ((1.0 - eta_vol_rated) / (pow(pressure_ratio_rated, 1.0 / gamma_refrigerant) - 1.0)) as c_clearance,
    current_timestamp() as dbt_updated_at
from operating_constants
