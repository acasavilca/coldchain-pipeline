{{ config(
    materialized='table',
) }}

with parsed_curves as (
    select
        refrigerant_code,
        refrigerant_name,
        curve_name,
        fit_type,
        -- Parse the JSON arrays into native BigQuery FLOAT64 arrays once
        array(select cast(v as float64) from unnest(json_extract_array(coefficients)) as v) as coeffs_array,
        array(select cast(v as float64) from unnest(json_extract_array(bounds)) as v) as bounds_array
    from {{ ref('seed_refrigerant_saturation_curves') }}
)

select
    refrigerant_code,
    refrigerant_name,
    -- Antoine Evap
    any_value(case when curve_name = 'antoine_evap' then fit_type end) as antoine_evap_fit_type,
    any_value(case when curve_name = 'antoine_evap' then coeffs_array end) as antoine_evap_coeffs_array,
    -- Antoine Cond
    any_value(case when curve_name = 'antoine_cond' then fit_type end) as antoine_cond_fit_type,
    any_value(case when curve_name = 'antoine_cond' then coeffs_array end) as antoine_cond_coeffs_array,
    -- Rho Suction Density
    any_value(case when curve_name = 'rho_suction_density' then fit_type end) as rho_suction_fit_type,
    any_value(case when curve_name = 'rho_suction_density' then coeffs_array end) as rho_suction_coeffs_array,
    any_value(case when curve_name = 'rho_suction_density' then bounds_array end) as rho_suction_bounds_array,
    -- H1 Suction Enthalpy
    any_value(case when curve_name = 'h1_suction_enthalpy' then fit_type end) as h1_suction_fit_type,
    any_value(case when curve_name = 'h1_suction_enthalpy' then coeffs_array end) as h1_suction_coeffs_array,
    any_value(case when curve_name = 'h1_suction_enthalpy' then bounds_array end) as h1_suction_bounds_array,
    -- H4 Liquid Enthalpy
    any_value(case when curve_name = 'h4_liquid_enthalpy' then fit_type end) as h4_liquid_fit_type,
    any_value(case when curve_name = 'h4_liquid_enthalpy' then coeffs_array end) as h4_liquid_coeffs_array,
    any_value(case when curve_name = 'h4_liquid_enthalpy' then bounds_array end) as h4_liquid_bounds_array
from parsed_curves
group by 1, 2
