{{ config(
    materialized='table',
) }}

with source as (
    select * from {{ ref('seed_crop_variety_reference') }}
),

renamed as (
    select
        -- Natural keys and attributes
        fruit_type_code,
        initcap(fruit_name) as fruit_name,
        biological_shelf_life_days,
        target_retention_limit_days,
        -- Operational thresholds
        door_lock_threshold_temp_c,
        refrigeration_target_t_setpoint_c,
        refrigeration_target_rh_setpoint_fraction,
        chilling_injury_threshold_c,
        freezing_injury_threshold_c,
        heat_risk_threshold_c,
        -- Other fruit-specific parameters
        box_packaging_tare_mass_kg,
        pallet_structural_base_tare_mass_kg,
        -- Audit metadata
        current_timestamp() as dbt_updated_at
    from source
)

select * from renamed
