{{ config(
    materialized='table',
) }}

with source as (
    select * from {{ ref('seed_equipment_design_constants') }}
)

select
    *,
    (evap_rated_airflow_m3_s / evap_rated_face_velocity_m_s) as evap_frontal_area_m2,
    (
        evap_airflow_proof_recovery_multiplier * evap_airflow_proof_trip_velocity_m_s
    ) as evap_airflow_recovery_threshold_m_s,
    -- Audit metadata
    current_timestamp() as dbt_updated_at
from source
