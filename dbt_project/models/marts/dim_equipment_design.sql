{{ config(
    materialized='table',
) }}

with source as (
    select * from {{ ref('seed_equipment_design_constants') }}
)

select
    *,
    -- Audit metadata
    current_timestamp() as dbt_updated_at
from source
