{{ config(
    materialized='table',
) }}

with source as (
    select * from {{ ref('seed_location_details') }}
)

select
    *,
    current_timestamp() as dbt_updated_at
from source
