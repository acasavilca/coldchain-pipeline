{{ config(materialized='table') }}
select
    *,
    current_timestamp() as dbt_updated_at
from {{ ref('seed_driver_reference') }}
