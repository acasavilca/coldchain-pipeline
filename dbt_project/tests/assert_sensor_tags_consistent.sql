-- Guards against silent row loss from tag renames or additions.
--
-- int_telemetry_sensor_gap_filled and int_batch_sensor_gap_filled inner-join
-- seed_sensor_polling_config on tag_name, and the pivot / forward-fill models take their
-- column lists from the telemetry_sensor_tags / batch_sensor_tags vars. A tag that is
-- missing from any of the three is dropped without an error. Any row returned here is a failure.

with observed as (
    select distinct 'stg_telemetry' as source_model, tag_name from {{ ref('stg_telemetry') }}
    union distinct
    select distinct 'stg_batch_sensors' as source_model, tag_name from {{ ref('stg_batch_sensors') }}
),

configured as (
    {% for tag in var('telemetry_sensor_tags') + var('batch_sensor_tags') %}
    select '{{ tag }}' as tag_name{% if not loop.last %} union all{% endif %}
    {% endfor %}
),

seed as (
    select distinct tag_name from {{ ref('seed_sensor_polling_config') }}
)

select
    o.source_model as checked_in,
    o.tag_name,
    'raw tag missing from seed_sensor_polling_config (rows silently dropped)' as problem
from observed as o
left join seed as s on o.tag_name = s.tag_name
where s.tag_name is null

union all

select
    o.source_model,
    o.tag_name,
    'raw tag missing from telemetry_sensor_tags / batch_sensor_tags vars (no column built)'
from observed as o
left join configured as c on o.tag_name = c.tag_name
where c.tag_name is null

union all

select
    'dbt_project.yml vars',
    c.tag_name,
    'configured tag missing from seed_sensor_polling_config'
from configured as c
left join seed as s on c.tag_name = s.tag_name
where s.tag_name is null
