{{ config(materialized='view') }}

with source as (
    select * from {{ source('coldchain_raw', 'weather_data') }}
),

renamed as (
    select
        -- identifiers
        cast(location_id as int64) as location_id,

        -- timestamps
        cast(timestamp as timestamp) as event_timestamp,
        cast(loaded_at as timestamp) as ingested_at,

        -- data
        -- cast(latitude as float64) as latitude,
        -- cast(longitude as float64) as longitude,
        cast(temperature_2m_c as float64) as ambient_temp_c,
        cast(relative_humidity_2m_pct / 100.0 as float64) as ambient_rh_frac,
        cast(dewpoint_2m_c as float64) as ambient_dewpoint_temp_c,
        cast(shortwave_radiation_wm2 as float64) as shortwave_radiation_wm2,
        cast(wind_speed_10m_ms as float64) as wind_speed_ms,
        cast(surface_pressure_hpa * 100.0 as float64) as ambient_pressure_pa,
        cast(soil_temperature_54cm_c as float64) as ground_temperature_c

    from source
)

select *
from renamed
qualify row_number() over (
    partition by location_id, event_timestamp
    order by ingested_at desc
) = 1
