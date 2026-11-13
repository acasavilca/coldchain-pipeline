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

{% if is_incremental() %}
    {% set watermark = ingest_watermark(this, 'reading_date_utc', hours=3, ts_type='date') %}
{% endif %}

with location_details as (
    select
        location_id,
        region,
        country_name,
        iana_timezone
    from {{ ref('dim_geographic_details') }}
),

new_rows as (
    select quantized_event_ts
    from {{ ref('int_telemetry_derived') }}
    {% if is_incremental() %}
        where
            ingested_at >= timestamp('{{ watermark }}')
    {% endif %}
),

bounds as (
    select
        timestamp(date(min(quantized_event_ts))) as min_ts,
        timestamp(date_add(date(max(quantized_event_ts)), interval 1 day)) as max_ts
    from new_rows
),

steady_state_telemetry as (
    select *
    from {{ ref('int_telemetry_derived') }}
    where
        compressor_status = 'steady_state'
        {% if is_incremental() %}
            and quantized_event_ts >= (select min_ts from bounds)
            and quantized_event_ts < (select max_ts from bounds)
        {% endif %}
)

select
    {{ dbt_utils.generate_surrogate_key(['t.location_id', 't.quantized_event_ts']) }} as compressor_performance_hk,
    t.location_id,
    t.active_refrigerant_code,
    t.active_refrigerant_name,
    t.active_fruit_type_code,
    t.active_fruit_name,
    t.compressor_model_code,
    t.defrost_phase_code,
    t.defrost_cycle_id,
    -- Timestamps (UTC)
    t.quantized_event_ts as reading_ts,
    -- Localized Datetimes for Operations Reporting
    -- quantized_event_local_dt as reading_local_dt,
    datetime(t.quantized_event_ts, loc.iana_timezone) as reading_local_dt,

    -- Compressor metrics
    t.compressor_power_w,
    t.t_suction_line_c,
    t.t_liquid_line_c,
    t.t_evap_c,
    t.t_cond_c,
    t.superheat_k,
    t.p_suction_line_bar,
    t.p_liquid_line_bar,
    t.compressor_pressure_ratio,
    t.compressor_eta_volumetric,
    t.t_room_air_return_c,
    t.t_room_air_supply_c,
    t.t_coil_evap_c,
    t.evap_air_velocity_m_s,
    t.airflow_ratio,
    t.velocity_over_trip_ratio,
    (t.t_room_air_return_c - t.t_room_air_supply_c) as return_supply_air_temp_delta_c,
    t.rho_suction,
    t.h1_suction_j_kg / 1000.0 as h1_suction_kj_kg,
    t.h4_liquid_j_kg / 1000.0 as h4_liquid_kj_kg,
    t.refrig_mass_flow_est_kg_s,
    t.q_cooling_estimated_w,
    t.cop,

    t.minutes_since_last_defrost,
    t.hours_since_last_defrost,
    t.minutes_since_last_lp_trip_reset,
    t.hours_since_last_lp_trip_reset,

    -- Metadata & Partitions
    loc.region,
    loc.country_name,
    loc.iana_timezone,
    date(t.quantized_event_ts) as reading_date_utc,
    t.ingested_at,
    current_timestamp() as dbt_updated_at
from steady_state_telemetry as t
left join location_details as loc
    on t.location_id = loc.location_id
