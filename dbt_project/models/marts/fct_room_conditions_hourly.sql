{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
        'field': 'reading_date_utc',
        'data_type': 'date',
        'granularity': 'day'
    },
    cluster_by=['location_id', 'active_fruit_type_code']
) }}


{% if is_incremental() and execute %}
    {% set watermark = ingest_watermark(this, 'reading_date_utc', hours=6, ts_type='date') %}
    {% set lower_bound_sql %}
        select cast(timestamp(date(min(ts))) as string) as min_ts
        from (
            select quantized_event_ts as ts
            from {{ ref('int_facility_events_forward_filled') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select phase_start_ts as ts
            from {{ ref('int_temperature_excursion_events_enriched') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select door_close_ts as ts
            from {{ ref('int_door_recovery_events') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select defrost_end_ts as ts
            from {{ ref('int_defrost_recovery_events') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select lp_trip_switch_off_ts as ts
            from {{ ref('int_lp_trip_recovery_events') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select event_timestamp as ts
            from {{ ref('stg_weather') }}
            where ingested_at >= timestamp('{{ watermark }}')
        )
    {% endset %}
    {% set b = run_query(lower_bound_sql) %}
    {% set min_ts = b.columns[0].values()[0] %}
    {% if min_ts is none %}
        {% set min_ts = '2100-01-01 00:00:00+00:00' %}
    {% endif %}
    {% if var('min_ts_override', none) %}
        {% set min_ts = var('min_ts_override') %}
    {% endif %}
{% endif %}

with crop_variety as (
    select
        fruit_type_code,
        fruit_name,
        refrigeration_target_t_setpoint_c,
        refrigeration_target_rh_setpoint_fraction
    from {{ ref('dim_crop_variety') }}
),

location_details as (
    select
        location_id,
        iana_timezone,
        region,
        country_name
    from {{ ref('dim_geographic_details') }}
),

telemetry_enriched as (
    select
        *,
        coalesce(defrost_phase_code between 1 and 5, false) as is_defrosting
    from {{ ref('int_facility_events_forward_filled') }}
    {% if is_incremental() %}
        where quantized_event_ts >= timestamp('{{ min_ts }}')
    {% endif %}
),

temp_excursion_by_hour as (
    select
        location_id,
        timestamp_trunc(phase_start_ts, hour) as excursion_hour_utc,
        sum(total_phase_duration_msec) / 1000.0 / count(distinct excursion_id) as avg_excursion_duration_sec,
        count(distinct excursion_id) as excursion_count,
        count(distinct if(primary_cause = 'defrost', excursion_id, null)) as excursion_count_defrost,
        count(distinct if(primary_cause = 'door', excursion_id, null)) as excursion_count_door,
        count(distinct if(primary_cause = 'low_p_trip', excursion_id, null)) as excursion_count_low_p_trip,
        count(distinct if(primary_cause = 'downtime', excursion_id, null)) as excursion_count_downtime,
        count(distinct if(primary_cause like 'indeterminate%', excursion_id, null)) as excursion_count_indeterminate,
        count(distinct if(primary_cause = 'other', excursion_id, null)) as excursion_count_other,
        max(ingested_at) as ingested_at
    from {{ ref('int_temperature_excursion_events_enriched') }}
    where
        temperature_compliance_status = 'out_of_spec'
        {% if is_incremental() %}
            and phase_start_ts >= timestamp('{{ min_ts }}')
        {% endif %}
    group by 1, 2
),

door_recovery_by_hour as (
    select
        location_id,
        timestamp_trunc(door_close_ts, hour) as recovery_hour_utc,
        avg(timestamp_diff(recovery_ts, door_close_ts, second)) as avg_setpoint_recovery_sec,
        count(*) as disruptive_door_close_count,
        countif(recovery_ts is not null) as door_close_with_recovery_count,
        max(ingested_at) as ingested_at
    from {{ ref('int_door_recovery_events') }}
    {% if is_incremental() %}
        where door_close_ts >= timestamp('{{ min_ts }}')
    {% endif %}
    group by 1, 2
),

defrost_recovery_by_hour as (
    select
        location_id,
        timestamp_trunc(defrost_end_ts, hour) as recovery_hour_utc,
        avg(timestamp_diff(recovery_ts, defrost_end_ts, second)) as avg_setpoint_recovery_sec,
        count(*) as disruptive_defrost_end_count,
        countif(recovery_ts is not null) as defrost_end_with_recovery_count,
        max(ingested_at) as ingested_at
    from {{ ref('int_defrost_recovery_events') }}
    {% if is_incremental() %}
        where defrost_end_ts >= timestamp('{{ min_ts }}')
    {% endif %}
    group by 1, 2
),

weather_by_hour as (
    select
        location_id,
        timestamp_trunc(event_timestamp, hour) as weather_hour_utc,
        avg(ambient_temp_c) as avg_ambient_temp_c,
        avg(ambient_rh_frac) as avg_ambient_rh_frac,
        avg(ambient_dewpoint_temp_c) as avg_ambient_dewpoint_temp_c,
        avg(shortwave_radiation_wm2) as avg_shortwave_radiation_wm2,
        avg(wind_speed_ms) as avg_wind_speed_ms,
        avg(ambient_pressure_pa) as avg_ambient_pressure_pa,
        avg(ground_temperature_c) as avg_ground_temperature_c,
        max(ingested_at) as ingested_at
    from {{ ref('stg_weather') }}
    {% if is_incremental() %}
        where event_timestamp >= timestamp('{{ min_ts }}')
    {% endif %}
    group by 1, 2
),

lp_trip_recovery_by_hour as (
    select
        location_id,
        timestamp_trunc(lp_trip_switch_off_ts, hour) as recovery_hour_utc,
        avg(timestamp_diff(recovery_ts, lp_trip_switch_off_ts, second)) as avg_setpoint_recovery_sec,
        count(*) as disruptive_lp_trip_reset_count,
        countif(recovery_ts is not null) as lp_trip_reset_with_recovery_count,
        max(ingested_at) as ingested_at
    from {{ ref('int_lp_trip_recovery_events') }}
    {% if is_incremental() %}
        where lp_trip_switch_off_ts >= timestamp('{{ min_ts }}')
    {% endif %}
    group by 1, 2
),

joined as (
    select
        te.location_id,
        loc.region,
        loc.country_name,
        loc.iana_timezone,
        te.active_fruit_type_code,
        timestamp_trunc(te.quantized_event_ts, hour) as reading_hour_utc,

        min(timestamp_trunc(datetime(te.quantized_event_ts, loc.iana_timezone), hour)) as reading_hour_local,

        avg(te.t_room_c) as avg_t_room_c,
        avg(te.rh_room_frac) as avg_rh_room_frac,
        stddev(te.rh_room_frac) as stddev_rh_room_frac,
        avg(te.co2_ppm) as avg_co2_ppm,
        avg(te.o2_pct) as avg_o2_pct,
        avg(te.t_room_air_return_c - te.t_room_air_supply_c) as avg_return_supply_air_temp_delta_c,
        avg(te.t_coil_evap_c) as avg_t_coil_evap_c,
        min(te.t_coil_evap_c) as min_t_coil_evap_c,
        max(te.t_coil_evap_c) as max_t_coil_evap_c,
        avg(te.evap_air_velocity_m_s) as avg_evap_air_velocity_m_s,
        min(te.evap_air_velocity_m_s) as min_evap_air_velocity_m_s,
        safe_divide(
            countif(te.temperature_compliance_status = 'in_spec'),
            countif(te.temperature_compliance_status != 'downtime')
        ) as frac_time_in_temp_spec,
        safe_divide(
            countif(te.temperature_compliance_status = 'in_spec' and not te.is_defrosting),
            countif(te.temperature_compliance_status != 'downtime' and not te.is_defrosting)
        ) as frac_time_in_temp_spec_excl_defrost,
        countif(te.temperature_compliance_status = 'unknown') / count(*) as frac_time_temp_unknown,
        countif(te.temperature_compliance_status = 'downtime') / count(*) as frac_time_temp_downtime,
        countif(te.is_defrosting) / count(*) as frac_time_defrost,
        count(distinct if(te.is_defrosting, te.defrost_cycle_id, null)) as defrost_cycles_per_hour,
        countif(te.is_low_p_switch_tripped) / count(*) as frac_time_low_p_switch_tripped,
        count(distinct if(te.is_low_p_switch_tripped, te.low_p_trip_episode_id, null)) as low_p_trips_per_hour,
        countif(te.is_internal_door_open) / count(*) as frac_time_internal_door_open,
        countif(te.is_external_door_open) / count(*) as frac_time_external_door_open,
        count(distinct if(te.is_internal_door_open, te.internal_opening_id, null)) as internal_door_openings_per_hour,
        count(distinct if(te.is_external_door_open, te.external_opening_id, null)) as external_door_openings_per_hour,
        countif(te.is_internal_door_open or te.is_external_door_open) / count(*) as frac_time_any_door_open,
        countif(te.temperature_compliance_status = 'out_of_spec') / count(*) as frac_time_temp_excursion,
        countif(
            te.temperature_compliance_status = 'out_of_spec'
            and te.temperature_excursion_primary_cause = 'defrost'
        ) / count(*) as frac_time_defrost_temp_excursion,
        countif(
            te.temperature_compliance_status = 'out_of_spec'
            and te.temperature_excursion_primary_cause = 'door'
        ) / count(*) as frac_time_door_temp_excursion,
        countif(
            te.temperature_compliance_status = 'out_of_spec'
            and te.temperature_excursion_primary_cause = 'low_p_trip'
        ) / count(*) as frac_time_low_p_trip_temp_excursion,
        countif(
            te.temperature_compliance_status = 'out_of_spec'
            and te.temperature_excursion_primary_cause = 'downtime'
        ) / count(*) as frac_time_downtime_temp_excursion,
        countif(
            te.temperature_compliance_status = 'out_of_spec'
            and te.temperature_excursion_primary_cause like 'indeterminate%'
        ) / count(*) as frac_time_indeterminate_temp_excursion,
        countif(
            te.temperature_compliance_status = 'out_of_spec'
            and te.temperature_excursion_primary_cause = 'other'
        ) / count(*) as frac_time_other_temp_excursion,
        max(te.ingested_at) as ingested_at
    from telemetry_enriched as te
    inner join location_details as loc
        on te.location_id = loc.location_id
    group by 1, 2, 3, 4, 5, 6
)

select
    {{ dbt_utils.generate_surrogate_key(['j.location_id', 'j.active_fruit_type_code', 'j.reading_hour_utc']) }} as room_conditions_hk,
    j.location_id,
    j.active_fruit_type_code,
    crop.fruit_name as active_fruit_name,
    j.reading_hour_utc,
    j.reading_hour_local,
    j.avg_t_room_c,
    j.avg_rh_room_frac,
    j.stddev_rh_room_frac,
    j.avg_co2_ppm,
    j.avg_o2_pct,
    j.avg_return_supply_air_temp_delta_c,
    j.avg_t_coil_evap_c,
    j.min_t_coil_evap_c,
    j.max_t_coil_evap_c,
    j.avg_evap_air_velocity_m_s,
    j.min_evap_air_velocity_m_s,
    wh.avg_ambient_temp_c,
    wh.avg_ambient_rh_frac,
    wh.avg_ambient_dewpoint_temp_c,
    wh.avg_shortwave_radiation_wm2,
    wh.avg_wind_speed_ms,
    wh.avg_ambient_pressure_pa,
    wh.avg_ground_temperature_c,
    j.frac_time_in_temp_spec * 100.0 as pct_time_in_temp_spec,
    j.frac_time_in_temp_spec_excl_defrost * 100.0 as pct_time_in_temp_spec_excl_defrost,
    j.frac_time_temp_unknown * 100.0 as pct_time_temp_unknown,
    j.frac_time_temp_downtime * 100.0 as pct_time_temp_downtime,
    j.frac_time_defrost * 100.0 as pct_time_defrost,
    j.defrost_cycles_per_hour,
    j.frac_time_low_p_switch_tripped * 100.0 as pct_time_low_p_switch_tripped,
    j.low_p_trips_per_hour,
    j.frac_time_internal_door_open * 100.0 as pct_time_internal_door_open,
    j.frac_time_external_door_open * 100.0 as pct_time_external_door_open,
    j.frac_time_any_door_open * 100.0 as pct_time_any_door_open,
    j.internal_door_openings_per_hour,
    j.external_door_openings_per_hour,
    (j.internal_door_openings_per_hour + j.external_door_openings_per_hour) as door_openings_per_hour,
    coalesce(eh.excursion_count, 0) as temperature_excursion_count,
    j.frac_time_temp_excursion * 3600.0 as total_temperature_excursion_duration_sec,
    eh.avg_excursion_duration_sec as avg_temperature_excursion_duration_sec,
    j.frac_time_defrost_temp_excursion * 3600.0 as temperature_excursion_duration_sec_defrost,
    j.frac_time_door_temp_excursion * 3600.0 as temperature_excursion_duration_sec_door,
    j.frac_time_low_p_trip_temp_excursion * 3600.0 as temperature_excursion_duration_sec_low_p_trip,
    j.frac_time_downtime_temp_excursion * 3600.0 as temperature_excursion_duration_sec_downtime,
    j.frac_time_indeterminate_temp_excursion * 3600.0 as temperature_excursion_duration_sec_indeterminate,
    j.frac_time_other_temp_excursion * 3600.0 as temperature_excursion_duration_sec_other,
    coalesce(eh.excursion_count_defrost, 0) as temperature_excursion_count_defrost,
    coalesce(eh.excursion_count_door, 0) as temperature_excursion_count_door,
    coalesce(eh.excursion_count_low_p_trip, 0) as temperature_excursion_count_low_p_trip,
    coalesce(eh.excursion_count_downtime, 0) as temperature_excursion_count_downtime,
    coalesce(eh.excursion_count_indeterminate, 0) as temperature_excursion_count_indeterminate,
    coalesce(eh.excursion_count_other, 0) as temperature_excursion_count_other,
    dh.avg_setpoint_recovery_sec as avg_setpoint_recovery_sec_after_door_close,
    coalesce(dh.disruptive_door_close_count, 0) as disruptive_door_close_count,
    coalesce(dh.door_close_with_recovery_count, 0) as door_close_with_recovery_count,
    dfh.avg_setpoint_recovery_sec as avg_setpoint_recovery_sec_after_defrost,
    coalesce(dfh.disruptive_defrost_end_count, 0) as disruptive_defrost_end_count,
    coalesce(dfh.defrost_end_with_recovery_count, 0) as defrost_end_with_recovery_count,
    lph.avg_setpoint_recovery_sec as avg_setpoint_recovery_sec_after_lp_trip_reset,
    coalesce(lph.disruptive_lp_trip_reset_count, 0) as disruptive_lp_trip_reset_count,
    coalesce(lph.lp_trip_reset_with_recovery_count, 0) as lp_trip_reset_with_recovery_count,

    -- Metadata & Partitions
    j.region,
    j.country_name,
    j.iana_timezone,
    date(j.reading_hour_utc) as reading_date_utc,
    greatest(
        j.ingested_at,
        coalesce(eh.ingested_at, j.ingested_at),
        coalesce(dh.ingested_at, j.ingested_at),
        coalesce(dfh.ingested_at, j.ingested_at),
        coalesce(lph.ingested_at, j.ingested_at),
        coalesce(wh.ingested_at, j.ingested_at)
    ) as ingested_at,
    current_timestamp() as dbt_updated_at
from joined as j
left join temp_excursion_by_hour as eh
    on
        j.location_id = eh.location_id
        and j.reading_hour_utc = eh.excursion_hour_utc
left join door_recovery_by_hour as dh
    on
        j.location_id = dh.location_id
        and j.reading_hour_utc = dh.recovery_hour_utc
left join defrost_recovery_by_hour as dfh
    on
        j.location_id = dfh.location_id
        and j.reading_hour_utc = dfh.recovery_hour_utc
left join lp_trip_recovery_by_hour as lph
    on
        j.location_id = lph.location_id
        and j.reading_hour_utc = lph.recovery_hour_utc
left join weather_by_hour as wh
    on
        j.location_id = wh.location_id
        and j.reading_hour_utc = wh.weather_hour_utc
left join crop_variety as crop
    on j.active_fruit_type_code = crop.fruit_type_code
