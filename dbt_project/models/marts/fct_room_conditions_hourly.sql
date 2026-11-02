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
    {% set lower_bound_sql %}
        select cast(
            -- timestamp_sub(
                timestamp(date(min(ts))) -- ,
            --     interval 1 day) 
            as string
        ) as min_ts
        from (
            select quantized_event_ts as ts
            from {{ ref('int_facility_events_forward_filled') }}
            where ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
            {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
            {% endif %}

            union all

            select phase_start_ts as ts
            from {{ ref('int_temperature_excursion_events_enriched') }}
            where ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
            {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
            {% endif %}

            union all

            select door_close_ts as ts
            from {{ ref('int_door_recovery_events') }}
            where ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
            {% else %}
                (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
            {% endif %}
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
    select *
    from {{ ref('int_facility_events_forward_filled') }}
    {% if is_incremental() %}
        where quantized_event_ts >= timestamp('{{ min_ts }}')
    {% endif %}
),

temp_excursion_by_hour as (
    select
        location_id,
        timestamp_trunc(phase_start_ts, hour) as excursion_hour_utc,
        sum(total_phase_duration_msec) / 1000.0 as total_excursion_duration_sec,
        avg(total_phase_duration_msec) / 1000.0 as avg_excursion_duration_sec,
        count(*) as excursion_count,
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
        count(*) as door_close_count,
        countif(recovery_ts is not null) as door_close_with_recovery_count,
        max(ingested_at) as ingested_at
    from {{ ref('int_door_recovery_events') }}
    {% if is_incremental() %}
        where door_close_ts >= timestamp('{{ min_ts }}')
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
        safe_divide(
            countif(te.temperature_compliance_status = 'in_spec'),
            countif(te.temperature_compliance_status != 'downtime')
        ) as frac_time_in_temp_spec,
        countif(te.temperature_compliance_status = 'unknown') / count(*) as frac_time_temp_unknown,
        countif(te.temperature_compliance_status = 'downtime') / count(*) as frac_time_temp_downtime,
        max(te.ingested_at) as ingested_at
    from telemetry_enriched as te
    inner join location_details as loc
        on te.location_id = loc.location_id
    group by 1, 2, 3, 4, 5, 6
)

select
    {{ dbt_utils.generate_surrogate_key(['j.location_id', 'j.reading_hour_utc']) }} as room_conditions_hk,
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
    j.frac_time_in_temp_spec * 100.0 as pct_time_in_temp_spec,
    j.frac_time_temp_unknown * 100.0 as pct_time_temp_unknown,
    j.frac_time_temp_downtime * 100.0 as pct_time_temp_downtime,
    coalesce(eh.excursion_count, 0) as temperature_excursion_count,
    eh.total_excursion_duration_sec as total_temperature_excursion_duration_sec,
    eh.avg_excursion_duration_sec as avg_temperature_excursion_duration_sec,
    dh.avg_setpoint_recovery_sec as avg_setpoint_recovery_sec_after_door_close,
    coalesce(dh.door_close_count, 0) as door_close_count,
    coalesce(dh.door_close_with_recovery_count, 0) as door_close_with_recovery_count,

    -- Metadata & Partitions
    j.region,
    j.country_name,
    j.iana_timezone,
    date(j.reading_hour_utc) as reading_date_utc,
    greatest(
        j.ingested_at,
        coalesce(eh.ingested_at, j.ingested_at),
        coalesce(dh.ingested_at, j.ingested_at)
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
left join crop_variety as crop
    on j.active_fruit_type_code = crop.fruit_type_code
