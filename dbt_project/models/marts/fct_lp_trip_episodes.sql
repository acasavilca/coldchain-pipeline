{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "low_p_trip_episode_start_ts",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "low_p_trip_episode_id"]
) }}

{% if is_incremental() and execute %}
    {% set checkpoint_table = ensure_checkpoint_table(ref('int_lp_trip_recovery_events')) %}
    {% set watermark = ingest_watermark(checkpoint_table, 'checkpoint_day', hours=6, ts_type='date') %}

    {% set raw_lower_bound_sql %}
        select cast(timestamp(date(min(ts))) as string) as raw_min_ts
        from (
            select event_timestamp as ts
            from {{ ref('int_lp_trip_events_enriched') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select lp_trip_switch_off_ts as ts
            from {{ ref('int_lp_trip_recovery_events') }}
            where ingested_at >= timestamp('{{ watermark }}')

            union all

            select quantized_event_ts as ts
            from {{ ref('int_telemetry_derived') }}
            where ingested_at >= timestamp('{{ watermark }}')
        )
    {% endset %}
    {% set b = run_query(raw_lower_bound_sql) %}
    {% set raw_min_ts = b.columns[0].values()[0] %}
    {% if raw_min_ts is none %}
        {% set raw_min_ts = '2100-01-01 00:00:00+00:00' %}
    {% endif %}
    {% if var('min_ts_override', none) %}
        {% set raw_min_ts = var('min_ts_override') %}
    {% endif %}

    {% set true_lower_bound_sql %}
       select cast(timestamp(date(min(low_p_trip_episode_start_ts))) as string) as min_ts
       from {{ ref('int_lp_trip_events_enriched') }}
       where
            low_p_trip_episode_id > 0
            and event_timestamp >= timestamp('{{ raw_min_ts }}')
    {% endset %}

    {% set a = run_query(true_lower_bound_sql) %}
    {% set min_ts = a.columns[0].values()[0] %}
    {% if min_ts is none %}
        {% set min_ts = raw_min_ts %}
    {% endif %}
{% endif %}

with equipment_constants as (
    select
        refrigerant_code,
        compressor_model_code,
        refrigerant_name,
        lp_trip_cut_out_bar,
        lp_trip_cut_in_bar
    from {{ ref('seed_compressor_rating_conditions') }}
),

lp_trip_events_collapsed as (
    select
        location_id,
        low_p_trip_episode_id,
        any_value(low_p_trip_episode_start_ts) as low_p_trip_episode_start_ts,
        any_value(
            if(
                is_low_p_switch_tripped,
                struct(event_timestamp as start_ts, next_event_timestamp as end_ts),
                null
            )
        ) as trip_on,
        any_value(
            if(
                not is_low_p_switch_tripped,
                struct(event_timestamp as start_ts, next_event_timestamp as end_ts),
                null
            )
        ) as trip_off,
        logical_or(low_p_trip_episode_anomalous) as low_p_trip_episode_anomalous,
        logical_or(orphan_episode) as orphan_episode,
        max(ingested_at) as ingested_at
    from {{ ref('int_lp_trip_events_enriched') }}
    {% if is_incremental() %}
        where low_p_trip_episode_start_ts >= timestamp('{{ min_ts }}')
    {% endif %}
    group by 1, 2
    having logical_or(is_low_p_switch_tripped)
),

recovery_events as (
    select
        location_id,
        low_p_trip_episode_id,
        recovery_ts,
        ingested_at,
        timestamp_diff(recovery_ts, lp_trip_switch_off_ts, second) as recovery_duration_sec
    from {{ ref('int_lp_trip_recovery_events') }}
    {% if is_incremental() %}
        where lp_trip_switch_off_ts >= timestamp('{{ min_ts }}')
    {% endif %}
),

telemetry_collapsed as (
    select
        location_id,
        low_p_trip_episode_id,
        min_by(
            struct(
                active_refrigerant_code,
                t_coil_evap_c,
                t_coil_evap_c_fill_status,
                p_suction_line_bar,
                p_suction_line_bar_fill_status,
                evap_air_velocity_m_s,
                airflow_ratio,
                velocity_over_trip_ratio,
                superheat_k,
                t_evap_c,
                minutes_since_last_defrost
            ),
            quantized_event_ts
        ) as telemetry_at_trip_start,
        max_by(
            struct(
                active_refrigerant_code,
                t_coil_evap_c,
                t_coil_evap_c_fill_status,
                p_suction_line_bar,
                p_suction_line_bar_fill_status,
                evap_air_velocity_m_s,
                airflow_ratio,
                velocity_over_trip_ratio,
                superheat_k,
                t_evap_c,
                minutes_since_last_defrost
            ),
            quantized_event_ts
        ) as telemetry_at_trip_end,
        max(ingested_at) as ingested_at
    from {{ ref('int_telemetry_derived') }}
    where
        is_low_p_switch_tripped
        {% if is_incremental() %}
            and quantized_event_ts >= timestamp('{{ min_ts }}')
        {% endif %}
    group by 1, 2
),

seed_rows as (
    {% if is_incremental() %}
        select
            location_id,
            low_p_trip_episode_start_ts
        from {{ this }}
        where low_p_trip_episode_start_ts < timestamp('{{ min_ts }}')
        qualify row_number() over (partition by location_id order by low_p_trip_episode_start_ts desc) = 1
    {% else %}
        select
            location_id,
            low_p_trip_episode_start_ts
        from lp_trip_events_collapsed
        where false
    {% endif %}
),

unioned as (
    select
        location_id,
        cast(null as int64) as low_p_trip_episode_id,
        low_p_trip_episode_start_ts,
        cast(null as struct<start_ts timestamp, end_ts timestamp>) as trip_on,
        cast(null as struct<start_ts timestamp, end_ts timestamp>) as trip_off,
        cast(null as bool) as low_p_trip_episode_anomalous,
        cast(null as bool) as orphan_episode,
        cast(null as timestamp) as ingested_at
    from seed_rows

    union all

    select
        location_id,
        low_p_trip_episode_id,
        low_p_trip_episode_start_ts,
        trip_on,
        trip_off,
        low_p_trip_episode_anomalous,
        orphan_episode,
        ingested_at
    from lp_trip_events_collapsed
),

lp_trip_episodes_with_intervals as (
    select
        *,
        timestamp_diff(
            low_p_trip_episode_start_ts,
            lag(low_p_trip_episode_start_ts) over (partition by location_id order by low_p_trip_episode_start_ts),
            second
        ) as low_p_trip_interval_sec
    from unioned
    {% if is_incremental() %}
        qualify low_p_trip_episode_start_ts >= timestamp('{{ min_ts }}')
    {% endif %}
)

select
    lpt.* except (trip_on, trip_off, ingested_at),
    r.* except (location_id, low_p_trip_episode_id, ingested_at),
    lpt.trip_on.start_ts as trip_on_start_ts,
    lpt.trip_on.end_ts as trip_on_end_ts,
    lpt.trip_off.start_ts as trip_off_start_ts,
    lpt.trip_off.end_ts as trip_off_end_ts,
    t.telemetry_at_trip_start.t_coil_evap_c as t_coil_evap_c_at_trip_start,
    t.telemetry_at_trip_start.t_coil_evap_c_fill_status as t_coil_evap_c_at_trip_start_fill_status,
    t.telemetry_at_trip_end.t_coil_evap_c as t_coil_evap_c_at_trip_end,
    t.telemetry_at_trip_end.t_coil_evap_c_fill_status as t_coil_evap_c_at_trip_end_fill_status,
    t.telemetry_at_trip_start.p_suction_line_bar as p_suction_line_bar_at_trip_start,
    t.telemetry_at_trip_start.p_suction_line_bar_fill_status as p_suction_line_bar_at_trip_start_fill_status,
    t.telemetry_at_trip_end.p_suction_line_bar as p_suction_line_bar_at_trip_end,
    t.telemetry_at_trip_end.p_suction_line_bar_fill_status as p_suction_line_bar_at_trip_end_fill_status,
    t.telemetry_at_trip_start.evap_air_velocity_m_s as evap_air_velocity_m_s_at_trip_start,
    t.telemetry_at_trip_start.airflow_ratio as airflow_ratio_at_trip_start,
    t.telemetry_at_trip_end.airflow_ratio as airflow_ratio_at_trip_end,
    t.telemetry_at_trip_start.velocity_over_trip_ratio as velocity_over_trip_ratio_at_trip_start,
    t.telemetry_at_trip_end.velocity_over_trip_ratio as velocity_over_trip_ratio_at_trip_end,
    t.telemetry_at_trip_start.superheat_k as superheat_k_at_trip_start,
    t.telemetry_at_trip_end.superheat_k as superheat_k_at_trip_end,
    t.telemetry_at_trip_start.t_evap_c as t_evap_c_at_trip_start,
    t.telemetry_at_trip_start.minutes_since_last_defrost as minutes_since_last_defrost_at_trip_start,
    eq_start.lp_trip_cut_out_bar as low_p_trip_cut_out_bar,
    eq_end.lp_trip_cut_in_bar as low_p_trip_cut_in_bar,
    eq_start.refrigerant_code as active_refrigerant_code_at_trip_start,
    eq_end.refrigerant_code as active_refrigerant_code_at_trip_end,
    (r.low_p_trip_episode_id is not null) as caused_temperature_disruption,
    (lpt.trip_off.start_ts is null and not coalesce(lpt.orphan_episode, false)) as is_trip_open,
    (t.telemetry_at_trip_start.p_suction_line_bar - eq_start.lp_trip_cut_out_bar) as p_suction_line_bar_deviation_at_trip_start,
    (t.telemetry_at_trip_end.p_suction_line_bar - eq_end.lp_trip_cut_in_bar) as p_suction_line_bar_deviation_at_trip_end,
    timestamp_diff(lpt.trip_on.end_ts, lpt.trip_on.start_ts, second) as trip_on_duration_sec,
    timestamp_diff(lpt.trip_off.end_ts, lpt.trip_off.start_ts, second) as trip_off_duration_sec,
    timestamp_diff(lpt.trip_off.end_ts, lpt.trip_on.start_ts, second) as trip_total_duration_sec,
    nullif(
        greatest(
            coalesce(lpt.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(r.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(t.ingested_at, timestamp('1970-01-01 00:00:00'))
        ),
        timestamp('1970-01-01 00:00:00')
    ) as ingested_at
from lp_trip_episodes_with_intervals as lpt
left join recovery_events as r
    on
        lpt.location_id = r.location_id
        and lpt.low_p_trip_episode_id = r.low_p_trip_episode_id
left join telemetry_collapsed as t
    on
        lpt.location_id = t.location_id
        and lpt.low_p_trip_episode_id = t.low_p_trip_episode_id
left join equipment_constants as eq_start
    on t.telemetry_at_trip_start.active_refrigerant_code = eq_start.refrigerant_code
left join equipment_constants as eq_end
    on t.telemetry_at_trip_end.active_refrigerant_code = eq_end.refrigerant_code
