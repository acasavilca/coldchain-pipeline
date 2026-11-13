{{ config(materialized='table') }}

{% set sensor_tags = var('telemetry_sensor_tags') %}

{% set min_interval_msec = get_min_polling_interval_ms(sensor_tags) %}

with compressor_events_detailed as (
    select
        location_id,
        cycle_id,
        ingested_at,
        {{ quantize_timestamp_ms('phase_start_ts', min_interval_msec) }} as quantized_event_ts,
        compressor_status,
        is_active,
        switch_reason_code,
        cycle_has_anomaly
    from {{ ref('int_compressor_events_detailed') }}
),

humidifier_events_enriched as (
    select
        location_id,
        cycle_id,
        ingested_at,
        {{ quantize_timestamp_ms('event_timestamp', min_interval_msec) }} as quantized_event_ts,
        num_discs_on,
        num_clusters_on,
        max_humid_capacity_utilized_pct,
        is_humidifying,
        cycle_has_anomaly
    from {{ ref('int_humidifier_events_enriched') }}
),

door_events_pivoted as (
    select
        location_id,
        {{ quantize_timestamp_ms('event_timestamp', min_interval_msec) }} as quantized_event_ts,
        max(ingested_at) as ingested_at,
        any_value(case when door_id = 0 then opening_id end) as internal_opening_id,
        any_value(case when door_id = 1 then opening_id end) as external_opening_id,
        any_value(case when door_id = 0 then is_open end) as is_internal_door_open,
        any_value(case when door_id = 1 then is_open end) as is_external_door_open,
        any_value(case when door_id = 0 then opening_has_anomaly end) as internal_opening_anomalous,
        any_value(case when door_id = 1 then opening_has_anomaly end) as external_opening_anomalous
    from {{ ref('int_door_events_enriched') }}
    group by 1, 2
),

crop_rotation_enriched as (
    select
        location_id,
        ingested_at,
        {{ quantize_timestamp_ms('event_timestamp', min_interval_msec) }} as quantized_event_ts,
        fruit_type_code,
        rotation_is_anomalous
    from {{ ref('int_crop_rotation_events_enriched') }}
),

refrigerant_rotation_enriched as (
    select
        location_id,
        ingested_at,
        {{ quantize_timestamp_ms('event_timestamp', min_interval_msec) }} as quantized_event_ts,
        refrigerant_code,
        rotation_is_anomalous
    from {{ ref('int_refrigerant_rotation_events_enriched') }}
),

temperature_excursion_enriched as (
    select
        location_id,
        excursion_id,
        ingested_at,
        {{ quantize_timestamp_ms('phase_start_ts', min_interval_msec) }} as quantized_event_ts,
        temperature_compliance_status,
        excursion_block_has_gaps,
        primary_cause
    from {{ ref('int_temperature_excursion_events_enriched') }}
    qualify row_number() over (
        partition by location_id, {{ quantize_timestamp_ms('phase_start_ts', min_interval_msec) }}
        order by phase_start_ts desc, phase_seq desc
    ) = 1
),

downtime_enriched as (
    select
        location_id,
        downtime_seq_id,
        ingested_at,
        {{ quantize_timestamp_ms('phase_start_ts', min_interval_msec) }} as quantized_event_ts,
        {{ quantize_timestamp_ms('phase_end_ts', min_interval_msec) }} as quantized_event_end_ts,
        phase_name
    from {{ ref('int_downtime_enriched') }}
),

downtime_filtered as (
    select
        location_id,
        downtime_seq_id,
        ingested_at,
        quantized_event_ts,
        phase_name
    from downtime_enriched
    where
        quantized_event_ts is not null
        and (quantized_event_ts < quantized_event_end_ts or quantized_event_end_ts is null)
),

defrost_enriched as (
    select
        location_id,
        defrost_cycle_id,
        ingested_at,
        {{ quantize_timestamp_ms('phase_start_ts', min_interval_msec) }} as quantized_event_ts,
        phase_start_ts,
        defrost_phase_code,
        termination_reason_code,
        has_missing_rows,
        if(defrost_phase_code = 0 and defrost_cycle_id > 0, phase_start_ts, null) as last_defrost_end_ts
    from {{ ref('int_defrost_events_enriched') }}
),

lp_trip_enriched as (
    select
        location_id,
        ingested_at,
        {{ quantize_timestamp_ms('event_timestamp', min_interval_msec) }} as quantized_event_ts,
        low_p_trip_episode_id,
        is_low_p_switch_tripped,
        low_p_trip_episode_anomalous,
        if(not is_low_p_switch_tripped, event_timestamp, null) as last_lp_trip_switch_off_ts
    from {{ ref('int_lp_trip_events_enriched') }}
)

select
    c.cycle_id as compressor_cycle_id,
    c.is_active as is_compressor_active,
    c.compressor_status,
    c.switch_reason_code as compressor_switch_reason_code,
    c.cycle_has_anomaly as compressor_cycle_anomalous,
    h.cycle_id as humidifier_cycle_id,
    h.is_humidifying as is_humidifier_active,
    h.num_discs_on,
    h.num_clusters_on,
    h.max_humid_capacity_utilized_pct,
    h.cycle_has_anomaly as humidifier_cycle_anomalous,
    d.internal_opening_id,
    d.external_opening_id,
    d.is_internal_door_open,
    d.is_external_door_open,
    d.internal_opening_anomalous,
    d.external_opening_anomalous,
    cr.fruit_type_code as active_fruit_type_code,
    cr.rotation_is_anomalous as crop_rotation_anomalous,
    r.refrigerant_code as active_refrigerant_code,
    r.rotation_is_anomalous as refrigerant_rotation_anomalous,
    te.excursion_id,
    te.temperature_compliance_status,
    te.excursion_block_has_gaps,
    te.primary_cause as temperature_excursion_primary_cause,
    dw.downtime_seq_id,
    dw.phase_name as downtime_phase,
    df.defrost_cycle_id,
    df.phase_start_ts as defrost_phase_start_ts,
    df.defrost_phase_code,
    df.termination_reason_code as defrost_termination_reason_code,
    df.has_missing_rows as defrost_cycle_has_missing_rows,
    df.last_defrost_end_ts,
    lpt.low_p_trip_episode_id,
    lpt.is_low_p_switch_tripped,
    lpt.low_p_trip_episode_anomalous,
    lpt.last_lp_trip_switch_off_ts,
    nullif(
        greatest(
            coalesce(c.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(h.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(d.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(cr.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(r.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(te.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(dw.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(df.ingested_at, timestamp('1970-01-01 00:00:00')),
            coalesce(lpt.ingested_at, timestamp('1970-01-01 00:00:00'))
        ),
        timestamp('1970-01-01 00:00:00')
    ) as ingested_at,
    coalesce(c.location_id, h.location_id, d.location_id, cr.location_id, r.location_id, te.location_id, dw.location_id, df.location_id, lpt.location_id) as location_id,
    coalesce(c.quantized_event_ts, h.quantized_event_ts, d.quantized_event_ts, cr.quantized_event_ts, r.quantized_event_ts, te.quantized_event_ts, dw.quantized_event_ts, df.quantized_event_ts, lpt.quantized_event_ts) as quantized_event_ts
from compressor_events_detailed as c
full outer join humidifier_events_enriched as h
    on c.location_id = h.location_id and c.quantized_event_ts = h.quantized_event_ts
full outer join door_events_pivoted as d
    on
        coalesce(c.location_id, h.location_id) = d.location_id
        and coalesce(c.quantized_event_ts, h.quantized_event_ts) = d.quantized_event_ts
full outer join crop_rotation_enriched as cr
    on
        coalesce(c.location_id, h.location_id, d.location_id) = cr.location_id
        and coalesce(c.quantized_event_ts, h.quantized_event_ts, d.quantized_event_ts) = cr.quantized_event_ts
full outer join refrigerant_rotation_enriched as r
    on
        coalesce(c.location_id, h.location_id, d.location_id, cr.location_id) = r.location_id
        and coalesce(c.quantized_event_ts, h.quantized_event_ts, d.quantized_event_ts, cr.quantized_event_ts) = r.quantized_event_ts
full outer join temperature_excursion_enriched as te
    on
        coalesce(c.location_id, h.location_id, d.location_id, cr.location_id, r.location_id) = te.location_id
        and coalesce(c.quantized_event_ts, h.quantized_event_ts, d.quantized_event_ts, cr.quantized_event_ts, r.quantized_event_ts) = te.quantized_event_ts
full outer join downtime_filtered as dw
    on
        coalesce(c.location_id, h.location_id, d.location_id, cr.location_id, r.location_id, te.location_id) = dw.location_id
        and coalesce(c.quantized_event_ts, h.quantized_event_ts, d.quantized_event_ts, cr.quantized_event_ts, r.quantized_event_ts, te.quantized_event_ts) = dw.quantized_event_ts
full outer join defrost_enriched as df
    on
        coalesce(c.location_id, h.location_id, d.location_id, cr.location_id, r.location_id, te.location_id, dw.location_id) = df.location_id
        and coalesce(c.quantized_event_ts, h.quantized_event_ts, d.quantized_event_ts, cr.quantized_event_ts, r.quantized_event_ts, te.quantized_event_ts, dw.quantized_event_ts) = df.quantized_event_ts
full outer join lp_trip_enriched as lpt
    on
        coalesce(c.location_id, h.location_id, d.location_id, cr.location_id, r.location_id, te.location_id, dw.location_id, df.location_id) = lpt.location_id
        and coalesce(c.quantized_event_ts, h.quantized_event_ts, d.quantized_event_ts, cr.quantized_event_ts, r.quantized_event_ts, te.quantized_event_ts, dw.quantized_event_ts, df.quantized_event_ts) = lpt.quantized_event_ts
