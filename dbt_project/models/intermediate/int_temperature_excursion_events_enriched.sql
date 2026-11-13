{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
      "field": "phase_end_ts",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "excursion_id", "temperature_compliance_status", "phase_seq"]
) }}

{% if is_incremental() and execute %}
    {% set watermark = ingest_watermark(this, 'phase_end_ts', hours=6) %}
    {% set lower_bound_sql %}
        select cast(timestamp(date(min(event_timestamp))) as string) as min_ts
        from {{ ref('int_room_temperature_compliance_readings') }}
        where ingested_at >= timestamp('{{ watermark }}')
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

with source_slice as (
    select
        location_id,
        event_timestamp,
        temperature_compliance_status,
        defrost_phase_code,
        defrost_cycle_has_missing_rows,
        open_door_count,
        low_p_trip_count,
        is_low_p_switch_tripped,
        low_p_trip_episode_anomalous,
        ingested_at
        -- low_p_trip_episode_id,
        -- defrost_termination_reason_code,
        -- last_door_close_ts,
        -- downtime_phase_name,
        -- defrost_cycle_id,
    from {{ ref('int_room_temperature_compliance_readings') }}
    {% if is_incremental() %}
        where event_timestamp >= timestamp('{{ min_ts }}')
    {% endif %}
),

seed_rows as (
    {% if is_incremental() %}
        select
            location_id,
            ingested_at,
            excursion_id as seed_excursion_id,
            phase_seq as seed_phase_seq,
            phase_start_ts as seed_phase_start_ts,
            temperature_compliance_status as seed_compliance_status,
            start_defrost_phase_code as seed_defrost_phase_code,
            start_defrost_cycle_has_missing_rows as seed_defrost_cycle_has_missing_rows,
            start_open_door_count as seed_open_door_count,
            start_low_p_trip_count as seed_low_p_trip_count,
            start_is_low_p_switch_tripped as seed_is_low_p_switch_tripped,
            start_low_p_trip_episode_anomalous as seed_low_p_trip_episode_anomalous,
            prev_phase_status as seed_prev_phase_status,
            overlaps_defrost as seed_overlaps_defrost,
            overlaps_door as seed_overlaps_door,
            overlaps_low_p_trip as seed_overlaps_low_p_trip,
            coalesce(last_known_status, temperature_compliance_status) as seed_known_compliance_status
        from {{ this }}
        where
            phase_start_ts < timestamp('{{ min_ts }}')
            and phase_end_ts >= timestamp('{{ min_ts }}')
        qualify row_number() over (partition by location_id order by phase_start_ts desc) = 1
    {% else %}
        select
            location_id,
            ingested_at,
            cast(null as int64) as seed_excursion_id,
            cast(null as int64) as seed_phase_seq,
            cast(null as timestamp) as seed_phase_start_ts,
            cast(null as string) as seed_compliance_status,
            cast(null as int64) as seed_defrost_phase_code,
            cast(null as bool) as seed_defrost_cycle_has_missing_rows,
            cast(null as int64) as seed_open_door_count,
            cast(null as int64) as seed_low_p_trip_count,
            cast(null as bool) as seed_is_low_p_switch_tripped,
            cast(null as bool) as seed_low_p_trip_episode_anomalous,
            cast(null as string) as seed_prev_phase_status,
            cast(null as bool) as seed_overlaps_defrost,
            cast(null as bool) as seed_overlaps_door,
            cast(null as bool) as seed_overlaps_low_p_trip,
            cast(null as string) as seed_known_compliance_status
        from source_slice
        where false
    {% endif %}
),

enriched_slice as (
    select
        slice.location_id,
        slice.event_timestamp,
        slice.temperature_compliance_status,
        slice.defrost_phase_code,
        slice.defrost_cycle_has_missing_rows,
        slice.open_door_count,
        slice.low_p_trip_count,
        slice.is_low_p_switch_tripped,
        slice.low_p_trip_episode_anomalous,
        seed.seed_compliance_status,
        seed.seed_known_compliance_status,
        seed.seed_excursion_id,
        seed.seed_phase_seq,
        slice.temperature_compliance_status in ('in_spec', 'out_of_spec') as include_flag,
        slice.ingested_at as ingested_at
        -- greatest(seed.ingested_at, slice.ingested_at) as ingested_at
    from source_slice as slice
    left join seed_rows as seed
        on slice.location_id = seed.location_id
),

detected_states as (
    select
        location_id,
        ingested_at,
        event_timestamp,
        temperature_compliance_status,
        defrost_phase_code,
        defrost_cycle_has_missing_rows,
        open_door_count,
        low_p_trip_count,
        is_low_p_switch_tripped,
        low_p_trip_episode_anomalous,
        include_flag,
        seed_excursion_id,
        seed_phase_seq,
        seed_compliance_status,
        seed_known_compliance_status,
        -- Detect state change (Returns true for first row or when status flips)
        coalesce(
            last_value(case when include_flag then temperature_compliance_status end ignore nulls) over (
                partition by location_id order by event_timestamp
                rows between unbounded preceding and 1 preceding
            ),
            seed_known_compliance_status
        ) as prev_row_known_status,
        coalesce(
            lag(temperature_compliance_status) over (partition by location_id order by event_timestamp),
            seed_compliance_status
        ) as prev_row_status
    from enriched_slice
),

flagged_states as (
    select
        *,
        -- A valid state change only occurs if current row is not null and it differs from the last known state
        -- include_flag and coalesce(temperature_compliance_status != prev_known_state, true) as excursion_state_changed
        (temperature_compliance_status is distinct from prev_row_status) as status_changed,
        (temperature_compliance_status is distinct from prev_row_known_status) as known_status_changed
    from detected_states
),

excursion_blocks as (
    select
        *,
        coalesce(seed_excursion_id, 0) + sum(
            case when include_flag and known_status_changed and temperature_compliance_status = 'out_of_spec' then 1 else 0 end
        ) over w as excursion_id,
        coalesce(seed_phase_seq, 0) + sum(
            case when status_changed then 1 else 0 end
        ) over w as phase_seq,
        lead(event_timestamp) over w as next_event_timestamp
        -- timestamp_diff(
        --     lead(event_timestamp) over w,
        --     event_timestamp,
        --     millisecond
        -- ) as row_duration_msec
    from flagged_states
    window
        w as (partition by location_id order by event_timestamp)
),

aggregate as (
    select
        e.location_id,
        e.excursion_id,
        e.temperature_compliance_status,
        e.phase_seq,
        any_value(e.seed_known_compliance_status) as seed_known_compliance_status,
        greatest(max(e.ingested_at), coalesce(max(s.ingested_at), max(e.ingested_at))) as ingested_at,
        -- Capture the exact boundaries of this specific status phase within the excursion block
        least(min(e.event_timestamp), coalesce(max(s.seed_phase_start_ts), min(e.event_timestamp))) as phase_start_ts,
        coalesce(max(e.next_event_timestamp), max(e.event_timestamp)) as phase_end_ts,
        -- Total duration of the actual out-of-spec violation within this block
        -- sum(row_duration_msec) as total_phase_duration_msec,
        -- Flag if this specific block contains any unknown data gaps
        logical_or(e.temperature_compliance_status = 'unknown') as excursion_block_has_gaps,
        any_value(s.seed_defrost_phase_code) as seed_defrost_phase_code,
        any_value(s.seed_defrost_cycle_has_missing_rows) as seed_defrost_cycle_has_missing_rows,
        any_value(s.seed_open_door_count) as seed_open_door_count,
        any_value(s.seed_low_p_trip_count) as seed_low_p_trip_count,
        any_value(s.seed_is_low_p_switch_tripped) as seed_is_low_p_switch_tripped,
        any_value(s.seed_low_p_trip_episode_anomalous) as seed_low_p_trip_episode_anomalous,
        any_value(s.seed_overlaps_defrost) as seed_overlaps_defrost,
        any_value(s.seed_overlaps_door) as seed_overlaps_door,
        any_value(s.seed_overlaps_low_p_trip) as seed_overlaps_low_p_trip,
        any_value(s.seed_prev_phase_status) as seed_prev_phase_status,
        array_agg(
            struct(
                e.defrost_phase_code,
                e.defrost_cycle_has_missing_rows,
                e.open_door_count,
                e.low_p_trip_count,
                e.is_low_p_switch_tripped,
                e.low_p_trip_episode_anomalous
            )
            order by e.event_timestamp asc
            limit 1
        )[safe_offset(0)] as earliest_row_in_phase_block,
        logical_or(coalesce(e.defrost_phase_code between 1 and 5, false)) as slice_overlaps_defrost,
        logical_or(coalesce(e.open_door_count > 0, false)) as slice_overlaps_door,
        logical_or(coalesce(e.low_p_trip_count > 0, false)) as slice_overlaps_low_p_trip,
        logical_or(s.seed_phase_seq is not null) as is_straddling
    from excursion_blocks as e
    left join seed_rows as s
        on
            e.location_id = s.location_id
            and e.phase_seq = s.seed_phase_seq
    group by 1, 2, 3, 4
),

resolving as (
    select
        * except (
            seed_defrost_phase_code,
            seed_defrost_cycle_has_missing_rows,
            seed_open_door_count,
            seed_low_p_trip_count,
            seed_is_low_p_switch_tripped,
            seed_low_p_trip_episode_anomalous,
            seed_overlaps_defrost,
            seed_overlaps_door,
            seed_prev_phase_status,
            earliest_row_in_phase_block,
            slice_overlaps_defrost,
            slice_overlaps_door,
            slice_overlaps_low_p_trip,
            is_straddling
        ),
        if(is_straddling, seed_defrost_phase_code, earliest_row_in_phase_block.defrost_phase_code) as start_defrost_phase_code,
        if(is_straddling, seed_defrost_cycle_has_missing_rows, earliest_row_in_phase_block.defrost_cycle_has_missing_rows) as start_defrost_cycle_has_missing_rows,
        if(is_straddling, seed_open_door_count, earliest_row_in_phase_block.open_door_count) as start_open_door_count,
        if(is_straddling, seed_low_p_trip_count, earliest_row_in_phase_block.low_p_trip_count) as start_low_p_trip_count,
        if(is_straddling, seed_is_low_p_switch_tripped, earliest_row_in_phase_block.is_low_p_switch_tripped) as start_is_low_p_switch_tripped,
        if(is_straddling, seed_low_p_trip_episode_anomalous, earliest_row_in_phase_block.low_p_trip_episode_anomalous) as start_low_p_trip_episode_anomalous,
        (coalesce(seed_overlaps_defrost, false) or slice_overlaps_defrost) as overlaps_defrost,
        (coalesce(seed_overlaps_door, false) or slice_overlaps_door) as overlaps_door,
        (coalesce(seed_overlaps_low_p_trip, false) or slice_overlaps_low_p_trip) as overlaps_low_p_trip,
        coalesce(
            lag(temperature_compliance_status) over (partition by location_id order by phase_seq),
            seed_prev_phase_status
        ) as prev_phase_status
    from aggregate
),

cause as (
    select
        *,
        case
            when temperature_compliance_status != 'out_of_spec' then null
            when prev_phase_status = 'downtime' then 'downtime'
            when start_low_p_trip_count > 0 and not start_low_p_trip_episode_anomalous then 'low_p_trip'
            when start_low_p_trip_count > 0 then 'indeterminate_low_p_trip_suspected'
            when start_defrost_phase_code between 1 and 5 and not start_defrost_cycle_has_missing_rows then 'defrost'
            when start_defrost_phase_code between 1 and 5 then 'indeterminate_defrost_suspected'
            when start_open_door_count > 0 then 'door'
            else 'other'
        end as primary_cause
    from resolving
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'excursion_id', 'phase_seq']) }} as temperature_excursion_hk,
    location_id,
    excursion_id,
    phase_seq,
    phase_start_ts,
    phase_end_ts,
    timestamp_diff(phase_end_ts, phase_start_ts, millisecond) as total_phase_duration_msec,
    temperature_compliance_status,
    excursion_block_has_gaps,
    ingested_at,
    start_defrost_phase_code,
    start_defrost_cycle_has_missing_rows,
    start_open_door_count,
    start_low_p_trip_count,
    start_is_low_p_switch_tripped,
    start_low_p_trip_episode_anomalous,
    overlaps_defrost,
    overlaps_door,
    overlaps_low_p_trip,
    prev_phase_status,
    primary_cause,
    coalesce(
        last_value(nullif(temperature_compliance_status, 'unknown') ignore nulls) over (
            partition by location_id order by phase_start_ts
            rows between unbounded preceding and current row
        ),
        seed_known_compliance_status
    ) as last_known_status
from cause
