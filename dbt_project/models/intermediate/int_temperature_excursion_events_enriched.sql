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
    {% set lower_bound_sql %}
        select cast(timestamp(date(min(event_timestamp))) as string) as min_ts
        from {{ ref('int_room_temperature_compliance_readings') }}
        where ingested_at >= {% if var('cutoff_override', none) %}
            timestamp('{{ var("cutoff_override") }}')
        {% else %}
            (select timestamp_sub(max(t.ingested_at), interval 6 hour) from {{ this }} as t)
        {% endif %}
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
        ingested_at
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
            -- event_timestamp,
            -- temperature_compliance_status,
            excursion_id as seed_excursion_id,
            phase_seq as seed_phase_seq,
            phase_start_ts as seed_phase_start_ts,
            temperature_compliance_status as seed_compliance_status,
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
            -- event_timestamp,
            -- temperature_compliance_status,
            cast(null as int64) as seed_excursion_id,
            cast(null as int64) as seed_phase_seq,
            cast(null as timestamp) as seed_phase_start_ts,
            cast(null as string) as seed_compliance_status,
            cast(null as string) as seed_known_compliance_status
        from source_slice where false
    {% endif %}
),

enriched_slice as (
    select
        slice.location_id,
        slice.event_timestamp,
        slice.temperature_compliance_status,
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
        logical_or(e.temperature_compliance_status = 'unknown') as excursion_block_has_gaps
    from excursion_blocks as e
    left join seed_rows as s
        on
            e.location_id = s.location_id
            and e.phase_seq = s.seed_phase_seq
            -- and e.temperature_compliance_status = s.temperature_compliance_status
    group by e.location_id, e.excursion_id, e.temperature_compliance_status, e.phase_seq
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
    coalesce(
        last_value(nullif(temperature_compliance_status, 'unknown') ignore nulls) over (
            partition by location_id order by phase_start_ts
            rows between unbounded preceding and current row
        ),
        seed_known_compliance_status
    ) as last_known_status
from aggregate
