{{ config(
    materialized='table',
    partition_by={
      "field": "phase_start_ts",
      "data_type": "timestamp",
      "granularity": "day"
    },
    cluster_by=["location_id", "defrost_cycle_id", "defrost_phase_code"]
) }}

with unpivoted_defrost_params as (
    select
        p.location_id,
        item.phase_code,
        item.max_expected_duration_sec,
        p.compressor_evaluation_interval_sec as tolerance_sec
    from {{ ref('dim_equipment_design') }} as p
    cross join unnest([
        struct(0 as phase_code, defrost_interval_sec as max_expected_duration_sec),
        struct(1 as phase_code, compressor_minimum_active_time_sec as max_expected_duration_sec),
        struct(2 as phase_code, defrost_settle_sec as max_expected_duration_sec),
        struct(3 as phase_code, defrost_max_heat_sec as max_expected_duration_sec),
        struct(4 as phase_code, defrost_drip_sec as max_expected_duration_sec),
        struct(5 as phase_code, defrost_fan_delay_max_sec as max_expected_duration_sec)
    ]) as item
),

defrost_events as (
    select
        location_id,
        defrost_phase_code,
        event_timestamp,
        ingested_at,
        termination_reason_code
    from {{ ref('stg_defrost_events') }}
),

detected_states as (
    select
        e.location_id,
        e.ingested_at,
        e.event_timestamp,
        e.defrost_phase_code,
        e.termination_reason_code,
        -- Detect state change:
        -- use lead to later evaluate is next timestamp and termination_reason_code can be taken by current phase
        coalesce(
            (e.defrost_phase_code != lead(e.defrost_phase_code, 1) over w),
            true
        ) as state_changed_flag,
        coalesce(
            (lead(e.defrost_phase_code, 1) over w = mod(e.defrost_phase_code + 1, 6)),
            true
        ) as is_contiguous_with_next,
        -- Record next defrost_phase_code
        lead(e.defrost_phase_code, 1) over w as next_defrost_phase_code,
        -- Record previous defrost_phase_code
        lag(e.defrost_phase_code, 1) over w as prev_defrost_phase_code,
        -- Define gap_exceeds_limit flag
        coalesce(
            (timestamp_diff(
                lead(e.event_timestamp) over w,
                e.event_timestamp,
                millisecond
            ) / 1000.0 > (p.max_expected_duration_sec + p.tolerance_sec)),
            false
        ) as gap_exceeds_limit
    from defrost_events as e
    left join unpivoted_defrost_params as p
        on
            e.location_id = p.location_id
            and e.defrost_phase_code = p.phase_code
    window
        w as (
            partition by e.location_id order by e.event_timestamp
            -- rows between unbounded preceding and current row
        )
),

detected_states_extra_flag as (
    select
        *,
        coalesce(
            lag(gap_exceeds_limit) over (partition by location_id order by event_timestamp),
            false
        ) as prev_gap_exceeds_limit
    from detected_states
),

defrost_blocks as (
    select
        location_id,
        ingested_at,
        event_timestamp as phase_start_ts,
        defrost_phase_code,
        -- is_contiguous_with_next,
        -- gap_exceeds_limit,
        sum(
            case
                when
                    defrost_phase_code = 1
                    or (defrost_phase_code = prev_defrost_phase_code and defrost_phase_code != 0)
                    or (prev_defrost_phase_code = 0 and defrost_phase_code != 0)
                    or (prev_defrost_phase_code is null and defrost_phase_code != 0)
                    or prev_gap_exceeds_limit
                    -- or (defrost_phase_code = prev_defrost_phase_code and defrost_phase_code != 0)
                    -- or (defrost_phase_code != 0 and prev_defrost_phase_code is null)
                    -- or (defrost_phase_code = 1 and next_defrost_phase_code is null)
                    -- or (defrost_phase_code < prev_defrost_phase_code and prev_defrost_phase_code not in (0, 5))
                    -- or gap_exceeds_limit
                    then 1
                else 0
            end
        ) over (
            partition by location_id order by event_timestamp
            rows between unbounded preceding and current row
        ) as defrost_cycle_id,
        not is_contiguous_with_next and not (
            next_defrost_phase_code = 1
            or (defrost_phase_code = 0 and next_defrost_phase_code != 0)
            or gap_exceeds_limit
        ) as break_inside_id,
        case
            when is_contiguous_with_next and not gap_exceeds_limit then lead(event_timestamp) over w
        end as phase_end_ts,
        case
            when defrost_phase_code != 3 then null
            when next_defrost_phase_code is null then -1 -- Last row, 'ongoing'
            when is_contiguous_with_next and not gap_exceeds_limit then lead(termination_reason_code) over w
            else -2 -- 'missing'
        end as termination_reason_code
    from detected_states_extra_flag
    window
        w as (
            partition by location_id order by event_timestamp
            -- rows between unbounded preceding and current row
        )
),

flagged_defrost_blocks as (
    select
        location_id,
        defrost_cycle_id,
        defrost_phase_code,
        ingested_at,
        phase_start_ts,
        phase_end_ts,
        termination_reason_code,
        break_inside_id,
        logical_or(defrost_phase_code between 1 and 5) over (
            partition by location_id, defrost_cycle_id
        ) as has_any_defrost_row,
        defrost_cycle_id = max(defrost_cycle_id) over (partition by location_id) as is_latest_cycle
        -- logical_or(not is_contiguous_with_next) over (partition by location_id, defrost_cycle_id) as has_missing_rows
    from defrost_blocks
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'defrost_cycle_id']) }} as defrost_cycle_hk,
    location_id,
    defrost_cycle_id,
    defrost_phase_code,
    phase_start_ts,
    phase_end_ts,
    termination_reason_code,
    has_any_defrost_row,
    (
        logical_or(break_inside_id) over w
        or (has_any_defrost_row and not logical_or(defrost_phase_code = 1) over w)
        or (has_any_defrost_row and not is_latest_cycle and not logical_or(defrost_phase_code = 0) over w)
    ) as has_missing_rows,
    min(phase_start_ts) over w as defrost_cycle_start_ts,
    max(ingested_at) over w as ingested_at
from flagged_defrost_blocks
window
    w as (partition by location_id, defrost_cycle_id)
