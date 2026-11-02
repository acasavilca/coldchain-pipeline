{{ config(materialized='table') }}

with room_temperature_compliance_readings as (
    select
        location_id,
        event_timestamp,
        t_room_c,
        active_fruit_type_code,
        refrigeration_target_t_setpoint_c,
        temperature_compliance_status,
        ingested_at
    from {{ ref('int_room_temperature_compliance_readings') }}
),

door_close_events as (
    select
        location_id,
        event_timestamp,
        -- event_timestamp as door_close_ts,
        max(row_duration_msec) as row_duration_msec,
        logical_or(opening_has_anomaly) as location_has_anomaly,
        max(ingested_at) as ingested_at
    from {{ ref('int_door_events_enriched') }}
    where is_open = false
    group by 1, 2
),

unioned as (
    select
        location_id,
        event_timestamp as event_ts,
        null as door_close_ts,
        temperature_compliance_status,
        -- null as is_any_door_open,
        'reading' as row_type,
        ingested_at
    from room_temperature_compliance_readings

    union all

    select
        location_id,
        event_timestamp as event_ts,
        event_timestamp as door_close_ts,
        null as temperature_compliance_status,
        -- is_any_door_open,
        'door_event' as row_type,
        ingested_at
    from door_close_events
),

forward_filled as (
    select
        location_id,
        event_ts,
        temperature_compliance_status,
        row_type,
        ingested_at,
        last_value(door_close_ts ignore nulls) over (
            partition by location_id
            order by
                event_ts,
                case when row_type = 'door_event' then 0 else 1 end
            rows between unbounded preceding and current row
        ) as door_close_ts
    from unioned
),

door_close_disrupted as (
    select distinct
        location_id,
        door_close_ts
    from forward_filled
    where
        row_type = 'reading'
        and event_ts >= door_close_ts
        and temperature_compliance_status = 'out_of_spec'
)

select
    ff.location_id,
    ff.door_close_ts,
    min(case when ff.temperature_compliance_status = 'in_spec' then ff.event_ts end) as recovery_ts,
    max(ff.ingested_at) as ingested_at
from forward_filled as ff
inner join door_close_disrupted as dcd
    on
        ff.location_id = dcd.location_id
        and ff.door_close_ts = dcd.door_close_ts
where
    ff.row_type = 'reading'
    and ff.event_ts > ff.door_close_ts
group by 1, 2
