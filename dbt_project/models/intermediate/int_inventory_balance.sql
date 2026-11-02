{{ config(materialized='table') }}

with pallet_events as (
    select
        location_id,
        arrival_ts as event_ts,
        arrival_net_fruit_mass_est_kg as mass_delta_kg,
        arrival_box_count as box_delta,
        'arrival' as event_type
    from {{ ref('int_pallet_history_enriched') }}
    where
        arrival_ts is not null
        and (dispatch_sequence_num = 1 or dispatch_sequence_num is null)  -- one arrival row per batch, not per dispatch

    union all

    select
        location_id,
        dispatch_ts as event_ts,
        -1 * dispatched_net_fruit_mass_est_kg as mass_delta_kg,
        -1 * box_count_moved as box_delta,
        'dispatch' as event_type
    from {{ ref('int_pallet_history_enriched') }}
    where dispatch_ts is not null
)


select
    location_id,
    event_ts,
    event_type,
    mass_delta_kg,
    box_delta,
    sum(mass_delta_kg) over (
        partition by location_id
        order by
            event_ts,
            case when event_type = 'arrival' then 0 else 1 end
        rows between unbounded preceding and current row
    ) as running_mass_balance_kg,
    sum(box_delta) over (
        partition by location_id
        order by
            event_ts,
            case when event_type = 'arrival' then 0 else 1 end
        rows between unbounded preceding and current row
    ) as running_box_balance
from pallet_events
