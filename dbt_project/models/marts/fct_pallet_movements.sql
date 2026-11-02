{{ config(
    materialized='incremental',
    incremental_strategy='insert_overwrite',
    partition_by={
        'field': 'arrival_date_utc',
        'data_type': 'date',
        'granularity': 'day'
    },
    cluster_by=['location_id', 'batch_id', 'fruit_type_code']
) }}

with location_details as (
    select
        location_id,
        region,
        country_name,
        iana_timezone
    from {{ ref('dim_geographic_details') }}
),

crop_variety as (
    select
        fruit_type_code,
        fruit_name
    from {{ ref('dim_crop_variety') }}
),

new_rows as (
    select arrival_ts
    from {{ ref('int_pallet_history_enriched') }}
    {% if is_incremental() %}
        where
            ingested_at >= {% if var('cutoff_override', none) %}
                timestamp('{{ var("cutoff_override") }}')
        {% else %}
            (select timestamp_sub(max(t.ingested_at), interval 3 hour) from {{ this }} as t)
        {% endif %}
    {% endif %}
),

bounds as (
    select
        timestamp(date(min(arrival_ts))) as min_ts,
        timestamp(date_add(date(max(arrival_ts)), interval 1 day)) as max_ts
    from new_rows
),

staged_pallet_history as (
    select *
    from {{ ref('int_pallet_history_enriched') }}
    {% if is_incremental() %}
        where
            arrival_ts >= (select min_ts from bounds)
            and arrival_ts < (select max_ts from bounds)
    {% endif %}
),

filtered_source as (
    select
        ph.*,
        loc.region,
        loc.country_name,
        loc.iana_timezone,
        crop.fruit_name,
        date(ph.arrival_ts) as arrival_date_utc
    from staged_pallet_history as ph
    left join location_details as loc
        on ph.location_id = loc.location_id
    left join crop_variety as crop
        on ph.fruit_type_code = crop.fruit_type_code
)

select
    {{ dbt_utils.generate_surrogate_key(['location_id', 'batch_id', "coalesce(dispatch_sequence_num, 0)"]) }} as pallet_movement_hk,
    location_id,
    batch_id,
    fruit_type_code,
    fruit_name,

    -- Timestamps (UTC)
    arrival_ts,
    dispatch_ts,
    arrival_supervisor_intervention_ts,
    dispatch_supervisor_intervention_ts,
    -- Localized Datetimes for Operations Reporting
    datetime(arrival_ts, iana_timezone) as arrival_local_dt,
    datetime(dispatch_ts, iana_timezone) as dispatch_local_dt,
    datetime(arrival_supervisor_intervention_ts, iana_timezone) as arrival_supervisor_intervention_local_dt,
    datetime(dispatch_supervisor_intervention_ts, iana_timezone) as dispatch_supervisor_intervention_local_dt,

    -- Dwell Metrics
    pallet_dwell_time_ms,
    pallet_dwell_time_ms / 3600000.0 as pallet_dwell_time_hours,
    dispatch_sequence_num,
    total_dispatch_count_for_batch,
    is_batch_fully_closed_out_now,
    is_pallet_fully_dispatched,

    -- Mass Metrics
    arrival_gross_fruit_mass_kg,
    arrival_net_fruit_mass_est_kg,
    gross_fruit_mass_moved_kg,
    dispatched_net_fruit_mass_est_kg,
    cumulative_dispatched_net_mass_est_kg,
    batch_total_net_mass_loss_est_kg,

    -- Quality Metrics
    quality_grade_at_arrival,
    quality_grade_at_dispatch,
    quality_grade_decrease,
    hours_past_shelf_life,
    hours_past_target_retention_limit,

    -- Metadata & Partitions
    region,
    country_name,
    iana_timezone,
    arrival_date_utc, -- System UTC Date used strictly for BigQuery partition boundaries
    ingested_at,
    current_timestamp() as dbt_updated_at
from filtered_source
